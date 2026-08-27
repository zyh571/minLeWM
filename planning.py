import torch
from einops import rearrange

from dataset.pusht_dset import ACTION_MEAN, ACTION_STD

"""
Planning solver. For planning, we use the Cross-Entropy Method (CEM). At each planning step,
CEM samples 300 candidate action sequences and optimizes them for a maximum of 30 iterations
in PushT and 10 iterations in the other environments. At each iteration, the top 30 trajectories are
retained to update the sampling distribution, and the initial sampling variance is set to 1. The planning
horizon is set to 5 steps, which corresponds to 25 environment timesteps due to the use of a frame skip
of 5. We employ a receding-horizon Model Predictive Control (MPC) scheme with a horizon of 5,
meaning that the entire optimized action sequence is executed before replanning. This configuration
follows the setup used in [18].
"""

@torch.no_grad()
def cross_entropy_method(model, o_goal, o_init, a_init,
                         H=5, N=300, K=30, T=30,
                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Args:
        model: LeWorldModel
        o_init: [M, C, H, W] initial history of observations where M is the timestep dim
        actions_init: [M-1, A] initial history of actions
        o_goal: [C, H, W] goal observation at the end of H
        H: planning horizon
        N: number of samples
        K: number of elites
        T: number of iterations
    """
    model.to(device)
    o_init = o_init.to(device)
    a_init = a_init.to(device)
    o_goal = o_goal.to(device)
    model.eval()

    M, img_c, img_h, img_w = o_init.shape
    _, A = a_init.shape
    D = model.d_model

    # initialise sampling distribution parameters 
    plan_mean = torch.zeros(H-1, A, device=device)
    plan_std = torch.ones(H-1, A, device=device)
    plan_best = plan_mean
    cost_best = float('inf')

    # encode observations into latents
    z_init = model.encode(o_init.reshape(1, M, img_c, img_h, img_w)).squeeze(0) # [M, D]
    z_goal = model.encode(o_goal.reshape(1, 1, img_c, img_h, img_w)).squeeze(0) # [1, D]

    z_init = z_init.expand(N, -1, -1)
    a_init = a_init.expand(N, -1, -1)
    for t in range(T):
        z = torch.cat([z_init, torch.zeros(N, H-1, D, device=device)], dim=1)
        sampled_plans = torch.randn(N, H-1, A, device=device) * plan_std + plan_mean
        a = torch.cat([a_init, sampled_plans], dim=1)
        # z: [N, M+H-1, D], a: [N, M+H-2, A]

        # rollout actions in the world model
        for h in range(H-1):
            z_hist = z[:, h:M+h, :]
            a_hist = a[:, h:M+h, :]
            z_next = model.predict(z_hist, a_hist)
            z[:, M+h, :] = z_next[:, -1, :]

        # select elites with lowest costs
        z_horz = z[:, -1, :]
        cost = (z_horz - z_goal).pow(2).sum(dim=-1)
        values, indices = torch.topk(cost, k=K, largest=False)
        elites = sampled_plans[indices]
        plan_mean = elites.mean(dim=0)
        plan_std = elites.std(dim=0)
        if values[0] < cost_best:
            plan_best = elites[0]
            cost_best = values[0]

    return plan_mean, plan_std, plan_best, cost_best



"""
The evaluation budget corresponds to the maximum number of actions the agent is allowed
to execute in the environment. The goal distance determines how far in the future the goal state
is sampled relative to the initial state. During evaluation, trajectories are sampled from the offline
dataset. The initial state is chosen by randomly sampling a state from a trajectory in the dataset,
while the goal state corresponds to a state occurring several timesteps later in the same trajectory.
This ensures that the goal is reachable and consistent with the dataset dynamics. In TwoRoom, the
evaluation budget is set to 50 steps, and the goal state is sampled 25 timesteps in the future. In PushT,
the evaluation budget is 50 steps and the goal is sampled 25 timesteps in the future. In OGBench-Cube
and Reacher, the evaluation budget is 50 steps, and the goal is sampled 25 timesteps in the future.
"""
def preprocess(visual):
    """Env frame (H, W, C) uint8 -> (C, H, W) float in [0,1]"""
    x = torch.from_numpy(visual).float() / 255.0
    return rearrange(x, "h w c -> c h w")

@torch.no_grad()
def model_predictive_control(model, env, o_goal, o_init, a_init, 
                             H=5, N=300, K=30, T=30,
                             budget=50, frameskip=5,
                             action_mean=ACTION_MEAN, action_std=ACTION_STD, transform=None,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Args:
        model, o_goal, o_init, a_init, H, N, K, T: cross_entropy_method arguments
        env: simulation environment set to final o_init state
        budget: maximum allowed environment steps
        frameskip: step interval between the observations given to model
        action_mean, action_std: dataset statistics used to normalise/unnormalise actions
        transform: same dataset transform applied to env visual
    """
    model.to(device)
    o_init = o_init.to(device)
    a_init = a_init.to(device)
    o_goal = o_goal.to(device)
    action_std = action_std.to(device)
    action_mean = action_mean.to(device)
    model.eval()

    steps = 0
    info = {}
    costs = []

    while steps < budget:
        plan, _, _, cost = cross_entropy_method(
            model, o_goal, o_init, a_init, H, N, K, T, device)
        costs.append(cost.item())
        
        for block in plan:
            subs = rearrange(block, "(f d) -> f d", f=frameskip)
            subs = subs * action_std + action_mean

            for act in subs:
                obs, _, _, info = env.step(act.cpu().numpy())
                steps += 1
                if steps >= budget:
                    break

            o_new = preprocess(obs["visual"]).to(device)
            if transform is not None:
                o_new = transform(o_new)
            o_init = torch.cat([o_init[1:], o_new[None]], dim=0)
            a_init = torch.cat([a_init[1:], block[None]], dim=0)

            if steps >= budget:
                break

    return info.get("max_coverage", 0.0), costs