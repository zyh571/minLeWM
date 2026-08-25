import torch
from model.lewm import LeWorldModel

# Cross Entropy Method + receding-horizon Model Predictive Control for one env

# TODO:
# add min std clamp
# add momentum for action statistics
@torch.no_grad()
def cross_entropy_method(model, obs, actions, obs_goal, 
                         num_iter=30, num_sample=300, num_top=30, horizon=5, #frame skip of 5, so 25 env timesteps
                         device='cuda' if torch.cuda.is_available() else 'cpu'): 
    # obs: [N, C, H, W], actions: [N-1, A], og: [C, H, W]
    # returns optimised actions
    model.to(device)
    obs = obs.to(device)
    actions = actions.to(device)
    obs_goal = obs_goal.to(device)

    model.eval()

    N, C, H, W = obs.shape
    _, A = actions.shape

    # encode o1 to get z1: [N, D], og to get zg: [D]
    z_init = model.encode(obs)
    z_goal = model.encode(obs_goal.reshape(1, 1, C, H, W)).squeeze()
    D = len(z_goal)

    a_mean = torch.zeros(horizon-1, A).to(device)
    a_std = torch.ones(horizon-1, A).to(device)
    z_init = z_init.expand(num_sample, N, D)
    a_init = actions.expand(num_sample, N-1, A)

    for i in range(num_iter):
        # randomly initialise multiple a1:H based on current distribution
        sampled_a = torch.randn(num_sample, horizon-1, A).to(device) * a_std + a_mean
        # predict z2 to zH --> H-1 predictions
        z = torch.empty(num_sample, N+horizon-1, D).to(device)
        z[:, :N, :] = z_init
        a = torch.cat([a_init, sampled_a], dim=1) # [num_sample, N+horizon-2, A]
      
        for t in range(0, horizon-1):
            z_hist = z[:, t:N+t, :]
            a_hist = a[:, t:N+t, :]
            z_next = model.predict(z_hist, a_hist)
            z[:, N+t, :] = z_next[:, -1, :]    
    
        # [B, D]
        z_horizon = z[:, -1, :]

        # minimise terminal latent goal-matching objective
        # take top a1:H with min cost(zH, zg) 
        # update distribution parameters
        cost = (z_horizon - z_goal).pow(2).sum(-1)
        values, indices = torch.topk(-cost, k=num_top)
        top_a = sampled_a[indices]
        a_mean = top_a.mean(dim=0)
        a_std = top_a.std(dim=0)

    return a_mean, a_std

# interacts with the environment?
def model_predictive_control():
    pass