import torch
from torch import nn

def make_quadrature(lam=1.0, T=16, t_min=0.2, t_max=4.0, device='cpu'):
    """Build quadrature knots and weights for the Epps-Pulley integral.
    
    Returns:
        t_knots : (T,) frequency knots, uniformly spaced in [t_min, t_max]
        weights : (T,) trapezoidal weights * Gaussian frequency weight w(t_k)
    """
    t_knots = torch.linspace(t_min, t_max, T, device=device)
    dt = t_knots[1] - t_knots[0]
    alpha = torch.full((T,), dt, device=device)         # trapezoidal weights
    alpha[0]  *= 0.5               # endpoints get half weight
    alpha[-1] *= 0.5
    return t_knots, alpha * torch.exp(-t_knots ** 2 / (2 * lam ** 2))

def random_unit_vectors(M, D, device='cpu'):
    """Sample M points uniformly from S^{D-1}."""
    u = torch.randn((M, D), device=device)
    u /= torch.norm(u, dim=1, keepdim=True)
    return u

def sigreg(Z, M=1024, lam=1.0, T=16, rng=None):
    """SIGReg = average Epps-Pulley statistic over M random projections.
    
    Z   : (N, D) batch of embeddings
    M   : number of random projection directions
    lam : Gaussian-weight bandwidth
    T   : number of quadrature knots
    """
    device = Z.device
    t_knots, weights = make_quadrature(lam=lam, T=T, device=device)
    U = random_unit_vectors(M, Z.shape[1], device)         # (M, D)
    H = U @ Z.T                                          # (M, N): M projections
    
    # Vectorized Epps-Pulley across all M projections
    angles = t_knots[None, :, None] * H[:, None, :]      # (M, T, N)
    C = torch.cos(angles).mean(dim=2)                      # (M, T)
    S = torch.sin(angles).mean(dim=2)                      # (M, T)
    G = torch.exp(-t_knots ** 2 / 2)                        # (T,)
    diff_sq = (C - G[None, :]) ** 2 + S ** 2             # (M, T)
    return Z.shape[0] * (weights * diff_sq).sum(dim=1).mean()  # scalar


class LeWMLoss(nn.Module):
    def __init__(self, reg_weight=0.1):
        super().__init__()
        self.reg_weight = reg_weight
    
    def forward(self, z_pred, z_target, z):
        B, T, D = z.shape
        pred_loss = (z_pred - z_target).pow(2).mean()
        sigreg_loss = sigreg(z.reshape(B*T, D))
        loss = pred_loss + self.reg_weight * sigreg_loss
        return {
            'loss': loss, 
            'pred_loss': pred_loss, 
            'sigreg_loss': sigreg_loss
        }

