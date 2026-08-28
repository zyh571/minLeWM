import torch
from torch import nn

from model.encoder import ViT
from model.predictor import AdaLNTransformer
from model.loss import LeWMLoss

"""
We apply a frame-skip of 5, grouping consecutive actions between frames into a single action block.
This choice enables computationally efficient longer-horizon predictions while maintaining informa-
tive temporal transitions. We use a batch size of 128 with sub-trajectories of size 4 corresponding to
4 frames and 4 blocks of 5 actions. Each frame is 224 × 224 pixels
"""

class LeWorldModel(nn.Module):
    def __init__(self, d_model=192,):
        super().__init__()
        self.d_model = d_model
        self.encoder = ViT(
            img_channels=3,
            img_size=224,
            patch_size=14,
            d_model=d_model,
            num_heads=3,
            num_layers=12,
        )
        self.encoder_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
        )
        self.predictor = AdaLNTransformer(
            d_model=d_model,
            d_action=10,
            num_heads=16,
            hist_len=3,
            num_layers=6,
        )
        self.predictor_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
        )
        self.criterion = LeWMLoss(reg_weight=0.1)
    
    def encode(self, x):
        # [B, T, C, H, W] -> [B*T, N, D]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.encoder(x)
        _, _, D = x.shape
        # get cls_token: [B*T, N, D] -> [B*T, D]
        x = x[:, 0, :]
        # batchnorm is gay so need [B*T, D]
        x = self.encoder_proj(x).reshape(B, T, D)
        return x
    
    def predict(self, x, a):
        # [B, T, D] -> [B, T, D]
        B, T, D = x.shape
        x = self.predictor(x, a)
        x = x.reshape(B*T, D)
        x = self.predictor_proj(x).reshape(B, T, D)
        return x

    def forward(self, o, a):
        # x: [B, T, C, H, W], a: [B, T, A]
        z = self.encode(o) # [B, T, D]
        z_pred = self.predict(z[:, :-1, :], a[:, :-1, :])
        z_targ = z[:, 1:, :]
        loss_dict = self.criterion(z_pred, z_targ, z)
        return loss_dict