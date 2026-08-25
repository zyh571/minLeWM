from encoder import ViT
from predictor import AdaLNTransformer

import torch
from torch import nn

class LeWorldModel(nn.Module):
    def __init__(self, d_model=192,):
        super().__init__()
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
    
    def encode(self, x):
        # [B, T, C, H, W] -> [B*T, N, D]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.encoder(x)
        _, _, D = x.shape
        # get cls_token: [B*T, N, D] -> [B*T, D]
        x = x[:, 0, :].squeeze()
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
        return z_pred, z_targ, z

if __name__ == '__main__':
    o = torch.randn(2, 3+1, 3, 224, 224)
    a = torch.randn(2, 3+1, 10)
    model = LeWorldModel()
    z_pred, z_targ, z = model(o, a)
    print(z_pred.shape, z_targ.shape, z.shape)
