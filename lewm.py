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
        # [B, N, C, H, W] -> [B*N, T, D]
        B, N, C, H, W = x.shape
        x = x.reshape(B*N, C, H, W)
        x = self.encoder(x)
        _, _, D = x.shape
        # [B*N, T, D] -> [B*N, D]
        x = x[:, 0, :].squeeze()
        # batchnorm is gay so need [B*N, D]
        x = self.encoder_proj(x).reshape(B, N, D)
        return x
    
    def predict(self, x, a):
        # [B, N, D] -> [B, N, D]
        B, N, D = x.shape
        x = self.predictor(x, a)
        x = x.reshape(B*N, D)
        x = self.predictor_proj(x).reshape(B, N, D)
        return x

    def forward(self, o, a):
        # x: [B, N, C, H, W], a: [B, N, A]
        z = self.encode(o) # [B, N, D]
        z_t = z[:, :-1, :]
        z_next = z[:, 1:, :]

        a_t = a[:, :-1, :]

        z_pred = self.predict(z_t, a_t)
        return z_pred, z_next, z

if __name__ == '__main__':
    o = torch.randn(2, 3+1, 3, 224, 224)
    a = torch.randn(2, 3+1, 2)
    model = LeWorldModel()
    z_pred, z_next, z = model(o, a)
    print(z_pred.shape, z_next.shape, z.shape)
