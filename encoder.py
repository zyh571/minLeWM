import torch
from torch import nn
import math

"""
The encoder is implemented as a Vision Transformer (ViT) [34]. Unless otherwise specified, we
use the tiny configuration (∼5M parameters) with a patch size of 14, 12 layers, 3 attention heads,
and hidden dimensions of 192. The observation embedding zt is constructed from the [CLS] token
embedding of the last layer, followed by a projection step. The projection step maps the [CLS] token
embedding into a new representation space using a 1-layer MLP with Batch Normalization [35]. This
step is necessary because the final ViT layer applies a Layer Normalization [36], which prevents our
anti-collapse objective from being optimized effectively.
"""

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wproj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        # B, nh, N, dh
        q = self.Wq(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        # B, nh, N, N
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        x = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, N, self.d_model)
        x = self.Wproj(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # assume [B, N, D]
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps) 
        x = self.gamma * x + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model),
        )
    def forward(self, x):
        x = self.ffn(x)
        return x
    
class ViTBlock(nn.Module):
    def __init__(self, d_model, num_heads, drop):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, drop)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.norm1(x)
        h = self.drop(self.attn(h))
        x = x + h
        h = self.norm2(x)
        h = self.drop(self.mlp(h))
        x = x + h
        return x

class ViT(nn.Module):
    def __init__(self, img_channels=3, img_size=224, patch_size=14, 
                 d_model=192, num_heads=3, num_layers=12, drop=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_channels = img_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size) ** 2
        self.d_model = d_model
        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.cls_tok = nn.Parameter(
            torch.randn(d_model)
        )
        self.pos_emb = nn.Parameter(
            torch.randn(self.num_patches+1, d_model)
        )
        self.blocks = nn.ModuleList(
            [ViTBlock(d_model, num_heads, drop) for l in range(num_layers)]
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.conv(x)
        x = x.reshape(B, self.d_model, self.num_patches).transpose(1, 2)
        x = torch.cat([self.cls_tok.expand(B, 1, -1), x], dim=1)
        x = self.drop(x + self.pos_emb)
        for block in self.blocks:
            x = block(x)

        return x

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    vit = ViT()
    print(vit(x).shape)
