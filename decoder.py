import torch
from torch import nn
import math

"""
Decoder (Visualization Only). For visualization, we decode the [CLS] token embedding (192 dim)
from the last encoder layer into an image using a lightweight transformer decoder. The [CLS]
representation is first projected to a hidden dimension and used as the key and value in cross-attention.
A fixed set of learnable query tokens, one for each patch of the target image, interacts with this global
representation through several cross-attention layers with residual MLP blocks. For an image of size
224 × 224 with patch size 16, this corresponds to P = (224/16)^2 = 196 learnable query tokens. The
resulting patch embeddings are then linearly projected to 16 × 16 × 3 pixel patches and rearranged to
produce a 224 × 224 RGB image. This decoder is used only as a diagnostic tool to visualize what
visual information is retained in the [CLS] representation.
"""

class CrossAttention(nn.Module):
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

    def forward(self, q, cls):
        B, T, D = q.shape # T = 196 if 224x224 img with patch 16
        # B, nh, T, dh
        q = self.Wq(q).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.Wk(cls).reshape(B, 1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.Wv(cls).reshape(B, 1, self.num_heads, self.d_head).transpose(1, 2)
        # B, nh, T, 1
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        x = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, self.d_model)
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

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, drop):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = CrossAttention(d_model, num_heads, drop)
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, q, cls):
        h = self.norm1(q)
        h = self.drop(self.attn(h, cls))
        q = q + h
        h = self.norm2(q)
        h = self.drop(self.mlp(q))
        q = q + h
        return q

class DecoderTransformer(nn.Module):
    def __init__(self, img_channels=3, img_size=224, patch_size=16, 
                 d_model=192, num_heads=3, num_layers=12, drop=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_channels = img_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size//patch_size
        self.num_patches = self.grid_size ** 2
        self.d_model = d_model
        
        self.q_toks = nn.Parameter(
            torch.randn(self.num_patches, d_model)
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, drop) for l in range(num_layers)]
        )
        self.convt = nn.ConvTranspose2d(
            in_channels=d_model,
            out_channels=img_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
    def forward(self, cls):
        B, D = cls.shape
        q = self.q_toks.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, cls)
        q = q.reshape(B, self.grid_size, self.grid_size, self.d_model).permute(0, 3, 1, 2)
        q = self.convt(q)
        return q

if __name__ == '__main__':
    cls = torch.randn(2, 192)
    decoder = DecoderTransformer()
    print(decoder(cls).shape)
