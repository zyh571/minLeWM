import torch
from torch import nn
import math

"""
The predictor is a transformer with 6 layers, 16 attention heads, and 10% dropout (∼10M parameters).
Actions are incorporated into the predictor through Adaptive Layer Normalization (AdaLN) [37]
applied at each layer. The AdaLN parameters are initialized to zero to stabilize training and ensure
that action conditioning impacts the predictor training progressively. The predictor takes as input a
history of N frame representations and predicts the next frame representation auto-regressively with
temporal causal masking to avoid looking at future embeddings. The predictor is also followed by a
projector network with the same implementation as the one used for the encoder.
"""

class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads, hist_len, drop):
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
        mask = torch.triu(torch.full((hist_len, hist_len), float('-inf')), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, D = x.shape
        # B, nh, T, dh
        q = self.Wq(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        # B, nh, T, T
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores + self.mask[None, None, :T, :T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        x = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, self.d_model)
        x = self.Wproj(x)

        return x

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, d_action, eps=1e-6):
        super().__init__()
        self.gamma = nn.Linear(d_action, d_model)
        self.beta = nn.Linear(d_action, d_model)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.ones_(self.beta.bias)
        self.eps = eps

    def forward(self, x, a):
        # x: [B, T, D], a: [B, T, A]
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = self.gamma(a) * x + self.beta(a)
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
    
class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model, d_action, num_heads, hist_len, drop):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(d_model, d_action)
        self.attn = CausalAttention(d_model, num_heads, hist_len, drop)
        self.norm2 = AdaptiveLayerNorm(d_model, d_action)
        self.mlp = FeedForward(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x, a):
        h = self.norm1(x, a)
        h = self.drop(self.attn(h))
        x = x + h
        h = self.norm2(x, a)
        h = self.drop(self.mlp(h))
        x = x + h
        return x

class AdaLNTransformer(nn.Module):
    def __init__(self, d_model=192, d_action=10, num_heads=16, hist_len=3, num_layers=6, drop=0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(hist_len, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(d_model, d_action, num_heads, hist_len, drop) for l in range(num_layers)]
        )
        self.drop = nn.Dropout(drop)
    def forward(self, x, a):
        x = self.drop(x + self.pos_emb)
        for block in self.blocks:
            x = block(x, a)
        return x
