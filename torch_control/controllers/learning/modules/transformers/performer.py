import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False):
        super().__init__()
        self.dim_heads = dim_heads
        self.nb_features = nb_features or dim_heads
        self.ortho_scaling = ortho_scaling
        self.causal = causal

        self.create_projection()

    def create_projection(self):
        projection_matrix = torch.randn(self.nb_features, self.dim_heads)
        q = self.ortho_scaling * projection_matrix
        self.register_buffer('projection_matrix', q)

    def forward(self, q, k, v):
        k = k @ self.projection_matrix.T
        q = q @ self.projection_matrix.T

        if self.causal:
            k_cumsum = k.cumsum(dim=-2)
            D_inv = 1.0 / torch.einsum('...nd,...d->...n', q, k_cumsum).type_as(q)
            context = torch.einsum('...nd,...ne->...de', k, v).type_as(v)
            out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        else:
            k_sum = torch.einsum('...nd,...d->...n', q, k).type_as(q)
            D_inv = 1.0 / k_sum
            context = torch.einsum('...nd,...ne->...de', k, v).type_as(v)
            out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)

        return out

class PerformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, m, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = FastAttention(dim_head, nb_features=m)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.proj_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.proj_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.attn.dim_heads

        qkv = self.proj_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, -1, h).permute(0, 2, 1, 3), qkv)

        out = self.attn(q, k, v)
        out = out.reshape(b, n, -1)
        x = x + self.proj_out(out)
        x = x + self.ff(self.norm2(x))
        return x

class Performer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, m, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PerformerBlock(dim, heads, dim_head, m, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
