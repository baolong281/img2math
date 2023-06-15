import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout, ln_bias=False):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln_1 = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attention = SelfAttention(n_embd, num_heads, dropout)
        self.ln_2 = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.head = MLP(n_embd, dropout, bias=False)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.head(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 2 * n_embd, bias=True)
        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, n_embd, num_heads, dropout, bias=False
    ):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd not divisible by num heads"
        self.n_heads = num_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // num_heads
        self.dk = self.head_dim**-0.5  # sqrt dk for scaling

        self.kqv = nn.Linear(n_embd, n_embd * 3, bias=bias)
        self.projection = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (B, T, n_embd)
        """

        B, T, C = x.shape  # batch size, num tokens, n_embd
        assert C == self.n_embd, "input size does not equal n_embd"

        kqv = self.kqv(x)  # (B, T, n_embd*3)
        kqv = kqv.reshape(B, T, 3, self.n_heads, self.head_dim)
        kqv = kqv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        k, q, v = kqv  # (B, n_heads, T, head_dim)

        attention = (
            q @ k.transpose(-1, -2)
        ) * self.dk  # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        attention = attention.softmax(dim=-1)  # (B, n_heads, T, T)
        attention = self.attn_dropout(attention)
        aggregated_attention = (
            attention @ v
        )  # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        x = aggregated_attention.transpose(1, 2)  # (B, T, n_heads, C)
        x = x.flatten(2)  # (B, T, C)
        x = self.projection(x)  # (B, T, C)
        x = self.proj_dropout(x)
        return x
