import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderTransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout, ln_bias=False):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attention = EncoderAttention(n_embd, num_heads, dropout)
        self.head = MLP(n_embd, dropout, bias=False)

    def forward(self, x):
        x = x + self.attention(self.ln(x))
        x = x + self.head(self.ln(x))
        return x

class DecoderTransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout, ln_bias=False):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attention = DecoderAttention(n_embd, num_heads, dropout)
        self.head = MLP(n_embd, dropout, bias=False)

    def forward(self, q, k, v):
        q, k, v = self.ln(q), self.ln(k), self.ln(v)
        x = q + self.attention(q, k, v)
        x = x + self.head(x)
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


class EncoderAttention(nn.Module):
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

class DecoderAttention(nn.Module):
    ''' 
    Multi-Head Attention module 
    from: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/SubLayers.py#L9
    '''


    def __init__(self, n_embd, n_head, dropout=0.75):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd not divisable by head size"

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.w_qs = nn.Linear(n_embd, n_embd, bias=False)
        self.w_ks = nn.Linear(n_embd, n_embd, bias=False)
        self.w_vs = nn.Linear(n_embd, n_embd, bias=False)
        self.fc = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_embd, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.head_dim)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.head_dim)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.head_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q = F.scaled_dot_product_attention(q, k, v, mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        q = self.layer_norm(q)

        return q
