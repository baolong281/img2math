import torch
import torch.nn as nn
import lightning as L


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.20):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln_1 = torch.nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_embd, num_heads, dropout)
        self.ln_2 = torch.nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.head(self.ln_2(x))
        return x


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, channels=1, embed_dim=512):
        """
        img size: image shape
        """
        super().__init__()
        h, w = img_size
        assert (
            w % patch_size == 0 and h % patch_size == 0
        ), "image not divisable by patch size"
        self.patch_size = patch_size
        self.n_patches = (h // patch_size) * (w // patch_size)

        self.projection = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        x: batch of images (B, H, W)
        return: (B, num_patches, num_embeddings)
        """
        x = self.projection(x)  # (B, n_embd, n_patches / 2, n_patches / 2)
        x = x.flatten(-2)  # (B, n_embd, n_patches)
        x = x.transpose(-2, -1)  # (B, n_patches, n_embd)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, n_embd, n_heads=8, bias=False, attn_dropout=0.20, proj_dropout=0.20
    ):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd not divisible by num heads"
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        self.dk = self.head_dim**-0.5  # sqrt dk for scaling

        self.kqv = nn.Linear(n_embd, n_embd * 3, bias=bias)
        self.projection = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        """
        x: (B, T, n_embd)
        returns: (B, T, n_embd)
        """

        (
            B,
            T,
            C,
        ) = x.shape  # batch size, num tokens, n_embd
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
        print(aggregated_attention.shape)
        x = aggregated_attention.transpose(1, 2)  # (B, T, n_heads, C)
        x = x.flatten(2)  # (B, T, C)
        x = self.projection(x)  # (B, T, C)
        x = self.proj_dropout(x)
        return x
