import torch
import torch.nn as nn
import lightning as L


class ViT(nn.Module):
    def __init__(self, img_shape, patch_size, n_embd, num_blocks=6, num_heads=8, dropout=.20, chanels=1):
        super().__init__()
        H, W = img_shape
        assert W % patch_size == 0 and H % patch_size == 0, 'image not divisable by patch size'
        self.shape = img_shape
        self.patch_size = patch_size
        self.num_patches = (W // patch_size) * (H // patch_size)

        self.patch_embeddings = PatchEmbeddings(img_shape, patch_size, channels=chanels, n_embd=n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, n_embd))
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(n_embd, num_heads=8) for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embd, eps=1e-5)
        self.head = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        img_patches = torch.cat([self.patch_embeddings(x), cls_tokens], dim=-2)
        enc = img_patches + self.pos_embed
        """
        x: (B, channels, H, W)
        return: (B, n_embd)
        """
        return enc



class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.20, ln_bias=False):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln_1 = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attention = SelfAttention(n_embd, num_heads, dropout)
        self.ln_2 = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.head = MLP(n_embd, bias=False)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.head(self.ln_2(x))
        return x


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.20, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = nn.GELU(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class PatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, channels=1, n_embd=512):
        """
        img size: image shape
        """
        super().__init__()
        H, W = img_size
        assert (
            W % patch_size == 0 and H % patch_size == 0
        ), "image not divisable by patch size"
        self.patch_size = patch_size
        self.n_patches = (H // patch_size) * (W // patch_size)

        self.projection = nn.Conv2d(
            channels,
            n_embd,
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
        print(aggregated_attention.shape)
        x = aggregated_attention.transpose(1, 2)  # (B, T, n_heads, C)
        x = x.flatten(2)  # (B, T, C)
        x = self.projection(x)  # (B, T, C)
        x = self.proj_dropout(x)
        return x
