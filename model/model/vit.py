from typing import Any
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torch.optim import Adam
import wandb
from model.transformer import EncoderTransformerBlock, MLP


class ViT(L.LightningModule):
    def __init__(self, img_shape, patch_size, n_embd, num_blocks=6, num_heads=8, dropout=.75, channels=1, output_classes=2, encoder=True, lr=1e-4):
        super().__init__()
        H, W = img_shape
        assert W % patch_size == 0 and H % patch_size == 0, 'image not divisable by patch size'
        self.lr = lr
        self.shape = img_shape
        self.patch_size = patch_size
        self.num_patches = (W // patch_size) * (H // patch_size)
        self.encoder = encoder
        self.output_classes = output_classes
        self.pos_embd_len = self.num_patches + 1 if not encoder else self.num_patches
        self.patch_embeddings = PatchEmbeddings(
            img_shape, patch_size, channels=channels, n_embd=n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.pos_embd_len, n_embd))
        self.pos_dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                EncoderTransformerBlock(n_embd, num_heads, dropout) for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embd, eps=1e-5)
        self.loss = nn.BCELoss()
        self.accuracy = Accuracy(
            task='binary', num_classes=1).to(self.device)

        if not encoder:
            self.head = nn.Sequential(
                MLP(n_embd, dropout), nn.Linear(n_embd, output_classes))
        else:
            self.head = nn.Linear(n_embd, n_embd)

        self.save_hyperparameters()

    def forward(self, x):
        """
        x: (B, channels, H, W)
        return: (B, n_embd)
        """
        B = x.shape[0]

        if not self.encoder:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            img_patches = torch.cat(
                [self.patch_embeddings(x), cls_tokens], dim=-2)
        else:
            img_patches = self.patch_embeddings(x)

        enc = img_patches + self.pos_embed
        x = self.pos_dropout(enc)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln(x)
        if not self.encoder:
            cls_token = x[:, 0]
            x = self.head(cls_token)
        else:
            x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        if self.output_classes > 1:
            loss = F.cross_entropy(preds, y)
            accuracy = self.accuracy(preds, y)
        else:
            loss = self.loss(preds, y)
            accuracy = self.accuracy(preds, y)

        self.log('train/loss', loss, on_step=True)
        self.log('train/accuracy', accuracy, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        if self.output_classes > 1:
            loss = F.cross_entropy(preds, y)
            accuracy = self.accuracy(preds, y)
        else:
            loss = self.loss(preds, y)
            accuracy = self.accuracy(preds, y)

        self.log('val/loss', loss, on_step=True)
        self.log('val/accuracy', accuracy, on_step=True)
        return preds

    def val_epoch_end(self, test_step_outputs):
        dummy_input = torch.zeros(1, 1, self.H, self.W)
        model_name = 'vit.onnx'
        torch.onnx.export(self, dummy_input, model_name)
        wandb.save(model_name)

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(), lr=self.lr)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, channels=1, n_embd=512):
        """
        img size: image shape
        """
        super().__init__()
        self.H, self.W = img_size
        assert (
            self.W % patch_size == 0 and self.H % patch_size == 0
        ), "image not divisable by patch size"
        self.patch_size = patch_size
        self.n_patches = (self.H // patch_size) * (self.W // patch_size)
        self.n_embd = n_embd

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
        # im = x[0, :]
        # print(im.shape)
        # im = im.reshape(self.n_patches, self.patch_size, self.patch_size).detach().numpy()
        # for i in im:
        # plt.imshow(i, cmap='gray')
        return x
