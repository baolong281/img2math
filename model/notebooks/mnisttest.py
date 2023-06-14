import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torchvision
from model.vit import ViT
import lightning as L
import wandb

BATCH_SIZE = 64

img_dims = [28, 28]
model = ViT(img_dims, 4, n_embd=256, encoder=False, output_classes=10, num_blocks=6, dropout=.60)
#model = ViT.load_from_checkpoint('./mnisttest/zc5enho0/checkpoints/epoch=4-step=2420.ckpt')

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=BATCH_SIZE, shuffle=True)

batch = next(iter(test_loader))


class ImagePredictionLogger(L.Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })


# checkpoint_callback = ModelCheckpoint(monitor='train accuracy', mode='max')
logger = WandbLogger(project='mnisttest')

trainer = L.Trainer(limit_train_batches=484, max_epochs=20, log_every_n_steps=20, deterministic=True,
                    logger=logger, callbacks=[ImagePredictionLogger(batch)], accelerator='mps')

trainer.fit(model, train_loader, test_loader)
wandb.finish()
