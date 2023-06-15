import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers.wandb import WandbLogger
from model.vit import ViT
from data.dataset import Im2LatexDataset
import lightning as L
import wandb

BATCH_SIZE = 10

img_dims = [256, 256]
model = ViT(img_dims, 16, n_embd=256, encoder=False,
            output_classes=1, num_blocks=3, dropout=.75)
# model = ViT.load_from_checkpoint('./mnisttest/zc5enho0/checkpoints/epoch=4-step=2420.ckpt')

img_dims = [224, 640]
model = ViT(img_dims, 16, n_embd=512, encoder=False, output_classes=1)
data = Im2LatexDataset(path_to_data="../data/",
                       tokenizer="../data/tokenizer.json", img_dims=img_dims, classification=True, batch_size=BATCH_SIZE, device=torch.device('mps'))
batch = next(iter(data.train))


class ImagePredictionLogger(L.Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = F.sigmoid(logits)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })


# checkpoint_callback = ModelCheckpoint(monitor='train accuracy', mode='max')
logger = WandbLogger(project='mnisttest')

trainer = L.Trainer(limit_train_batches=484, max_epochs=20, log_every_n_steps=15, deterministic=True,
                    logger=logger, callbacks=[ImagePredictionLogger(batch)], accelerator='mps')

trainer.fit(model, data.train, data.val)
wandb.finish()
