import sys
sys.path.append("../")
import torch
from data.dataset import Im2LatexDataset
from model.model import Img2MathModel
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
import wandb
import argparse


parser = argparse.ArgumentParser(description='Get train args')
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)

args = parser.parse_args()

def main():
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print('******* train parameters *******')
    print(f"epochs: {EPOCHS}")
    print(f"batch size: {BATCH_SIZE}")

    img_dims = [256, 256]
    data = Im2LatexDataset(path_to_data="../data/",
                           tokenizer="../data/tokenizer.json", img_dims=img_dims, batch_size=BATCH_SIZE, device=torch.device('cuda'))
    vocab_size = len(data.tokenizer.get_vocab())
    model = Img2MathModel(512, 256, vocab_size, .75, img_dims, 16)

    logger = WandbLogger(project='img2math')

    trainer = L.Trainer(max_epochs=EPOCHS, log_every_n_steps=200, deterministic=True,
                        logger=logger, accelerator='cuda')

    trainer.fit(model, data.train, data.test)
    wandb.finish()

if __name__ == "__main__":
    main()
