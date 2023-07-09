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
parser.add_argument('--accelerator', type=str)

args = parser.parse_args()

def main():
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    N_EMBD = 512
    BLOCK_SIZE = 256
    DROPOUT = .75
    PATCH_SIZE = 16
    IMG_DIMS = [256, 256]
    device = torch.device(args.accelerator)

    print('******* train parameters *******')
    print(f"epochs: {EPOCHS}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"device: {device}")
    print(f"n_embd: {N_EMBD}")
    print(f"block size: {BLOCK_SIZE}")
    print(f"dropout: {DROPOUT}")
    print(f"patch size: {PATCH_SIZE}")
    print(f"img dims: {IMG_DIMS}")
    print('********************************')

    data = Im2LatexDataset(path_to_data="../data/",
                           tokenizer="../data/tokenizer.json", img_dims=IMG_DIMS, batch_size=BATCH_SIZE, device=device)
    vocab_size = len(data.tokenizer.get_vocab())
    model = Img2MathModel(N_EMBD, BLOCK_SIZE, vocab_size, DROPOUT, IMG_DIMS, PATCH_SIZE)

    logger = WandbLogger(project='img2math')

    trainer = L.Trainer(max_epochs=EPOCHS, log_every_n_steps=20, deterministic=True,
                        logger=logger, accelerator=args.accelerator)

    trainer.fit(model, data.train, data.test)
    wandb.finish()

if __name__ == "__main__":
    main()
