import sys
sys.path.append("../")
import torch
from data.dataset import Im2LatexDataset
from model.model import Img2MathModel
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
import wandb
import argparse
from transformers import PreTrainedTokenizerFast


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
    TOKENIZER_FILE = '../data/tokenizer.json'
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

    def decode(tokenizer, sequence):
        dec = [tokenizer.decode(tok) for tok in sequence]
        return ''.join([detok.replace('Ġ', ' ') for detok in dec])

    class ImagePredictionLogger(L.Callback):
        
        def __init__(self, val_samples, num_samples=10):
            super().__init__()
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_FILE, padding_side='left')
            self.val_imgs, self.val_labels = val_samples
            self.val_imgs = self.val_imgs[:num_samples]
            self.val_labels = [decode(self.tokenizer, label) for label in self.val_labels['input_ids'][:num_samples]]


        def on_validation_epoch_end(self, trainer, pl_module):
            val_imgs = self.val_imgs.to(device=pl_module.device)

            pred_tokens = pl_module.generate(val_imgs)
            preds = [decode(self.tokenizer, pred) for pred in pred_tokens]

            trainer.logger.experiment.log({
                "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                             for x, pred, y in zip(val_imgs, preds, self.val_labels)],
                "global_step": trainer.global_step
            })
            
    data = Im2LatexDataset(path_to_data="../data/",
                           tokenizer=TOKENIZER_FILE, img_dims=IMG_DIMS, batch_size=BATCH_SIZE, device=device)
    vocab_size = len(data.tokenizer.get_vocab())

    batch = next(iter(data.val))

    model = Img2MathModel(N_EMBD, BLOCK_SIZE, vocab_size, DROPOUT, IMG_DIMS, PATCH_SIZE)

    logger = WandbLogger(project='img2math')

    trainer = L.Trainer(max_epochs=EPOCHS, log_every_n_steps=5, deterministic=False,
                        logger=logger, limit_val_batches=.10, accelerator=args.accelerator, callbacks=[ImagePredictionLogger(batch)])

    trainer.fit(model, data.train, data.test)
    wandb.finish()

if __name__ == "__main__":
    main()
