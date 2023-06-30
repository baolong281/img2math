from model.vit import ViT
from model.decoder import Decoder
import torch
import lightning as L
from torch.optim import Adam

#def __init__(self, img_shape, patch_size, n_embd, num_blocks=6, num_heads=8, dropout=.75, channels=1, output_classes=2, encoder=True, lr=1e-4):
#def __init__(self, n_embd, block_size, vocab_size, dropout, num_blocks=6, num_heads=8):
#def forward(self, input_seq, enc_output, trg_seq=None,  mask=None):

class Img2MathModel(L.LightningModule):
    def __init__(self, n_embd, block_size, vocab_size, dropout, img_shape, patch_size, encoder_blocks=4, decoder_blocks=6, num_heads=8, lr=1e-4):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.patch_size = patch_size
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.num_heads = num_heads
        self.encoder = ViT(img_shape, patch_size, n_embd, num_blocks=encoder_blocks, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(n_embd, block_size, vocab_size, dropout, num_blocks=decoder_blocks, num_heads=num_heads)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, img, input_seq=None, trg_seq=None,  mask=None):
        #make dummy input seqs for each bach with last element being BOS token
        if input_seq is None:
            B = img.shape[0]
            input_seq = torch.zeros((B, 256), dtype=torch.int)
            input_seq[: -1] = 1

        encodings = self.encoder(img)
        logits, loss = self.decoder(input_seq, encodings, trg_seq=trg_seq, mask=mask)
        return logits, loss 

    def training_step(self, batch, batch_idx):
        img, labels = batch
        trg_seq, input_mask = labels['input_ids'], labels['attention_mask']
        zero = torch.zeros(1, 1).to(self.device)
        input_seq = torch.cat((zero, trg_seq), dim=-1)
        input_seq = input_seq[:, :-1].int()
        input_mask = torch.cat((zero, input_mask), dim=-1)
        input_mask= input_mask[:, :-1].int()

        _, loss  = self.forward(img, input_seq=input_seq, trg_seq=trg_seq, mask=input_mask)
        self.log('train/loss', loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        img, labels = batch
        trg_seq, input_mask = labels['input_ids'], labels['attention_mask']
        zero = torch.zeros(1, 1).to(self.device)
        input_seq = torch.cat((zero, trg_seq), dim=-1)
        input_seq = input_seq[:, :-1].int()
        input_mask = torch.cat((zero, input_mask), dim=-1)
        input_mask= input_mask[:, :-1].int()

        _, loss  = self.forward(img, input_seq=input_seq, trg_seq=trg_seq, mask=input_mask)
        self.log('val/loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
