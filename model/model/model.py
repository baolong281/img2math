from model.vit import ViT
from model.decoder import Decoder
import torch
import numpy as np  
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam

#def __init__(self, img_shape, patch_size, n_embd, num_blocks=6, num_heads=8, dropout=.75, channels=1, output_classes=2, encoder=True, lr=1e-4):
#def __init__(self, n_embd, block_size, vocab_size, dropout, num_blocks=6, num_heads=8):
#def forward(self, input_seq, enc_output, trg_seq=None,  mask=None):

class Img2MathModel(L.LightningModule):
    def __init__(self, n_embd, block_size, vocab_size, dropout, img_shape, patch_size, encoder_blocks=4, decoder_blocks=6, num_heads=8, lr=1e-4, teacher_rate=.70):
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
        self.teacher_rate = teacher_rate
        self.save_hyperparameters()

    def forward(self, img, trg_seq=None, mask=None):
        B = img.shape[0]
        pred_sequence = torch.zeros((B, self.block_size), dtype=torch.int).detach().to(self.device)
        pred_sequence[: -1] = 1
        logit_sequence = torch.zeros(B, self.block_size, self.vocab_size).to(self.device)
        image_encodings = self.encoder(img)
        out_seq = torch.zeros((B, self.block_size), dtype=torch.long).to(self.device)

        for i in range(self.block_size):
            logits = self.decoder(pred_sequence, image_encodings) # B, self.vocab_size
            pred = torch.argmax(logits, dim=-1).detach()

            out_seq = torch.cat((out_seq, pred), dim=-1)

            if np.random.rand() < self.teacher_rate and trg_seq is not None:
                pred = torch.zeros((B, 1), dtype=torch.long).to(self.device)
                pred[torch.arange(B), 0] = trg_seq[torch.arange(B), i]

            pred_sequence = torch.cat((pred_sequence, pred), dim=-1).detach()
            pred_sequence = pred_sequence[:, 1:]

            if trg_seq is not None:
                logit_sequence = torch.cat((logit_sequence, logits), dim=1)
                logit_sequence = logit_sequence[:, 1:]


        loss = None
        if trg_seq is not None:
            loss = F.cross_entropy(logit_sequence.transpose(1, 2), trg_seq)

        return out_seq, loss 
        

    def training_step(self, batch, batch_idx):
        img, labels = batch
        trg_seq, input_mask = labels['input_ids'], labels['attention_mask']
        _, loss  = self.forward(img, trg_seq=trg_seq, mask=input_mask)
        self.log('train/loss', loss, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        img, labels = batch
        trg_seq, input_mask = labels['input_ids'], labels['attention_mask']
        _, loss  = self.forward(img, trg_seq=trg_seq, mask=input_mask)
        self.log('val/loss', loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
