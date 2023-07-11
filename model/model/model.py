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

    def forward(self, img, input_seq, trg_seq=None,  mask=None):
        #make dummy input seqs for each bach with last element being BOS token
        encodings = self.encoder(img)

        logits, loss = self.decoder(input_seq, encodings, trg_seq=trg_seq, mask=mask)
        return logits, loss 

    def generate(self, img):
        B = img.shape[0]
        sequence = torch.zeros((B, self.block_size), dtype=torch.int).to(self.device)
        sequence[:, -1] = 1

        for i in range(self.block_size):
            logits, _ = self.forward(img, input_seq=sequence)
            pred = torch.argmax(logits, dim=-1)
            sequence = torch.cat((sequence, pred), dim=1).int()
            sequence = sequence[:, 1:]

        return sequence

    def training_step(self, batch, batch_idx):
        img, labels = batch
        B = img.shape[0]
        trg_seq, mask = labels['input_ids'], labels['attention_mask']

        #generating triangle mask
        # tokens_start = pre_mask.argmax() + 1
        # input_mask = torch.ones(B, self.block_size, tokens_start, dtype=torch.bool).triu(diagonal=tokens_start).to(self.device)
        # print(input_mask, trg_seq)

        zero = torch.zeros(B, 1).to(self.device)
        input_seq = torch.cat((zero, trg_seq), dim=-1)
        input_seq = input_seq[:, :-1].int().to(self.device)

        _, loss  = self.forward(img, input_seq=input_seq, trg_seq=trg_seq, mask=mask)
        print(loss)
        self.log('train/loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels = batch
        B = img.shape[0]
        trg_seq, pre_mask = labels['input_ids'], labels['attention_mask']

        #generating triangle mask
        # tokens_start = pre_mask.argmax() + 1
        # input_mask = torch.ones(B, self.block_size, tokens_start, dtype=torch.bool).triu(diagonal=tokens_start).to(self.device)
        # print(input_mask, trg_seq)

        zero = torch.zeros(B, 1).to(self.device)
        input_seq = torch.cat((zero, trg_seq), dim=-1)
        input_seq = input_seq[:, :-1].int().to(self.device)

        _, loss  = self.forward(img, input_seq=input_seq, trg_seq=trg_seq)
        self.log('val/loss', loss, on_step=True)
        return loss


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
