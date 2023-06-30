import torch.nn as nn
import torch
from model.vit import ViT
from model.decoder import Decoder

#def __init__(self, img_shape, patch_size, n_embd, num_blocks=6, num_heads=8, dropout=.75, channels=1, output_classes=2, encoder=True, lr=1e-4):
#def __init__(self, n_embd, block_size, vocab_size, dropout, num_blocks=6, num_heads=8):
#def forward(self, input_seq, enc_output, trg_seq=None,  mask=None):

class Img2MathModel(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, dropout, img_shape, patch_size, encoder_blocks=4, decoder_blocks=6, num_heads=8):
        super().__init__()
        self.encoder = ViT(img_shape, patch_size, n_embd, num_blocks=encoder_blocks, num_heads=num_heads, dropout=dropout)
        self.decoder = Decoder(n_embd, block_size, vocab_size, dropout, num_blocks=decoder_blocks, num_heads=num_heads)

    def forward(self, img, input_seq=None, trg_seq=None,  mask=None):
        B = img.shape[0]
        #make dummy input seqs for each bach with last element being BOS token
        if input_seq is None:
            input_seq = torch.zeros((B, 256), dtype=torch.int)
            input_seq[: -1] = 1
        encodings = self.encoder(img)
        out = self.decoder(input_seq, encodings, trg_seq=trg_seq, mask=mask)
        return out
