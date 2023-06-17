import torch.nn as nn
import torch
from model.transformer import DecoderTransformerBlock

class Decoder(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, dropout, num_blocks=6, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.word_embeddings = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.block_size, n_embd))
        self.transformer_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(n_embd, num_heads, dropout)
            ]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        pass

    def forward(self, trg_seq, enc_output):
        embeddings = self.word_embeddings(trg_seq)
        return embeddings
