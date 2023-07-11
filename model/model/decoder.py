import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import MLP

class Decoder(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, dropout, num_blocks=6, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.word_embeddings = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.block_size, n_embd))
        self.transformer_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(n_embd, num_heads, dropout) for _ in range(num_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        pass

    def forward(self, input_seq, enc_output, trg_seq=None,  mask=None):
        input_embeddings = self.word_embeddings(input_seq)
        input_embeddings = input_embeddings + self.pos_embed
        x = self.ln(input_embeddings)

        for layer in self.transformer_blocks:
            x = layer(x, enc_output, mask)

        out = self.ln(x)
        # print(out.shape, out)

        if trg_seq is not None :
            # if we are given some desired targets also calculate the loss
            logits = self.head(out)
            print(logits.shape, trg_seq.shape)
            loss = F.cross_entropy(logits.transpose(1, 2), trg_seq)
            print("functional")
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(out[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
         
        return logits, loss

class DecoderTransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        self.embed_dim = n_embd
        self.num_heads = num_heads
        self.dropout = dropout

        self.ln = torch.nn.LayerNorm(n_embd, elementwise_affine=False)
        self.attention = DecoderAttention(n_embd, num_heads, dropout)
        self.head = MLP(n_embd, dropout, bias=False)

    def forward(self, embeddings, enc_output,  mask=None):
        embeddings = self.ln(embeddings)
        embeddings = embeddings + self.attention(embeddings, embeddings, embeddings, mask=mask) #calculate masked multihead attention
        q, k, v = self.ln(embeddings), self.ln(enc_output), self.ln(enc_output) # feed output of masked attention into cross attention
        cross_attention = q + self.attention(q, k, v)
        cross_attention = self.ln(cross_attention)
        out = cross_attention + self.head(cross_attention)
        out = self.ln(out)
        return out


class DecoderAttention(nn.Module):
    ''' 
    Multi-Head Attention module 
    from: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/SubLayers.py#L9
    '''


    def __init__(self, n_embd, n_head, dropout=0.75):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd not divisable by head size"

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.w_qs = nn.Linear(n_embd, n_embd, bias=False)
        self.w_ks = nn.Linear(n_embd, n_embd, bias=False)
        self.w_vs = nn.Linear(n_embd, n_embd, bias=False)
        self.fc = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_embd, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.head_dim)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.head_dim)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.head_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        #     mask = mask.unsqueeze(1)
        #     if mask.dtype != torch.float32:
        #         mask = mask.float()

        q = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q)

        return q
