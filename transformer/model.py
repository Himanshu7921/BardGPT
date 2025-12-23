from embedding import EmbeddingBlock
from block import Decoder, LinearProjection
import torch
import torch.nn as nn

class DecoderOnlyTransformerModel(nn.Module):
    """
    This is Decoder Only Transformer
    """
    def __init__(self, max_seq_length: int, vocab_size: int, d_model: int, d_ff: int, h: int, use_fixed_positional_embeddings: bool, dropout: float, N: int):
        super().__init__()
        self.embeddings = EmbeddingBlock(max_seq_length = max_seq_length,
                    vocab_size = vocab_size,
                    d_model = d_model,
                    use_fixed_positional_embeddings = use_fixed_positional_embeddings,
                    dropout = dropout
        )
        self.decoder = Decoder(N = N,
                    d_model= d_model,
                    d_ff = d_ff,
                    h = h,
                    dropout = dropout)
        
        self.linear_projection = LinearProjection(d_model = d_model, vocab_size = vocab_size)

        # Weight Tying, weight of Projected layer = Weight of Token embedding Layer
        self.linear_projection.linear_layer.weight = self.embeddings.token_embedding.embeddings.weight
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x.shape = (B, T)
        x_enc = self.embeddings(x)                      # (B, T, d_model)
        decoder_output = self.decoder(x_enc, mask)      # (B, T, d_model)
        logits = self.linear_projection(decoder_output) # (B, T, vocab_size)
        return logits

# Test cases
if __name__ == "__main__":
    B, T, d_model, d_ff = 2, 8, 20, 30
    vocab_size = 65
    x = torch.randint(0, vocab_size, (B, T))
    mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
    model = DecoderOnlyTransformerModel(max_seq_length = 10,
                vocab_size = vocab_size,
                d_model = d_model,
                d_ff = d_ff,
                h = 2,
                use_fixed_positional_embeddings = True,
                dropout = 0.1,
                N = 2)

    logits = model(x, mask)
    print(logits.shape)  # (B, T, vocab_size) --> (2, 8, 65)
