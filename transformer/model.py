from embedding import EmbeddingBlock
from block import Decoder, LinearProjection
import torch
import torch.nn as nn

class DecoderOnlyTransformerModel(nn.Module):
    """
    A Decoder-Only Transformer (GPT-style) for generative language modeling.

    This model consists of an initial embedding layer (token + positional), followed by a series of Transformer decoder blocks,
    and a final linear projection layer for token prediction.

    Attributes:
        self.embeddings (EmbeddingBlock): Layer that handles token and positional encoding.
        self.decoder (Decoder): A stack of N decoder blocks for feature extraction.
        self.linear_projection (LinearProjection): Final layer projecting features back to vocab space.
    """
    def __init__(self, max_seq_length: int, vocab_size: int, d_model: int, d_ff: int, h: int, use_fixed_positional_embeddings: bool, dropout: float, N: int):
        """
        Initializes the model architecture with specified hyperparameters.

        Args:
            max_seq_length (int): The absolute maximum context length allowed.
            vocab_size (int): Total number of unique tokens in the character set.
            d_model (int): The embedding dimension for each token.
            d_ff (int): The internal dimension of the feed-forward network.
            N (int): The total number of decoder blocks to stack.
            h (int): Number of attention heads in each block.
            dropout (float): Regularization probability used throughout the model.
            use_fixed_positional_embeddings (bool): Whether to use sinusoidal or learned positions.
        """
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
        """
        Processes a batch of input sequences through the model.

        Args:
            x (torch.Tensor): Input token indices of shape (Batch, Sequence_Length).
            mask (torch.Tensor): Casual mask to prevent attending to future tokens.

        Returns:
            torch.Tensor: Predicted logits for the next token in the sequence.
        """
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
