import torch
import torch.nn as nn

class LearnedPositionalEmbeddings(nn.Module):
    """
    Implements a learned positional embedding layer.

    This layer allows the model to learn a unique vector representation for 
    each absolute position in a sequence, up to the maximum sequence length.

    Attributes:
        self.max_seq_length (int): The maximum sequence length supported.
        self.embeddings (nn.Embedding): A lookup table of shape (max_seq_length, d_model).
    """
    def __init__(self, max_seq_length: int, d_model: int):
        """
        Initializes the learned positional embedding layer.

        Args:
            max_seq_length (int): The absolute maximum context length.
            d_model (int): The embedding dimension for each position.
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embeddings = nn.Embedding(max_seq_length, d_model)

    def forward(self, x: torch.Tensor):
        """
        Retrieves positional embeddings for the input sequence indices.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T).

        Returns:
            torch.Tensor: Positional embeddings of shape (1, T, d_model).
        """
        _, T, _ = x.shape
        assert T <= self.max_seq_length, "Sequence length exceeds max_seq_length"
        idx = torch.arange(T, device = x.device)
        pos_embeddings = self.embeddings(idx) # (T, d_model)
        return x + pos_embeddings.unsqueeze(0) # (1, T, d_model)
    
class FixedPositionalEmbeddings(nn.Module):
    """
    Implements sinusoidal fixed positional embeddings.

    Uses sine and cosine functions of different frequencies to encode 
    positional information without requiring trainable parameters.

    Attributes:
        self.pe (torch.Tensor): Precomputed sinusoidal buffer of shape (max_seq_length, d_model).
    """
    def __init__(self, max_seq_length: int, d_model: int):
        """
        Computes the sinusoidal position encoding matrix.

        Args:
            max_seq_length (int): The absolute maximum context length.
            d_model (int): The embedding dimension.
        """
        super().__init__()
        self.max_seq_length = max_seq_length

        pos = torch.arange(max_seq_length).unsqueeze(1)      # (T, 1)
        i = torch.arange(d_model).unsqueeze(0)               # (1, d_model)

        angles = pos / (10000 ** (2 * (i // 2) / d_model))

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds positional encodings to the input tensor.

        The encoding is added element-wise to the input tensor 'x'. The 
        positional tensor is sliced to match the sequence length 'T' of 
        the input.
        """
        B, T, _ = x.shape
        assert T <= self.max_seq_length, "T(x.shape[1]) exceeds max_seq_length"
        return x + self.pe[:T].unsqueeze(0)
    
class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        """
        A trainable lookup table for token embeddings.

        Converts integer token indices into dense vectors of a fixed size (d_model).

        Args:
             vocab_size (int): Size of the dictionary of embeddings.
             d_model (int): The size of each embedding vector.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        """
        Maps input indices to their corresponding embedding vectors.

        Args:
            x (torch.LongTensor): Integer tensor of shape (Batch, Time) 
                containing token indices.

        Returns:
            torch.Tensor: Embedded output of shape (Batch, Time, d_model).

        """
        # x.shape = (B, T) --> (B, T, d_model)
        return self.embeddings(x)
    
# -------------------------------------------------------------------------------------------------------------------------

class EmbeddingBlock(nn.Module):
    """
    A comprehensive embedding block that combines tokens and positions.

    This class is responsible for encoding raw token indices into a 
    vectorized space by summing TokenEmbeddings and PositionalEmbeddings, 
    followed by a dropout layer for regularization.

    Attributes:
        self.token_embedding (TokenEmbeddings): Standard token-to-vector mapping.
        self.pos_embedding (Union[LearnedPositionalEmbeddings, FixedPositionalEmbeddings]): The chosen positional encoding mechanism.
        self.dropout (nn.Dropout): Regularization layer.
    """
    def __init__(self, max_seq_length: int, vocab_size:int, d_model: int, use_fixed_positional_embeddings: bool, dropout: float):
        """
         Initializes the embedding block.

        Args:
            max_seq_length (int): Maximum supported context length.
            vocab_size (int): Size of the character/token vocabulary.
            d_model (int): Hidden dimension size.
            use_fixed_pos (bool): If True, use sinusoidal embeddings; else learned.
            dropout (float): Probability of dropping units during training.
        """
        super().__init__()
        if use_fixed_positional_embeddings:
            self.pos_embedding = FixedPositionalEmbeddings(max_seq_length, d_model)
        else:
            self.pos_embedding = LearnedPositionalEmbeddings(max_seq_length, d_model)
        self.token_embedding = TokenEmbeddings(vocab_size, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Processes token indices into regularized, position-aware vectors.

        Args:
            x (torch.Tensor): LongTensor of token indices with shape (Batch, Time).

        Returns:
            torch.Tensor: Combined embeddings of shape (Batch, Time, d_model).
        """
        x_token_embd = self.token_embedding(x) * (self.d_model ** 0.5)
        x_pos_embd = self.pos_embedding(x_token_embd)
        return self.dropout(x_pos_embd)