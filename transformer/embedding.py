import torch
import torch.nn as nn

class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_length: int, d_model: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embeddings = nn.Embedding(max_seq_length, d_model)

    def forward(self, x: torch.Tensor):
        _, T, _ = x.shape
        assert T <= self.max_seq_length, "Sequence length exceeds max_seq_length"
        idx = torch.arange(T, device = x.device)
        pos_embeddings = self.embeddings(idx) # (T, d_model)
        return x + pos_embeddings.unsqueeze(0) # (1, T, d_model)
    
class FixedPositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_length: int, d_model: int):
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
        B, T, _ = x.shape
        assert T <= self.max_seq_length, "T(x.shape[1]) exceeds max_seq_length"
        return x + self.pe[:T].unsqueeze(0)
    
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor):
        # x.shape = (B, T) --> (B, T, d_model)
        return self.embeddings(x)
    
# -------------------------------------------------------------------------------------------------------------------------

class EmbeddingBlock(nn.Module):
    """
    This class is responsible for Encoding any given 'x' into TokenEmbedings + PositionalEmbeddings (fixed or Learned),
    followed by a Dropout Layer

    Because in the Original Paper they used Dropout in the Embeddings also

    Input: x: torch.Tensor, x.shape = (B, T)
    Output: x_pos_embd: torch.Tensor, x_pos_embd = (B, T, d_model)
    """
    def __init__(self, max_seq_length: int, vocab_size:int, d_model: int, use_fixed_positional_embeddings: bool, dropout: float):
        super().__init__()
        if use_fixed_positional_embeddings:
            self.pos_embedding = FixedPositionalEmbeddings(max_seq_length, d_model)
        else:
            self.pos_embedding = LearnedPositionalEmbeddings(max_seq_length, d_model)
        self.token_embedding = TokenEmbeddings(vocab_size, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x_token_embd = self.token_embedding(x) * (self.d_model ** 0.5)
        x_pos_embd = self.pos_embedding(x_token_embd)
        return self.dropout(x_pos_embd)