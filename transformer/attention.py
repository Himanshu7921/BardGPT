import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Implements a Multi-Head Attention mechanism for Transformer architectures.

    This class performs parallel attention operations by splitting the model dimension (d_model) into 
    multiple heads (h), allowing the model to jointly attend to information from different representation subspaces.

    Attributes:
        self.d_model (int): The toatl dimension of the model.
        self.h (int): The number of attention heads.
        self.d_k (int): Dimension of each key/query head (d_model // h).
        self.d_v (int): Dimension of each value head (d_model // h).
        self.W_Q (nn.Linear): Linear projections for Queries.
        self.W_K (nn.Linear): Linear projections for Keys.
        self.W_V (nn.Linear): Linear projections for Values.
        self.W_O (nn.Linear): Final Linear projections layer.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        Initializes the Multi-Head Attention layer.

        Args:
            d_model (int): The embedding dimension of the input.
            h (int): The number of attention heads (must be a divisor of d_model).
            dropout (float): Dropout probability for attention scores.
            
        Raises:
            AssertionError: If d_model is not divisible by h.
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_model // h
        self.d_v = self.d_model // h
        self.dropout = nn.Dropout(dropout)
        self.W_Q = nn.Linear(self.d_model, self.h * self.d_k, bias = False)
        self.W_K = nn.Linear(self.d_model, self.h * self.d_k, bias = False)
        self.W_V= nn.Linear(self.d_model, self.h * self.d_v, bias = False)
        self.W_O = nn.Linear(self.d_model, self.d_model)
    
    def get_attention_scores(self, query: torch.Tensor, key: torch.Tensor, mask = None):
        """
        Calculates scaled dot-product attention scores.

        Args:
            query (torch.Tensor): Projected query tensor of shape (B, h, T, d_k).
            key (torch.Tensor): Projected key tensor of shape (B, h, T, d_k).
            mask (torch.Tensor, optional): Casual or padding mask to hide specific tokens.

        Returns:
            torch.Tensor: Normalized attention probabilities of shape (B, h, T, T)
        """
        attention_scores = (query @ key.transpose(-2, -1)) / (self.d_k ** 0.5)
        # query @ key --> (B, h, T, d_k) @ (B, h, d_k, T) --> (B, h, T, T)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):
        """
        Executes the multi-head attention forward pass.

        Args: 
            q (torch.Tensor): Input query tensor of shape (B, T, d_model).
            k (torch.Tensor): Input key tensor of shape (B, T, d_model).
            v (torch.Tensor): Input value tensor of shape (B, T, d_model)
            mask (torch.Tensor, optional): Casual or padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output of the multi-head attention after concatenation and final linear
                        projection, of shape (B, T, d_model). 
        """
        B, T, _ = q.shape
        query = self.W_Q(q).view(B, T, self.h, self.d_k).transpose(1, 2)
        key = self.W_K(k).view(B, T, self.h, self.d_k).transpose(1, 2)
        value = self.W_V(v).view(B, T, self.h, self.d_k).transpose(1, 2)
        attention_scores = self.get_attention_scores(query, key, mask)

        Z = attention_scores @ value
        # attention_scores @ value --> (B, h, T, T) @ (B, h, T, d_k) --> (B, h, T, d_k) --> --> (B, T, h, d_k)
        Z = Z.transpose(1, 2).contiguous().view(B, T, self.d_model)
        Z = self.W_O(Z) # Output Projection
        return self.dropout(Z)

