import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        This class is implementation of Multi-Head Attention,
        for Detailed Explanation on shapes and Mathematics, linear Algebra visit: notebooks/12_Implementing_transformer.ipynb
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
        attention_scores = (query @ key.transpose(-2, -1)) / (self.d_k ** 0.5)
        # query @ key --> (B, h, T, d_k) @ (B, h, d_k, T) --> (B, h, T, T)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim = -1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None):
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

