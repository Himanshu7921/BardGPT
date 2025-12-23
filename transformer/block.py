import torch
import torch.nn as nn
from embedding import EmbeddingBlock
from attention import MultiHeadAttention

class TransformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        This Implementation of TransformerFeedForward consists of
        - 2 Linear Projections
             > Layer-1: Projects from d_model --> d_ff
             > Layer-2: Projects from d_ff --> d_model
        - Dropout for each Linear Projections
        - Non-Linear Activation Function
        """
        super().__init__()
        self.LinearLayer_1 = nn.Linear(d_model, d_ff, bias = True)
        self.LinearLayer_2 = nn.Linear(d_ff, d_model, bias = True)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.GELU() # or we can use, nn.RELU()
    
    def forward(self, x):
        x = self.LinearLayer_1(x) # Projection from d_model --> d_ff
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.LinearLayer_2(x) # Project back from d_ff --> d_model
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        This class is responsible for Normalizing the input before feeding into next Layer,
        to stabilize activations by normalizing across the feature dimension
        This Implementation is Equivalent to PyTorch's nn.LayerNorm()
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = torch.mean(x, dim = -1, keepdim = True)
        var = torch.var(x, dim = -1, keepdim = True, unbiased = False)
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        return self.gamma * x_norm + self.beta

class ResidualConnections(nn.Module):
    """
    This class implements residual connections.

    Instead of completely replacing the input with a non-linear transformation F(x),
    the residual connection preserves the original input via an identity shortcut and
    adds the transformed output to it:

        y = x + F(x)

    This allows each sublayer to learn a residual correction rather than a full
    representation, improving gradient flow, preserving information across layers,
    and enabling stable training of deep Transformer architectures.

    NOTE: 
        In the Official "Attention is All you need" Paper they implemented Post-LayerNorm Residual Connection
    > Official Implementation says: 
        - That is, the output of each sub-layer is
           LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
           itself
    > What I've implemented is called as, Pre-LayerNorm Residual Connections
        - x + Dropout(SubLayer(LayerNorm(x))
        - Also Added Dropout to the sublayer's output for regularization
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LinearProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        In the Original Implementation they called it as Linear, and is been used after Decoder Blocks
        """
        super().__init__()
        self.linear_layer = nn.Linear(in_features = d_model, out_features = vocab_size, bias = False)
    
    def forward(self, x):
        return self.linear_layer(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, h: int, dropout: float):
        """
        Note: In the Paper They have showed ResidalConnections as skip connections acting as wires
        Add&Norm are Applied in Residual Connections
        """
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model = d_model, h = h, dropout = dropout)
        self.residual_connections_attention = ResidualConnections(d_model = d_model, dropout = dropout)

        self.feed_forward = TransformerFeedForward(d_ff = d_ff, d_model = d_model, dropout = dropout)
        self.residual_connections_feed_forward = ResidualConnections(d_model = d_model, dropout = dropout)

    def forward(self, x: torch.Tensor, mask):
        x =  self.residual_connections_attention(x,
            lambda x: self.self_attention_block(x, x, x, mask)
        )

        x = self.residual_connections_feed_forward(x,
            self.feed_forward
        )
        
        return x

class Decoder(nn.Module):
    """
    In Original Paper they used stack of N = 6 identical layers
    """
    def __init__(self, N: int, d_model: int, d_ff: int, h: int, dropout: float):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model = d_model, d_ff = d_ff, h = h, dropout = dropout)
            for _ in range(N)]
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, mask)
        return x # x.shape = (B, T, d_model) but for predictions we need (B, T, vocab_size) --> Another Linear Projections
