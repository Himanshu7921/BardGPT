import torch
import torch.nn as nn
from embedding import EmbeddingBlock
from attention import MultiHeadAttention

class TransformerFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN) used in Transformer blocks.

    This module consists of two linear transformations with a non-linear activation 
    function (GELU/ReLU) and dropout in between. It projects the input to a higher-dimensional 
    space (d_ff) and then back to the model dimension (d_model).

    Attributes:
        LinearLayer_1 (nn.Linear): The first linear projection (d_model -> d_ff).
        LinearLayer_2 (nn.Linear): The second linear projection (d_ff -> d_model).
        dropout (nn.Dropout): Dropout layer applied after activations and projections.
        activation_fn (nn.Module): Non-linear activation function (default: GELU).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initializes the TransformerFeedForward module.

        Args:
            d_model (int): The dimensionality of the input and output logical space.
            d_ff (int): The dimensionality of the inner/hidden layer (feed-forward size).
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.LinearLayer_1 = nn.Linear(d_model, d_ff, bias = True)
        self.LinearLayer_2 = nn.Linear(d_ff, d_model, bias = True)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = nn.GELU() # or we can use, nn.RELU()
    
    def forward(self, x):
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.LinearLayer_1(x) # Projection from d_model --> d_ff
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.LinearLayer_2(x) # Project back from d_ff --> d_model
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    """
    Implements Layer Normalization to stabilize the hidden state dynamics.

    Layer normalization normalizes the activations across the feature dimension (d_model) 
    for each training example independently. This is equivalent to PyTorch's 
    nn.LayerNorm implementation.

    Attributes:
        eps (float): A small value added to the variance for numerical stability.
        d_model (int): The dimensionality of the input features.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initializes the LayerNorm module.

        Args:
            d_model (int): The number of expected features in the input.
            eps (float, optional): Value added to denominator for numerical stability. 
                Defaults to 1e-6.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        """
        Applies Layer Normalization to the input tensor.

        The calculation follows: y = gamma * (x - mean) / sqrt(var + eps) + beta.

        Args:
            x (torch.Tensor): Input tensor where normalization is applied to the last dim.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
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
        """
        Initializes the ResidualConnections module with LayerNorm and Dropout layers.

        This setup prepares the components necessary for the Pre-LayerNorm residual 
        logic, ensuring the normalization happens on the expected feature dimension 
        and that regularization is applied to the sublayer output.

        Args:
            d_model (int): The dimensionality of the input features (embedding size). 
                Used to initialize the LayerNorm parameters.
            dropout (float): The probability of an element to be zeroed during the 
                dropout phase for regularization.
        """
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer):
        """
        Applies the residual connection logic.

        Args:
            x (torch.Tensor): The input tensor.
            sublayer (callable): A function or lambda representing the sub-layer 
                transformation (e.g., self-attention or feed-forward).

        Returns:
            torch.Tensor: The normalized and summed output.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class LinearProjection(nn.Module):
    """ 
    A linear transformation layer to project model features to vocabulary space.

    Commonly used at the end of the Transformer to convert the decoder output 
    into logit scores for each token in the vocabulary.

    Attributes:
        linear_layer (nn.Linear): The linear projection (d_model -> vocab_size).
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
         Initializes the LinearProjection layer.

        Args:
            d_model (int): The dimensionality of the input features.
            vocab_size (int): The total size of the vocabulary.
        """
        super().__init__()
        self.linear_layer = nn.Linear(in_features = d_model, out_features = vocab_size, bias = False)
    
    def forward(self, x):
        """
        Projects the hidden state to logit space.

        Args:
            x (torch.Tensor): Tensor from the last decoder block.

        Returns:
            torch.Tensor: Predicted logits for the next token.
        """
        return self.linear_layer(x)
    
class DecoderBlock(nn.Module):
    """
        A single block/layer of the Transformer Decoder.

    This block applies a multi-head self-attention mechanism followed by a 
    position-wise feed-forward network. Each sub-layer is wrapped in a 
    residual connection and layer normalization (Pre-LayerNorm).

    Attributes:
        self_attention_block (MultiHeadAttention): The self-attention mechanism.
        feed_forward (TransformerFeedForward): The position-wise feed-forward network.
        residual_connections_attention (ResidualConnections): Residual wrapper for attention.
        residual_connections_feed_forward (ResidualConnections): Residual wrapper for FFN.
    """
    def __init__(self, d_model: int, d_ff: int, h: int, dropout: float):
        """
        Initializes the DecoderBlock with self-attention and feed-forward sub-layers.

        This constructor instantiates the core components of a Transformer decoder 
        layer, including multi-head attention and a position-wise feed-forward network, 
        each wrapped in a Pre-LayerNorm residual connection.

        Args:
            d_model (int): The number of expected features in the input (embedding dimension).
            d_ff (int): The dimensionality of the inner layer in the feed-forward network.
            h (int): The number of attention heads.
            dropout (float): The dropout probability applied within sub-layers and 
                residual connections for regularization.
        """
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model = d_model, h = h, dropout = dropout)
        self.residual_connections_attention = ResidualConnections(d_model = d_model, dropout = dropout)

        self.feed_forward = TransformerFeedForward(d_ff = d_ff, d_model = d_model, dropout = dropout)
        self.residual_connections_feed_forward = ResidualConnections(d_model = d_model, dropout = dropout)

    def forward(self, x: torch.Tensor, mask):
        """
        Passes the input through the decoder block.

        Args:
            x (torch.Tensor): The input tensor from the previous layer or embedding.
            mask (torch.Tensor): Mask to prevent attention to future tokens or padding.

        Returns:
            torch.Tensor: The processed tensor of the same shape as input.
        """
        x =  self.residual_connections_attention(x,
            lambda x: self.self_attention_block(x, x, x, mask)
        )

        x = self.residual_connections_feed_forward(x,
            self.feed_forward
        )
        
        return x

class Decoder(nn.Module):
    """
    The full Transformer Decoder module.

    Consists of a stack of N identical DecoderBlocks. In the original 
    'Attention is All You Need' paper, N is typically set to 6.

    Attributes:
        decoder_blocks (nn.ModuleList): A list containing N instances of DecoderBlock.
    """
    def __init__(self, N: int, d_model: int, d_ff: int, h: int, dropout: float):
        """
        Initializes the Decoder.

        Args:
            N (int): The number of decoder blocks to stack.
            d_model (int): The dimensionality of the model's hidden states.
            d_ff (int): The dimensionality of the feed-forward hidden layer.
            h (int): The number of attention heads.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model = d_model, d_ff = d_ff, h = h, dropout = dropout)
            for _ in range(N)]
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Passes the input through the stack of decoder blocks.

        Args:
            x (torch.Tensor): Input embeddings of shape (B, T, d_model).
            mask (torch.Tensor): Look-ahead mask to prevent attending to future tokens.

        Returns:
            torch.Tensor: Final hidden states of shape (B, T, d_model).
        """
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, mask)
        return x # x.shape = (B, T, d_model) but for predictions we need (B, T, vocab_size) --> Another Linear Projections
