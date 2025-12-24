import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "transformer-feed-forward", text: "TransformerFeedForward", level: 2 },
  { id: "layer-norm", text: "LayerNorm", level: 2 },
  { id: "residual-connections", text: "ResidualConnections", level: 2 },
  { id: "linear-projection", text: "LinearProjection", level: 2 },
  { id: "decoder-block", text: "DecoderBlock", level: 2 },
  { id: "decoder", text: "Decoder", level: 2 },
];

export default function ApiBlock() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.block" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.block</h1>
        
        <p className="text-lg text-muted-foreground">
          Transformer building blocks: feedforward networks, normalization, residual 
          connections, and decoder blocks.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module contains the fundamental building blocks that compose a transformer 
          decoder layer. Each component is designed to be modular and reusable.
        </p>

        <h2 id="transformer-feed-forward">TransformerFeedForward</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> TransformerFeedForward(nn.Module)
        </div>

        <p>
          Position-wise feedforward network applied independently to each position. 
          Consists of two linear transformations with a GELU activation between them.
        </p>

        <ApiTable 
          parameters={[
            { name: "d_model", type: "int", description: "Input and output dimension." },
            { name: "d_ff", type: "int", description: "Hidden layer dimension (typically 4× d_model)." },
            { name: "dropout", type: "float", description: "Dropout probability." },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          filename="transformer/block.py"
          language="python"
          code={`class TransformerFeedForward(nn.Module):
    """
    Position-wise feedforward network.
    
    FFN(x) = Dropout(GELU(xW₁ + b₁))W₂ + b₂
    
    Input shape:  (B, S, d_model)
    Output shape: (B, S, d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)      # (B, S, d_ff)
        x = F.gelu(x)            # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)      # (B, S, d_model)
        return x`}
        />

        <Callout type="tip" title="GELU vs ReLU">
          GELU (Gaussian Error Linear Unit) provides smoother gradients than ReLU and 
          has become the standard activation in modern transformers. It approximates 
          <code>x × Φ(x)</code> where Φ is the standard Gaussian CDF.
        </Callout>

        <h2 id="layer-norm">LayerNorm</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> LayerNorm(nn.Module)
        </div>

        <p>
          Layer normalization applied across the last dimension (feature dimension).
        </p>

        <ApiTable 
          parameters={[
            { name: "d_model", type: "int", description: "Feature dimension to normalize over." },
            { name: "eps", type: "float", description: "Small constant for numerical stability.", default: "1e-6" },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          language="python"
          code={`class LayerNorm(nn.Module):
    """
    Layer normalization with learnable affine parameters.
    
    LN(x) = γ × (x - μ) / (σ + ε) + β
    
    Where μ and σ are computed across the last dimension.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize and apply affine transformation
        return self.gamma * (x - mean) / (std + self.eps) + self.beta`}
        />

        <p><strong>Attributes:</strong></p>
        <ul>
          <li><code>gamma</code> — Learnable scale parameter, initialized to ones</li>
          <li><code>beta</code> — Learnable shift parameter, initialized to zeros</li>
          <li><code>eps</code> — Small constant to prevent division by zero</li>
        </ul>

        <h2 id="residual-connections">ResidualConnections</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> ResidualConnections(nn.Module)
        </div>

        <p>
          Pre-LayerNorm residual connection wrapper. Applies normalization before the 
          sublayer, then adds the residual.
        </p>

        <CodeBlock
          language="python"
          code={`class ResidualConnections(nn.Module):
    """
    Pre-LN residual connection.
    
    Output = x + Dropout(Sublayer(LayerNorm(x)))
    
    Pre-LN (normalize first) is preferred over Post-LN for training stability,
    as it keeps the residual pathway clean and gradients flowing smoothly.
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        # Pre-LN: normalize, apply sublayer, add residual
        return x + self.dropout(sublayer(self.norm(x)))`}
        />

        <Callout type="info" title="Pre-LN vs Post-LN">
          <strong>Pre-LN</strong> (used here): <code>x + Sublayer(Norm(x))</code><br/>
          <strong>Post-LN</strong> (original paper): <code>Norm(x + Sublayer(x))</code><br/>
          Pre-LN provides better gradient flow and training stability, especially for 
          deeper models.
        </Callout>

        <h2 id="linear-projection">LinearProjection</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> LinearProjection(nn.Module)
        </div>

        <p>
          Linear projection layer used as the output head to project hidden states 
          to vocabulary size.
        </p>

        <CodeBlock
          language="python"
          code={`class LinearProjection(nn.Module):
    """
    Output projection layer.
    
    Projects from d_model to vocab_size.
    Bias is disabled to support weight tying with embeddings.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, d_model) -> (B, S, vocab_size)
        return self.linear(x)`}
        />

        <h2 id="decoder-block">DecoderBlock</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> DecoderBlock(nn.Module)
        </div>

        <p>
          A single decoder block containing self-attention and feedforward sublayers, 
          each wrapped with residual connections.
        </p>

        <ApiTable 
          parameters={[
            { name: "config", type: "TransformerConfig", description: "Configuration containing d_model, d_ff, n_heads, dropout." },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          language="python"
          code={`class DecoderBlock(nn.Module):
    """
    Single decoder block.
    
    Structure:
    1. Self-attention with residual connection
    2. Feedforward network with residual connection
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        # Self-attention sublayer
        self.self_attention = MultiHeadAttention(
            config.d_model, 
            config.n_heads, 
            config.dropout
        )
        self.residual1 = ResidualConnections(config.d_model, config.dropout)
        
        # Feedforward sublayer
        self.feed_forward = TransformerFeedForward(
            config.d_model, 
            config.d_ff, 
            config.dropout
        )
        self.residual2 = ResidualConnections(config.d_model, config.dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask))
        
        # Feedforward with residual
        x = self.residual2(x, self.feed_forward)
        
        return x`}
        />

        <h2 id="decoder">Decoder</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> Decoder(nn.Module)
        </div>

        <p>
          Stack of N decoder blocks that form the main body of the transformer.
        </p>

        <CodeBlock
          language="python"
          code={`class Decoder(nn.Module):
    """
    Stack of N decoder blocks.
    
    Each block applies self-attention and feedforward transformations
    with residual connections.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all decoder blocks.
        
        Args:
            x: Input tensor (B, S, d_model)
            mask: Attention mask (S, S)
        
        Returns:
            Output tensor (B, S, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x`}
        />

        <Callout type="info" title="Layer Count">
          The number of decoder blocks is controlled by <code>config.n_layers</code>. 
          More layers increase model capacity but also training time and memory usage.
        </Callout>
      </div>
    </DocLayout>
  );
}
