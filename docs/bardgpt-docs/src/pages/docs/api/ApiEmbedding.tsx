import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "token-embeddings", text: "TokenEmbeddings", level: 2 },
  { id: "fixed-positional-embeddings", text: "FixedPositionalEmbeddings", level: 2 },
  { id: "learned-positional-embeddings", text: "LearnedPositionalEmbeddings", level: 2 },
  { id: "embedding-block", text: "EmbeddingBlock", level: 2 },
  { id: "constraints", text: "Sequence Length Constraints", level: 2 },
];

export default function ApiEmbedding() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.embedding" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.embedding</h1>
        
        <p className="text-lg text-muted-foreground">
          Token and positional embedding layers that convert discrete tokens into 
          continuous representations.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          Embeddings transform discrete token indices into dense vectors that the 
          transformer can process. BardGPT supports both learned and fixed (sinusoidal) 
          positional embeddings.
        </p>

        <h2 id="token-embeddings">TokenEmbeddings</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> TokenEmbeddings(nn.Module)
        </div>

        <p>
          Lookup table that converts token indices to dense vectors.
        </p>

        <ApiTable 
          parameters={[
            { name: "vocab_size", type: "int", description: "Size of the vocabulary." },
            { name: "d_model", type: "int", description: "Dimension of the embedding vectors." },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          filename="transformer/embedding.py"
          language="python"
          code={`class TokenEmbeddings(nn.Module):
    """
    Token embedding layer.
    
    Converts token indices to dense vectors of dimension d_model.
    
    Input:  (B, S) integer token indices in range [0, vocab_size)
    Output: (B, S, d_model) continuous embeddings
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) as in original paper
        return self.embedding(x) * math.sqrt(self.d_model)`}
        />

        <Callout type="info" title="Embedding Scaling">
          Embeddings are scaled by <code>√d_model</code> to maintain variance. This 
          ensures the embedding magnitudes are comparable to positional embeddings and 
          attention score scales.
        </Callout>

        <h2 id="fixed-positional-embeddings">FixedPositionalEmbeddings</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> FixedPositionalEmbeddings(nn.Module)
        </div>

        <p>
          Sinusoidal positional embeddings as described in the original Transformer paper. 
          These are not learned but computed using a fixed formula.
        </p>

        <ApiTable 
          parameters={[
            { name: "d_model", type: "int", description: "Dimension of the positional vectors." },
            { name: "max_seq_length", type: "int", description: "Maximum sequence length supported." },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          language="python"
          code={`class FixedPositionalEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings (non-learnable).
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Deterministic (no learned parameters)
    - Can extrapolate to longer sequences than seen during training
    - Relative positions have consistent representations
    """
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        
        # Precompute positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) or (B, S, d_model)
        seq_length = x.size(1)
        return self.pe[:, :seq_length, :]  # (1, S, d_model)`}
        />

        <h2 id="learned-positional-embeddings">LearnedPositionalEmbeddings</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> LearnedPositionalEmbeddings(nn.Module)
        </div>

        <p>
          Learnable positional embeddings where each position has a trainable vector. 
          This is the default in BardGPT.
        </p>

        <ApiTable 
          parameters={[
            { name: "d_model", type: "int", description: "Dimension of the positional vectors." },
            { name: "max_seq_length", type: "int", description: "Maximum sequence length supported." },
          ]} 
          title="Constructor Parameters" 
        />

        <CodeBlock
          language="python"
          code={`class LearnedPositionalEmbeddings(nn.Module):
    """
    Learnable positional embeddings.
    
    Each position has an independent learnable vector.
    
    Properties:
    - Learned from data
    - Cannot extrapolate beyond max_seq_length
    - More flexible than fixed embeddings
    """
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S)
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device)
        return self.embedding(positions)  # (S, d_model)`}
        />

        <h2 id="embedding-block">EmbeddingBlock</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> EmbeddingBlock(nn.Module)
        </div>

        <p>
          Combined embedding layer that adds token embeddings and positional embeddings, 
          then applies dropout.
        </p>

        <CodeBlock
          language="python"
          code={`class EmbeddingBlock(nn.Module):
    """
    Combined token and positional embedding layer.
    
    Output = Dropout(TokenEmbed(x) + PosEmbed(x))
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.token_embed = TokenEmbeddings(config.vocab_size, config.d_model)
        
        if config.use_fixed_positional_embeddings:
            self.pos_embed = FixedPositionalEmbeddings(
                config.d_model, 
                config.max_seq_length
            )
        else:
            self.pos_embed = LearnedPositionalEmbeddings(
                config.d_model, 
                config.max_seq_length
            )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices (B, S)
        
        Returns:
            Combined embeddings (B, S, d_model)
        """
        tok_emb = self.token_embed(x)  # (B, S, d_model)
        pos_emb = self.pos_embed(x)    # (S, d_model) or (1, S, d_model)
        
        return self.dropout(tok_emb + pos_emb)`}
        />

        <h2 id="constraints">Sequence Length Constraints</h2>
        <p>
          Both positional embedding types have a maximum sequence length constraint:
        </p>

        <ul>
          <li>
            <strong>seq_length</strong> — The actual sequence length used during training
          </li>
          <li>
            <strong>max_seq_length</strong> — The maximum length the model can handle
          </li>
        </ul>

        <Callout type="warning" title="Length Limit">
          Input sequences must satisfy <code>seq_length ≤ max_seq_length</code>. 
          For learned embeddings, exceeding this limit will cause an index error. 
          For fixed embeddings, you can technically extrapolate, but performance 
          may degrade.
        </Callout>

        <CodeBlock
          language="python"
          code={`# Valid: seq_length (256) <= max_seq_length (512)
config = TransformerConfig(
    seq_length=256,
    max_seq_length=512
)

# Will fail at runtime:
# config = TransformerConfig(seq_length=1024, max_seq_length=512)`}
        />
      </div>
    </DocLayout>
  );
}
