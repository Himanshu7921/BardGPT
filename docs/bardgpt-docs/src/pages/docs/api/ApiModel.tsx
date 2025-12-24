import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "decoder-only-transformer-model", text: "DecoderOnlyTransformerModel", level: 2 },
  { id: "constructor", text: "Constructor", level: 3 },
  { id: "forward-method", text: "forward()", level: 3 },
  { id: "tensor-shapes", text: "Tensor Shapes", level: 2 },
  { id: "weight-tying", text: "Weight Tying", level: 2 },
  { id: "usage-example", text: "Usage Example", level: 2 },
];

const constructorParams = [
  { name: "config", type: "TransformerConfig", description: "Configuration dataclass containing all model hyperparameters." },
];

const forwardParams = [
  { name: "x", type: "torch.Tensor", description: "Input token indices of shape (batch_size, seq_length)." },
  { name: "mask", type: "torch.Tensor | None", description: "Optional attention mask. If None, a causal mask is generated automatically.", default: "None" },
];

export default function ApiModel() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.model" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.model</h1>
        
        <p className="text-lg text-muted-foreground">
          The main model module containing the <code>DecoderOnlyTransformerModel</code> class.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module defines the top-level transformer model that orchestrates all 
          components: embeddings, decoder blocks, and the output projection. It serves 
          as the entry point for training and inference.
        </p>

        <CodeBlock
          language="python"
          code={`from transformer.model import DecoderOnlyTransformerModel
from transformer.utils import TransformerConfig

config = TransformerConfig(d_model=384, n_layers=6)
model = DecoderOnlyTransformerModel(config)`}
        />

        <h2 id="decoder-only-transformer-model">DecoderOnlyTransformerModel</h2>
        
        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> DecoderOnlyTransformerModel(nn.Module)
        </div>

        <p>
          The complete decoder-only transformer model for autoregressive language modeling. 
          Combines token embeddings, positional embeddings, a stack of decoder blocks, 
          and a linear output head.
        </p>

        <h3 id="constructor">Constructor</h3>
        
        <CodeBlock
          language="python"
          code={`def __init__(self, config: TransformerConfig)`}
        />

        <ApiTable parameters={constructorParams} title="Parameters" />

        <p><strong>Attributes created:</strong></p>
        <ul>
          <li><code>config</code> — Stored configuration object</li>
          <li><code>embedding</code> — <code>EmbeddingBlock</code> for token + positional embeddings</li>
          <li><code>decoder</code> — <code>Decoder</code> containing N decoder blocks</li>
          <li><code>norm</code> — Final <code>LayerNorm</code> before output projection</li>
          <li><code>linear_head</code> — <code>LinearProjection</code> to vocabulary size</li>
        </ul>

        <CodeBlock
          filename="transformer/model.py"
          language="python"
          code={`class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer (tokens + positions)
        self.embedding = EmbeddingBlock(config)
        
        # Stack of N decoder blocks
        self.decoder = Decoder(config)
        
        # Final layer normalization
        self.norm = LayerNorm(config.d_model)
        
        # Output projection to vocabulary
        self.linear_head = LinearProjection(config.d_model, config.vocab_size)
        
        # Weight tying: share embedding and output weights
        self.linear_head.linear.weight = self.embedding.token_embed.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)`}
        />

        <h3 id="forward-method">forward()</h3>

        <CodeBlock
          language="python"
          code={`def forward(
    self, 
    x: torch.Tensor, 
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor`}
        />

        <ApiTable parameters={forwardParams} title="Parameters" />

        <p><strong>Returns:</strong></p>
        <ul>
          <li>
            <code>torch.Tensor</code> — Output logits of shape 
            <code>(batch_size, seq_length, vocab_size)</code>
          </li>
        </ul>

        <CodeBlock
          language="python"
          code={`def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward pass through the complete model.
    
    Args:
        x: Input token indices (B, S)
        mask: Optional attention mask (S, S). If None, causal mask is created.
    
    Returns:
        Logits over vocabulary (B, S, vocab_size)
    """
    B, S = x.shape
    
    # Create causal mask if not provided
    if mask is None:
        mask = causal_mask(S, x.device)
    
    # Embeddings: (B, S) -> (B, S, d_model)
    x = self.embedding(x)
    
    # Decoder blocks: (B, S, d_model) -> (B, S, d_model)
    x = self.decoder(x, mask)
    
    # Final normalization
    x = self.norm(x)
    
    # Project to vocabulary: (B, S, d_model) -> (B, S, vocab_size)
    logits = self.linear_head(x)
    
    return logits`}
        />

        <h2 id="tensor-shapes">Tensor Shapes</h2>
        <p>
          Throughout the forward pass, tensors maintain consistent batch and sequence 
          dimensions:
        </p>

        <div className="overflow-x-auto">
          <table className="api-table">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Tensor Shape</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Input</td>
                <td><code>(B, S)</code></td>
                <td>Token indices (integers)</td>
              </tr>
              <tr>
                <td>After Embedding</td>
                <td><code>(B, S, d_model)</code></td>
                <td>Token + position embeddings</td>
              </tr>
              <tr>
                <td>Through Decoder</td>
                <td><code>(B, S, d_model)</code></td>
                <td>Hidden states (preserved)</td>
              </tr>
              <tr>
                <td>After Norm</td>
                <td><code>(B, S, d_model)</code></td>
                <td>Normalized hidden states</td>
              </tr>
              <tr>
                <td>Output</td>
                <td><code>(B, S, vocab_size)</code></td>
                <td>Logits over vocabulary</td>
              </tr>
            </tbody>
          </table>
        </div>

        <Callout type="info" title="Notation">
          <code>B</code> = batch size, <code>S</code> = sequence length, 
          <code>d_model</code> = embedding dimension
        </Callout>

        <h2 id="weight-tying">Weight Tying</h2>
        <p>
          The model implements weight tying between the input embedding matrix and the 
          output projection layer. This is a common technique that:
        </p>
        <ul>
          <li>Reduces total parameter count by <code>vocab_size × d_model</code></li>
          <li>Encourages consistent representations between input and output</li>
          <li>Often improves generalization</li>
        </ul>

        <CodeBlock
          language="python"
          code={`# Weight tying in __init__:
self.linear_head.linear.weight = self.embedding.token_embed.embedding.weight`}
        />

        <h2 id="usage-example">Usage Example</h2>

        <CodeBlock
          language="python"
          code={`import torch
from transformer.model import DecoderOnlyTransformerModel
from transformer.utils import TransformerConfig

# Create model
config = TransformerConfig(
    vocab_size=65,
    d_model=384,
    n_layers=6,
    n_heads=6
)
model = DecoderOnlyTransformerModel(config)

# Forward pass
batch_size, seq_length = 4, 256
x = torch.randint(0, config.vocab_size, (batch_size, seq_length))

logits = model(x)  # (4, 256, 65)

# Get predictions for next token
next_token_logits = logits[:, -1, :]  # (4, 65)
next_token = torch.argmax(next_token_logits, dim=-1)  # (4,)`}
        />
      </div>
    </DocLayout>
  );
}
