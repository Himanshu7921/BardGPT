import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "transformer-config", text: "TransformerConfig", level: 2 },
  { id: "causal-mask", text: "causal_mask()", level: 2 },
  { id: "load-checkpoint", text: "load_checkpoint()", level: 2 },
  { id: "save-checkpoint", text: "save_checkpoint()", level: 2 },
  { id: "generate", text: "generate()", level: 2 },
  { id: "dataloader", text: "DataLoader", level: 2 },
  { id: "top-k-logits", text: "top_k_logits()", level: 2 },
  { id: "top-p-logits", text: "top_p_logits()", level: 2 },
  { id: "print-model-summary", text: "print_model_summary()", level: 2 },
  { id: "checkpoint-name", text: "checkpoint_name()", level: 2 },
];

export default function ApiUtils() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.utils" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.utils</h1>
        
        <p className="text-lg text-muted-foreground">
          Configuration, data loading, checkpointing, and utility functions.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module contains essential utilities for configuring models, loading data, 
          saving/loading checkpoints, and text generation.
        </p>

        <h2 id="transformer-config">TransformerConfig</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-muted-foreground">@dataclass</span><br/>
          <span className="text-primary">class</span> TransformerConfig
        </div>

        <p>
          Dataclass containing all model and training hyperparameters.
        </p>

        <CodeBlock
          language="python"
          code={`@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    
    # Vocabulary
    vocab_size: int = 65
    
    # Model dimensions
    d_model: int = 384          # Hidden state dimension
    d_ff: int = 1536            # Feedforward hidden dimension
    
    # Architecture
    n_layers: int = 6           # Number of decoder blocks
    n_heads: int = 6            # Attention heads
    
    # Regularization
    dropout: float = 0.2
    
    # Sequence length
    seq_length: int = 256       # Training context window
    max_seq_length: int = 512   # Maximum supported length
    
    # Training
    steps: int = 5000           # Training iterations
    
    # Positional embeddings
    use_fixed_positional_embeddings: bool = False`}
        />

        <h2 id="causal-mask">causal_mask()</h2>

        <CodeBlock
          language="python"
          code={`def causal_mask(seq_length: int, device: torch.device) -> torch.Tensor`}
        />

        <ApiTable 
          parameters={[
            { name: "seq_length", type: "int", description: "Length of the sequence." },
            { name: "device", type: "torch.device", description: "Device to create the mask on." },
          ]} 
          title="Parameters" 
        />

        <p><strong>Returns:</strong> <code>torch.Tensor</code> of shape <code>(seq_length, seq_length)</code></p>

        <CodeBlock
          language="python"
          code={`def causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.
    
    The mask prevents attention to future positions:
    - mask[i, j] = 0 if j <= i (allowed)
    - mask[i, j] = -inf if j > i (blocked)
    
    Example for seq_length=4:
    [[  0, -inf, -inf, -inf],
     [  0,    0, -inf, -inf],
     [  0,    0,    0, -inf],
     [  0,    0,    0,    0]]
    """
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask`}
        />

        <h2 id="load-checkpoint">load_checkpoint()</h2>

        <CodeBlock
          language="python"
          code={`def load_checkpoint(filepath: str) -> tuple[nn.Module, TransformerConfig]`}
        />

        <ApiTable 
          parameters={[
            { name: "filepath", type: "str", description: "Path to the checkpoint file (.pt)." },
          ]} 
          title="Parameters" 
        />

        <p><strong>Returns:</strong> Tuple of <code>(model, config)</code></p>

        <CodeBlock
          language="python"
          code={`def load_checkpoint(filepath: str) -> tuple[nn.Module, TransformerConfig]:
    """
    Load a model checkpoint.
    
    Returns:
        model: Loaded model with weights
        config: TransformerConfig used to create the model
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    config = checkpoint['config']
    
    model = DecoderOnlyTransformerModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config`}
        />

        <h2 id="save-checkpoint">save_checkpoint()</h2>

        <CodeBlock
          language="python"
          code={`def save_checkpoint(
    model: nn.Module, 
    config: TransformerConfig, 
    step: int, 
    path: str = "checkpoints"
) -> str`}
        />

        <ApiTable 
          parameters={[
            { name: "model", type: "nn.Module", description: "Model to save." },
            { name: "config", type: "TransformerConfig", description: "Model configuration." },
            { name: "step", type: "int", description: "Current training step." },
            { name: "path", type: "str", description: "Directory to save checkpoint.", default: '"checkpoints"' },
          ]} 
          title="Parameters" 
        />

        <p><strong>Returns:</strong> <code>str</code> — Path to saved checkpoint file</p>

        <CodeBlock
          language="python"
          code={`def save_checkpoint(model, config, step, path="checkpoints"):
    """Save model checkpoint with configuration."""
    os.makedirs(path, exist_ok=True)
    
    filepath = os.path.join(path, f"model_step_{step}.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': step
    }
    
    torch.save(checkpoint, filepath)
    return filepath`}
        />

        <h2 id="generate">generate()</h2>

        <p>See the <a href="/docs/api/sample">transformer.sample</a> documentation for the full <code>generate()</code> function.</p>

        <h2 id="dataloader">DataLoader</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">class</span> DataLoader
        </div>

        <p>
          Handles text data loading, vocabulary building, and batch generation.
        </p>

        <CodeBlock
          language="python"
          code={`class DataLoader:
    """
    Data loader for character-level language modeling.
    
    Attributes:
        stoi: Character to index mapping
        itos: Index to character mapping
        train_data: Training split tensor
        val_data: Validation split tensor
    """
    def __init__(self, filepath: str, config: TransformerConfig):
        # Read text file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Build vocabulary (character-level)
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode full text
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # 90/10 train/val split
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        self.seq_length = config.seq_length
        self.batch_size = 64
    
    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of sequences.
        
        Args:
            split: "train" or "val"
        
        Returns:
            x: Input sequences (batch_size, seq_length)
            y: Target sequences (batch_size, seq_length)
        """
        data = self.train_data if split == "train" else self.val_data
        
        # Random starting indices
        ix = torch.randint(len(data) - self.seq_length, (self.batch_size,))
        
        # Sliding window: x[t] predicts y[t] = x[t+1]
        x = torch.stack([data[i:i+self.seq_length] for i in ix])
        y = torch.stack([data[i+1:i+self.seq_length+1] for i in ix])
        
        return x, y`}
        />

        <Callout type="info" title="Sliding Window">
          Batches are created using sliding windows over the text. The target sequence 
          is the input shifted by one position, creating the autoregressive training 
          signal where each token predicts the next.
        </Callout>

        <h2 id="top-k-logits">top_k_logits()</h2>

        <CodeBlock
          language="python"
          code={`def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor`}
        />

        <ApiTable 
          parameters={[
            { name: "logits", type: "torch.Tensor", description: "Unnormalized log probabilities (B, vocab_size)." },
            { name: "k", type: "int", description: "Number of top tokens to keep. 0 disables filtering." },
          ]} 
          title="Parameters" 
        />

        <CodeBlock
          language="python"
          code={`def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filter logits to keep only top-k values."""
    if k == 0:
        return logits
    
    values, _ = torch.topk(logits, k)
    min_value = values[:, -1].unsqueeze(-1)
    
    return torch.where(
        logits < min_value,
        torch.full_like(logits, float('-inf')),
        logits
    )`}
        />

        <h2 id="top-p-logits">top_p_logits()</h2>

        <CodeBlock
          language="python"
          code={`def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor`}
        />

        <ApiTable 
          parameters={[
            { name: "logits", type: "torch.Tensor", description: "Unnormalized log probabilities (B, vocab_size)." },
            { name: "p", type: "float", description: "Cumulative probability threshold (0.0 to 1.0)." },
          ]} 
          title="Parameters" 
        />

        <CodeBlock
          language="python"
          code={`def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep smallest set of tokens with cumulative prob >= p.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    return logits.masked_fill(indices_to_remove, float('-inf'))`}
        />

        <h2 id="print-model-summary">print_model_summary()</h2>

        <CodeBlock
          language="python"
          code={`def print_model_summary(model: nn.Module) -> None`}
        />

        <p>Prints a summary of the model architecture and parameter counts.</p>

        <CodeBlock
          language="python"
          code={`def print_model_summary(model: nn.Module) -> None:
    """Print model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")`}
        />

        <h2 id="checkpoint-name">checkpoint_name()</h2>

        <CodeBlock
          language="python"
          code={`def checkpoint_name(step: int) -> str`}
        />

        <p>Generates a standardized checkpoint filename.</p>

        <CodeBlock
          language="python"
          code={`def checkpoint_name(step: int) -> str:
    """Generate checkpoint filename for a given step."""
    return f"model_step_{step}.pt"`}
        />
      </div>
    </DocLayout>
  );
}
