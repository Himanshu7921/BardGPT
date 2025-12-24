import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "sampling-process", text: "Sampling Process", level: 2 },
  { id: "temperature", text: "Temperature", level: 2 },
  { id: "top-k-sampling", text: "Top-K Sampling", level: 2 },
  { id: "top-p-sampling", text: "Top-P (Nucleus) Sampling", level: 2 },
  { id: "sliding-window", text: "Sliding Window Generation", level: 2 },
  { id: "practical-examples", text: "Practical Examples", level: 2 },
];

const samplingParams = [
  { name: "prompt", type: "str", description: "Initial text to condition the generation.", default: '""' },
  { name: "max_tokens", type: "int", description: "Maximum number of tokens to generate.", default: "500" },
  { name: "temperature", type: "float", description: "Controls randomness. Lower = more deterministic.", default: "1.0" },
  { name: "top_k", type: "int", description: "Limit sampling to top K most likely tokens. 0 = disabled.", default: "0" },
  { name: "top_p", type: "float", description: "Nucleus sampling threshold. 1.0 = disabled.", default: "1.0" },
];

export default function Sampling() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Sampling" }]} />
      
      <div className="doc-prose">
        <h1>Sampling & Text Generation</h1>
        
        <p className="text-lg text-muted-foreground">
          How BardGPT generates text using temperature scaling, top-k, and top-p 
          sampling strategies.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          Text generation in BardGPT is autoregressive: the model generates one token 
          at a time, using previously generated tokens as context. The sampling strategy 
          determines how the next token is selected from the model's probability distribution.
        </p>

        <ApiTable parameters={samplingParams} title="Sampling Parameters" />

        <h2 id="sampling-process">Sampling Process</h2>
        <p>
          The core generation loop follows these steps:
        </p>

        <CodeBlock
          filename="transformer/sample.py"
          language="python"
          code={`def generate(
    model: nn.Module,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    data_loader = None
) -> str:
    """
    Generate text autoregressively from a prompt.
    
    Args:
        model: Trained transformer model
        prompt: Initial text string
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-K filtering (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        data_loader: DataLoader with stoi/itos mappings
    
    Returns:
        Generated text string
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    tokens = [data_loader.stoi[c] for c in prompt]
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get context (sliding window if needed)
            context = tokens[:, -model.config.seq_length:]
            
            # Forward pass
            logits = model(context)
            logits = logits[:, -1, :]  # Last position only
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = top_p_logits(logits, top_p)
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode tokens to string
    output = ''.join([data_loader.itos[t.item()] for t in tokens[0]])
    return output`}
        />

        <h2 id="temperature">Temperature</h2>
        <p>
          Temperature controls the randomness of sampling by scaling the logits before 
          applying softmax:
        </p>

        <div className="bg-muted/50 rounded-lg p-4 my-4 text-center font-mono">
          P(token) = softmax(logits / temperature)
        </div>

        <ul>
          <li><strong>temperature = 1.0</strong>: Standard probability distribution</li>
          <li><strong>temperature &lt; 1.0</strong>: Sharper distribution (more deterministic)</li>
          <li><strong>temperature &gt; 1.0</strong>: Flatter distribution (more random)</li>
          <li><strong>temperature → 0</strong>: Approaches greedy decoding (always pick max)</li>
        </ul>

        <CodeBlock
          language="python"
          code={`# Lower temperature = more focused, repetitive text
generate(model, "ROMEO:", temperature=0.3)
# Output: "ROMEO: I love thee, I love thee, I love thee..."

# Higher temperature = more creative, sometimes nonsensical
generate(model, "ROMEO:", temperature=1.2)
# Output: "ROMEO: What zephyrs blow through castle'd moonbeams..."`}
        />

        <h2 id="top-k-sampling">Top-K Sampling</h2>
        <p>
          Top-K sampling limits the candidate tokens to the K most likely options, 
          then renormalizes probabilities:
        </p>

        <CodeBlock
          language="python"
          code={`def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only the top-k tokens.
    
    Args:
        logits: (batch_size, vocab_size) unnormalized log probabilities
        k: Number of top tokens to keep
    
    Returns:
        Filtered logits with -inf for tokens outside top-k
    """
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

        <Callout type="tip" title="Recommended Values">
          For character-level models like BardGPT, <code>top_k=40</code> is a good 
          starting point. This limits sampling to the 40 most likely characters while 
          maintaining diversity.
        </Callout>

        <h2 id="top-p-sampling">Top-P (Nucleus) Sampling</h2>
        <p>
          Top-P sampling (also called nucleus sampling) selects from the smallest set 
          of tokens whose cumulative probability exceeds the threshold P:
        </p>

        <CodeBlock
          language="python"
          code={`def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.
    
    Keeps the smallest set of tokens with cumulative probability >= p.
    
    Args:
        logits: (batch_size, vocab_size) unnormalized log probabilities
        p: Cumulative probability threshold (0.0 to 1.0)
    
    Returns:
        Filtered logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    
    # Scatter back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, 
        index=sorted_indices, 
        src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    return logits`}
        />

        <p>
          Top-P is often preferred over Top-K because it adapts to the distribution 
          shape. When the model is confident, fewer tokens are considered; when 
          uncertain, more options remain available.
        </p>

        <h2 id="sliding-window">Sliding Window Generation</h2>
        <p>
          When generating text longer than the model's context window 
          (<code>seq_length</code>), a sliding window approach is used:
        </p>

        <CodeBlock
          language="python"
          code={`# If tokens exceed seq_length, use only the last seq_length tokens
context = tokens[:, -model.config.seq_length:]

# This allows generating arbitrarily long text while staying
# within the model's positional embedding limits`}
        />

        <Callout type="warning" title="Context Limitation">
          The sliding window means the model loses access to tokens outside the 
          current window. For long generations, this can lead to repetition or 
          loss of coherence with earlier content.
        </Callout>

        <h2 id="practical-examples">Practical Examples</h2>

        <p><strong>Conservative generation (low temperature, high top-k):</strong></p>
        <CodeBlock
          language="python"
          code={`text = generate(
    model,
    prompt="HAMLET:",
    max_tokens=200,
    temperature=0.5,
    top_k=20
)
# Produces focused, coherent text close to training distribution`}
        />

        <p><strong>Creative generation (balanced settings):</strong></p>
        <CodeBlock
          language="python"
          code={`text = generate(
    model,
    prompt="HAMLET:",
    max_tokens=200,
    temperature=0.8,
    top_p=0.9
)
# Balances coherence with creativity`}
        />

        <p><strong>Experimental generation (high randomness):</strong></p>
        <CodeBlock
          language="python"
          code={`text = generate(
    model,
    prompt="HAMLET:",
    max_tokens=200,
    temperature=1.2,
    top_p=0.95
)
# More surprising outputs, occasionally incoherent`}
        />
      </div>
    </DocLayout>
  );
}
