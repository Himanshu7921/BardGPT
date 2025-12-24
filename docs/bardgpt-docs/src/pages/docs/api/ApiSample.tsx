import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "generate-text", text: "generate_text()", level: 2 },
  { id: "sampling-strategies", text: "Sampling Strategies", level: 2 },
  { id: "sliding-window", text: "Sliding Window", level: 2 },
  { id: "output-formatting", text: "Output Formatting", level: 2 },
  { id: "usage-examples", text: "Usage Examples", level: 2 },
];

const generateParams = [
  { name: "model", type: "nn.Module", description: "Trained transformer model." },
  { name: "prompt", type: "str", description: "Initial text to condition generation.", default: '""' },
  { name: "max_tokens", type: "int", description: "Maximum number of tokens to generate.", default: "500" },
  { name: "temperature", type: "float", description: "Sampling temperature (higher = more random).", default: "1.0" },
  { name: "top_k", type: "int", description: "Top-K sampling (0 = disabled).", default: "0" },
  { name: "top_p", type: "float", description: "Nucleus sampling threshold (1.0 = disabled).", default: "1.0" },
  { name: "data_loader", type: "DataLoader", description: "DataLoader with vocabulary mappings." },
];

export default function ApiSample() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.sample" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.sample</h1>
        
        <p className="text-lg text-muted-foreground">
          Text generation and sampling functions for autoregressive inference.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module provides functions for generating text from a trained model. 
          The generation process is autoregressive: tokens are generated one at a time, 
          with each new token conditioned on all previously generated tokens.
        </p>

        <h2 id="generate-text">generate_text()</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">def</span> generate_text(
            <br />
            <span className="pl-4">model: nn.Module,</span>
            <br />
            <span className="pl-4">prompt: str = "",</span>
            <br />
            <span className="pl-4">max_tokens: int = 500,</span>
            <br />
            <span className="pl-4">temperature: float = 1.0,</span>
            <br />
            <span className="pl-4">top_k: int = 0,</span>
            <br />
            <span className="pl-4">top_p: float = 1.0,</span>
            <br />
            <span className="pl-4">data_loader: DataLoader = None</span>
            <br />
          ) -&gt; str
        </div>

        <p>
          Generate text autoregressively from a trained model.
        </p>

        <ApiTable parameters={generateParams} title="Parameters" />

        <p><strong>Returns:</strong> <code>str</code> — Generated text including the prompt</p>

        <CodeBlock
          filename="transformer/sample.py"
          language="python"
          code={`def generate_text(
    model: nn.Module,
    prompt: str = "",
    max_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    data_loader = None
) -> str:
    """
    Generate text from a trained model.
    
    The generation process:
    1. Encode prompt to token indices
    2. For each new token:
       a. Get model predictions for current context
       b. Apply temperature scaling
       c. Apply top-k and/or top-p filtering
       d. Sample next token from probability distribution
       e. Append to context (with sliding window if needed)
    3. Decode token indices back to text
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    if prompt:
        tokens = torch.tensor(
            [data_loader.stoi[c] for c in prompt],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
    else:
        # Start with random token if no prompt
        tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Sliding window: keep only last seq_length tokens
            context = tokens[:, -model.config.seq_length:]
            
            # Get logits
            logits = model(context)
            logits = logits[:, -1, :]  # Only last position
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-K filtering
            if top_k > 0:
                logits = top_k_logits(logits, top_k)
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                logits = top_p_logits(logits, top_p)
            
            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode to string
    generated = ''.join([data_loader.itos[t.item()] for t in tokens[0]])
    return generated`}
        />

        <h2 id="sampling-strategies">Sampling Strategies</h2>

        <h3>Temperature</h3>
        <p>
          Temperature controls the sharpness of the probability distribution:
        </p>
        <ul>
          <li><code>temperature &lt; 1.0</code>: More focused, deterministic output</li>
          <li><code>temperature = 1.0</code>: Original distribution</li>
          <li><code>temperature &gt; 1.0</code>: More random, creative output</li>
        </ul>

        <CodeBlock
          language="python"
          code={`# Apply temperature
logits = logits / temperature

# As temperature -> 0, distribution becomes one-hot (greedy)
# As temperature -> infinity, distribution becomes uniform`}
        />

        <h3>Top-K Sampling</h3>
        <p>
          Limits sampling to the K most likely tokens:
        </p>
        <CodeBlock
          language="python"
          code={`# Keep only top 40 most likely tokens
logits = top_k_logits(logits, k=40)

# All other tokens get -inf probability
# Probabilities are renormalized over remaining tokens`}
        />

        <h3>Top-P (Nucleus) Sampling</h3>
        <p>
          Dynamically selects the smallest set of tokens whose cumulative probability 
          exceeds the threshold:
        </p>
        <CodeBlock
          language="python"
          code={`# Keep smallest set with cumulative probability >= 0.9
logits = top_p_logits(logits, p=0.9)

# Adapts to distribution shape:
# - Confident predictions: fewer tokens considered
# - Uncertain predictions: more tokens considered`}
        />

        <Callout type="tip" title="Combining Strategies">
          You can combine temperature with either top-k or top-p (or both). 
          A common setup: <code>temperature=0.8, top_p=0.9</code>
        </Callout>

        <h2 id="sliding-window">Sliding Window</h2>
        <p>
          When generating text longer than the model's maximum sequence length, 
          a sliding window keeps only the most recent tokens:
        </p>

        <CodeBlock
          language="python"
          code={`# If tokens exceed seq_length, truncate from the left
context = tokens[:, -model.config.seq_length:]

# Example with seq_length=256:
# tokens has 300 characters -> context uses last 256
# Earlier characters are "forgotten" by the model`}
        />

        <Callout type="warning" title="Context Limitation">
          The sliding window means the model cannot reference tokens beyond its 
          context window. For very long generations, this may cause:
          <ul className="mt-2 mb-0">
            <li>Loss of long-range coherence</li>
            <li>Repetition of earlier content</li>
            <li>Forgetting of character names or plot points</li>
          </ul>
        </Callout>

        <h2 id="output-formatting">Output Formatting</h2>
        <p>
          The generated text is returned as a raw string including the original prompt. 
          To extract only the generated portion:
        </p>

        <CodeBlock
          language="python"
          code={`prompt = "ROMEO:"
full_output = generate_text(model, prompt=prompt, ...)

# Extract only generated text
generated_only = full_output[len(prompt):]
print(generated_only)`}
        />

        <h2 id="usage-examples">Usage Examples</h2>

        <h3>Basic Generation</h3>
        <CodeBlock
          language="python"
          code={`from transformer.sample import generate_text
from transformer.utils import load_checkpoint, DataLoader, TransformerConfig

# Load model and data
model, config = load_checkpoint("checkpoints/model_step_5000.pt")
data_loader = DataLoader("data/tiny_shakespeare.txt", config)

# Generate
text = generate_text(
    model=model,
    prompt="HAMLET:",
    max_tokens=300,
    data_loader=data_loader
)
print(text)`}
        />

        <h3>Conservative Generation</h3>
        <CodeBlock
          language="python"
          code={`# Low temperature, restricted vocabulary
text = generate_text(
    model=model,
    prompt="JULIET:",
    max_tokens=200,
    temperature=0.5,
    top_k=20,
    data_loader=data_loader
)
# Produces focused, coherent text`}
        />

        <h3>Creative Generation</h3>
        <CodeBlock
          language="python"
          code={`# Higher temperature, nucleus sampling
text = generate_text(
    model=model,
    prompt="PROSPERO:",
    max_tokens=200,
    temperature=0.9,
    top_p=0.95,
    data_loader=data_loader
)
# Produces more varied, surprising text`}
        />

        <h3>Batch Generation</h3>
        <CodeBlock
          language="python"
          code={`# Generate multiple samples with different prompts
prompts = ["ROMEO:", "JULIET:", "HAMLET:", "MACBETH:"]

for prompt in prompts:
    text = generate_text(
        model=model,
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        data_loader=data_loader
    )
    print(f"\\n{'='*40}\\n{text}")`}
        />
      </div>
    </DocLayout>
  );
}
