import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "installation", text: "Installation", level: 2 },
  { id: "configuration", text: "Configuration", level: 2 },
  { id: "training", text: "Training the Model", level: 2 },
  { id: "generating-text", text: "Generating Text", level: 2 },
];

export default function QuickStart() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Quick Start" }]} />
      
      <div className="doc-prose">
        <h1>Quick Start</h1>
        
        <p className="text-lg text-muted-foreground">
          Get BardGPT running on your machine in just a few steps.
        </p>

        <h2 id="installation">Installation</h2>
        <p>Clone the repository and install dependencies:</p>

        <CodeBlock
          language="bash"
          code={`# Clone the repository
git clone https://github.com/Himanshu7921/BardGPT.git
cd BardGPT

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt`}
        />

        <Callout type="tip" title="GPU Support">
          For faster training, ensure you have PyTorch installed with CUDA support. 
          Visit <a href="https://pytorch.org" target="_blank" rel="noopener noreferrer">pytorch.org</a> for 
          installation instructions specific to your system.
        </Callout>

        <h2 id="configuration">Configuration</h2>
        <p>
          Model configuration is handled through the <code>TransformerConfig</code> dataclass 
          in <code>transformer/utils.py</code>. Here are the default settings:
        </p>

        <CodeBlock
          filename="transformer/utils.py"
          language="python"
          code={`@dataclass
class TransformerConfig:
    vocab_size: int = 65           # Character vocabulary size
    d_model: int = 384             # Embedding dimension
    d_ff: int = 1536               # Feedforward dimension (4x d_model)
    n_layers: int = 6              # Number of decoder blocks
    n_heads: int = 6               # Attention heads
    dropout: float = 0.2           # Dropout probability
    seq_length: int = 256          # Context window size
    max_seq_length: int = 512      # Maximum sequence length
    steps: int = 5000              # Training steps
    use_fixed_positional_embeddings: bool = False`}
        />

        <h2 id="training">Training the Model</h2>
        <p>To train BardGPT on the Tiny Shakespeare dataset:</p>

        <CodeBlock
          language="python"
          code={`from transformer.train import train_model
from transformer.utils import TransformerConfig

# Create configuration
config = TransformerConfig(
    d_model=384,
    n_layers=6,
    n_heads=6,
    steps=5000
)

# Start training
model = train_model(config)`}
        />

        <p>Or run the training script directly:</p>

        <CodeBlock
          language="bash"
          code={`python -m transformer.train`}
        />

        <p>
          Training progress will be logged to the console and TensorBoard. Checkpoints are 
          saved automatically to the <code>checkpoints/</code> directory.
        </p>

        <h2 id="generating-text">Generating Text</h2>
        <p>After training, generate text using the sample module:</p>

        <CodeBlock
          language="python"
          code={`from transformer.sample import generate_text
from transformer.utils import load_checkpoint

# Load trained model
model, config = load_checkpoint("checkpoints/model_step_5000.pt")

# Generate text
prompt = "ROMEO:"
generated = generate_text(
    model=model,
    prompt=prompt,
    max_tokens=500,
    temperature=0.8,
    top_k=40
)

print(generated)`}
        />

        <p>Example output:</p>

        <CodeBlock
          language="text"
          code={`ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon...`}
        />

        <Callout type="info" title="Sampling Parameters">
          Adjust <code>temperature</code> (0.1–1.0) for creativity and <code>top_k</code> to 
          limit vocabulary diversity. Lower values produce more focused, repetitive text; 
          higher values increase variety.
        </Callout>
      </div>
    </DocLayout>
  );
}
