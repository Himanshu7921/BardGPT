import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "transformer-config", text: "TransformerConfig", level: 2 },
  { id: "model-parameters", text: "Model Parameters", level: 3 },
  { id: "training-parameters", text: "Training Parameters", level: 3 },
  { id: "example-configurations", text: "Example Configurations", level: 2 },
];

const modelParams = [
  { name: "vocab_size", type: "int", description: "Size of the token vocabulary. For character-level models, this is the number of unique characters.", default: "65" },
  { name: "d_model", type: "int", description: "Dimensionality of token embeddings and hidden states throughout the model.", default: "384" },
  { name: "d_ff", type: "int", description: "Dimensionality of the feedforward network hidden layer. Typically 4× d_model.", default: "1536" },
  { name: "n_layers", type: "int", description: "Number of stacked decoder blocks in the transformer.", default: "6" },
  { name: "n_heads", type: "int", description: "Number of attention heads. Must divide d_model evenly.", default: "6" },
  { name: "dropout", type: "float", description: "Dropout probability applied throughout the model.", default: "0.2" },
  { name: "seq_length", type: "int", description: "Length of input sequences during training (context window).", default: "256" },
  { name: "max_seq_length", type: "int", description: "Maximum sequence length the positional embeddings can handle.", default: "512" },
];

const trainingParams = [
  { name: "steps", type: "int", description: "Total number of training steps (gradient updates).", default: "5000" },
  { name: "use_fixed_positional_embeddings", type: "bool", description: "Use sinusoidal (fixed) positional embeddings instead of learned.", default: "False" },
];

export default function Configuration() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Configuration" }]} />
      
      <div className="doc-prose">
        <h1>Configuration</h1>
        
        <p className="text-lg text-muted-foreground">
          Complete reference for all BardGPT configuration options and hyperparameters.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          All model and training configuration is centralized in the <code>TransformerConfig</code> 
          dataclass located in <code>transformer/utils.py</code>. This single configuration 
          object controls the model architecture, training behavior, and sampling parameters.
        </p>

        <h2 id="transformer-config">TransformerConfig</h2>
        
        <CodeBlock
          filename="transformer/utils.py"
          language="python"
          code={`from dataclasses import dataclass

@dataclass
class TransformerConfig:
    # Model architecture
    vocab_size: int = 65
    d_model: int = 384
    d_ff: int = 1536
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.2
    seq_length: int = 256
    max_seq_length: int = 512
    
    # Training
    steps: int = 5000
    use_fixed_positional_embeddings: bool = False`}
        />

        <h3 id="model-parameters">Model Parameters</h3>
        <ApiTable parameters={modelParams} title="Model Architecture Parameters" />

        <Callout type="warning" title="Constraint">
          <code>d_model</code> must be divisible by <code>n_heads</code>. Each attention head 
          operates on a subspace of dimension <code>d_model / n_heads</code>.
        </Callout>

        <h3 id="training-parameters">Training Parameters</h3>
        <ApiTable parameters={trainingParams} title="Training Parameters" />

        <h2 id="example-configurations">Example Configurations</h2>
        
        <p><strong>Small Model (fast training, ~10M parameters):</strong></p>
        <CodeBlock
          language="python"
          code={`config = TransformerConfig(
    d_model=256,
    d_ff=1024,
    n_layers=4,
    n_heads=4,
    dropout=0.1,
    seq_length=128,
    steps=3000
)`}
        />

        <p><strong>Medium Model (balanced, ~25M parameters):</strong></p>
        <CodeBlock
          language="python"
          code={`config = TransformerConfig(
    d_model=384,
    d_ff=1536,
    n_layers=6,
    n_heads=6,
    dropout=0.2,
    seq_length=256,
    steps=5000
)`}
        />

        <p><strong>Large Model (slower, ~50M parameters):</strong></p>
        <CodeBlock
          language="python"
          code={`config = TransformerConfig(
    d_model=512,
    d_ff=2048,
    n_layers=8,
    n_heads=8,
    dropout=0.2,
    seq_length=512,
    steps=10000
)`}
        />

        <Callout type="tip" title="GPU Memory">
          Larger models require more GPU memory. If you encounter out-of-memory errors, 
          reduce <code>d_model</code>, <code>n_layers</code>, or <code>seq_length</code>.
        </Callout>
      </div>
    </DocLayout>
  );
}
