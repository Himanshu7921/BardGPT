import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { Callout } from "@/components/Callout";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "what-is-bardgpt", text: "What is BardGPT?", level: 2 },
  { id: "core-components", text: "Core Components", level: 2 },
  { id: "project-structure", text: "Project Structure", level: 2 },
  { id: "prerequisites", text: "Prerequisites", level: 2 },
  { id: "next-steps", text: "Next Steps", level: 2 },
];

export default function Introduction() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Introduction" }]} />
      
      <div className="doc-prose">
        <h1>Introduction to BardGPT</h1>
        
        <p className="text-lg text-muted-foreground">
          A clean, modular implementation of a decoder-only transformer for learning 
          and experimentation with language models.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          BardGPT is an educational implementation of the GPT (Generative Pre-trained Transformer) 
          architecture built from scratch using PyTorch. The project focuses on clarity and 
          modularity, making it an excellent resource for understanding how modern language 
          models work under the hood.
        </p>

        <h2 id="what-is-bardgpt">What is BardGPT?</h2>
        <p>
          BardGPT implements the decoder-only transformer architecture, the same foundational 
          design used in models like GPT-2, GPT-3, and other autoregressive language models. 
          The model is trained on the Tiny Shakespeare dataset, learning to generate 
          Shakespeare-style text character by character.
        </p>

        <Callout type="info" title="Training Philosophy">
          BardGPT is designed for educational purposes. Rather than optimizing for performance, 
          the codebase prioritizes readability and understanding of core transformer concepts.
        </Callout>

        <h2 id="core-components">Core Components</h2>
        <p>The implementation consists of several key modules:</p>

        <ul>
          <li>
            <strong>transformer.model</strong> — The main <code>DecoderOnlyTransformerModel</code> class 
            that orchestrates all components
          </li>
          <li>
            <strong>transformer.attention</strong> — Multi-head self-attention mechanism with 
            causal masking
          </li>
          <li>
            <strong>transformer.block</strong> — Transformer blocks including feedforward networks, 
            layer normalization, and residual connections
          </li>
          <li>
            <strong>transformer.embedding</strong> — Token and positional embedding layers
          </li>
          <li>
            <strong>transformer.utils</strong> — Configuration, data loading, and utility functions
          </li>
          <li>
            <strong>transformer.train</strong> — Training loop with learning rate scheduling 
            and checkpointing
          </li>
          <li>
            <strong>transformer.sample</strong> — Text generation with temperature and 
            top-k/top-p sampling
          </li>
        </ul>

        <h2 id="project-structure">Project Structure</h2>
        
        <CodeBlock
          filename="Repository Layout"
          language="text"
          code={`BardGPT/
├── transformer/
│   ├── attention.py      # Multi-head attention
│   ├── block.py          # Decoder blocks & components
│   ├── embedding.py      # Embeddings
│   ├── model.py          # Main model class
│   ├── utils.py          # Config & utilities
│   ├── train.py          # Training pipeline
│   └── sample.py         # Text generation
├── data/
│   └── tiny_shakespeare.txt
├── checkpoints/          # Saved model weights
├── requirements.txt
└── README.md`}
        />

        <h2 id="prerequisites">Prerequisites</h2>
        <p>To work with BardGPT, you'll need:</p>

        <ul>
          <li>Python 3.8 or higher</li>
          <li>PyTorch 1.9+ with CUDA support (optional but recommended)</li>
          <li>Basic understanding of neural networks and transformers</li>
        </ul>

        <CodeBlock
          filename="Installation"
          language="bash"
          code={`git clone https://github.com/Himanshu7921/BardGPT.git
cd BardGPT
pip install -r requirements.txt`}
        />

        <h2 id="next-steps">Next Steps</h2>
        <p>Now that you understand what BardGPT is, explore these topics:</p>

        <div className="grid gap-3 mt-4 not-prose">
          <Link 
            to="/docs/quickstart" 
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-primary/30 hover:bg-card transition-colors group"
          >
            <div>
              <p className="font-medium group-hover:text-primary transition-colors">Quick Start</p>
              <p className="text-sm text-muted-foreground">Get up and running in minutes</p>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
          </Link>
          <Link 
            to="/docs/architecture" 
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-primary/30 hover:bg-card transition-colors group"
          >
            <div>
              <p className="font-medium group-hover:text-primary transition-colors">Architecture</p>
              <p className="text-sm text-muted-foreground">Deep dive into the model structure</p>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
          </Link>
          <Link 
            to="/docs/api/model" 
            className="flex items-center justify-between p-4 rounded-lg border border-border hover:border-primary/30 hover:bg-card transition-colors group"
          >
            <div>
              <p className="font-medium group-hover:text-primary transition-colors">API Reference</p>
              <p className="text-sm text-muted-foreground">Detailed documentation for all modules</p>
            </div>
            <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
          </Link>
        </div>
      </div>
    </DocLayout>
  );
}
