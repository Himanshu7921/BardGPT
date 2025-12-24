import { Link } from "react-router-dom";
import { Header } from "@/components/Header";
import { ArrowRight, Github, BookOpen, Cpu, Terminal, Zap, Layers, Box } from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Cpu,
    title: "Decoder-Only Architecture",
    description: "Pure decoder transformer with causal masking, designed for autoregressive language generation.",
  },
  {
    icon: Layers,
    title: "Modular Design",
    description: "Clean separation of attention, embeddings, and transformer blocks for easy customization.",
  },
  {
    icon: Terminal,
    title: "Training Pipeline",
    description: "Complete training loop with gradient clipping, warmup scheduling, and checkpoint saving.",
  },
  {
    icon: Zap,
    title: "Flexible Sampling",
    description: "Support for temperature, top-k, and top-p (nucleus) sampling strategies.",
  },
  {
    icon: BookOpen,
    title: "Educational Focus",
    description: "Well-documented codebase designed for learning transformer internals from scratch.",
  },
  {
    icon: Box,
    title: "Configurable",
    description: "Single configuration dataclass controls all model hyperparameters and training settings.",
  },
];

const quickLinks = [
  { label: "Quick Start", href: "/docs/quickstart", description: "Get up and running in minutes" },
  { label: "Architecture", href: "/docs/architecture", description: "Understand the model structure" },
  { label: "API Reference", href: "/docs/api/model", description: "Detailed API documentation" },
  { label: "Training Guide", href: "/docs/training", description: "Train your own model" },
];

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero Section */}
      <section className="relative overflow-hidden hero-gradient">
        <div className="container px-4 md:px-8 py-20 md:py-32">
          <div className="max-w-3xl mx-auto text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-full px-4 py-1.5 mb-6">
              <span className="text-xs font-medium text-primary">Open Source</span>
              <span className="text-muted-foreground">•</span>
              <span className="text-xs text-muted-foreground">MIT License</span>
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
              BardGPT
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-4">
              A Decoder-Only Transformer Implementation
            </p>
            
            <p className="text-muted-foreground max-w-xl mx-auto mb-8">
              Build and train GPT-style language models from scratch. Clean, modular PyTorch implementation 
              designed for learning and experimentation.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button asChild size="lg" className="gap-2">
                <Link to="/docs">
                  Get Started
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="gap-2">
                <a href="https://github.com/Himanshu7921/BardGPT" target="_blank" rel="noopener noreferrer">
                  <Github className="w-4 h-4" />
                  View on GitHub
                </a>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Preview */}
      <section className="border-t border-border">
        <div className="container px-4 md:px-8 py-16 md:py-24">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-12">
              <h2 className="text-2xl md:text-3xl font-semibold mb-4">Model Architecture</h2>
              <p className="text-muted-foreground max-w-2xl mx-auto">
                A clean implementation of the decoder-only transformer architecture, featuring multi-head 
                self-attention, layer normalization, and position-wise feedforward networks.
              </p>
            </div>
            
            <div className="bg-card border border-border rounded-xl p-6 md:p-8">
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Key Components</h3>
                  <ul className="space-y-3 text-sm">
                    <li className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <span className="text-primary text-xs font-medium">1</span>
                      </div>
                      <div>
                        <p className="font-medium">Token & Position Embeddings</p>
                        <p className="text-muted-foreground">Learnable embeddings with optional fixed sinusoidal positions</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <span className="text-primary text-xs font-medium">2</span>
                      </div>
                      <div>
                        <p className="font-medium">Multi-Head Self-Attention</p>
                        <p className="text-muted-foreground">Scaled dot-product attention with causal masking</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <span className="text-primary text-xs font-medium">3</span>
                      </div>
                      <div>
                        <p className="font-medium">Decoder Blocks</p>
                        <p className="text-muted-foreground">Pre-LN residual connections with GELU feedforward</p>
                      </div>
                    </li>
                    <li className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                        <span className="text-primary text-xs font-medium">4</span>
                      </div>
                      <div>
                        <p className="font-medium">Linear Output Head</p>
                        <p className="text-muted-foreground">Vocabulary projection with optional weight tying</p>
                      </div>
                    </li>
                  </ul>
                </div>
                
                <div className="bg-muted/50 rounded-lg p-6 font-mono text-sm">
                  <div className="text-muted-foreground mb-2"># Model forward pass</div>
                  <div className="space-y-1">
                    <p><span className="text-primary">x</span> = token_embed(tokens)</p>
                    <p><span className="text-primary">x</span> = x + pos_embed(positions)</p>
                    <p><span className="text-primary">x</span> = dropout(x)</p>
                    <p className="text-muted-foreground"># N decoder layers</p>
                    <p><span className="text-muted-foreground">for</span> block <span className="text-muted-foreground">in</span> decoder:</p>
                    <p className="pl-4"><span className="text-primary">x</span> = block(x, mask)</p>
                    <p><span className="text-primary">logits</span> = linear(norm(x))</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-border bg-muted/30">
        <div className="container px-4 md:px-8 py-16 md:py-24">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-semibold mb-4">Features</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Everything you need to understand and train transformer language models.
            </p>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {features.map((feature) => (
              <div key={feature.title} className="feature-card">
                <feature.icon className="w-8 h-8 text-primary mb-4" />
                <h3 className="font-medium mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="border-t border-border">
        <div className="container px-4 md:px-8 py-16 md:py-24">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-semibold mb-4">Explore the Documentation</h2>
            <p className="text-muted-foreground">Jump to the section that interests you most.</p>
          </div>
          
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-4xl mx-auto">
            {quickLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                className="group p-4 rounded-lg border border-border hover:border-primary/30 hover:bg-card transition-colors"
              >
                <h3 className="font-medium mb-1 group-hover:text-primary transition-colors">
                  {link.label}
                </h3>
                <p className="text-sm text-muted-foreground">{link.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border">
        <div className="container px-4 md:px-8 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">B</span>
              </div>
              <span className="text-sm text-muted-foreground">
                BardGPT Documentation
              </span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link to="/legal/license" className="hover:text-foreground transition-colors">License</Link>
              <Link to="/contribute" className="hover:text-foreground transition-colors">Contribute</Link>
              <a 
                href="https://github.com/Himanshu7921/BardGPT" 
                target="_blank" 
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
