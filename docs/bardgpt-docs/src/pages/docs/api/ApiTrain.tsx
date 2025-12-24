import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "train-model", text: "train_model()", level: 2 },
  { id: "load-data", text: "load_data()", level: 2 },
  { id: "training-loop-details", text: "Training Loop Details", level: 2 },
  { id: "hyperparameters", text: "Hyperparameters", level: 2 },
];

const trainModelParams = [
  { name: "config", type: "TransformerConfig", description: "Configuration object with all hyperparameters." },
];

export default function ApiTrain() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[
        { label: "Docs", href: "/docs" },
        { label: "API Reference", href: "/docs/api/model" },
        { label: "transformer.train" }
      ]} />
      
      <div className="doc-prose">
        <h1>transformer.train</h1>
        
        <p className="text-lg text-muted-foreground">
          Training pipeline with gradient clipping, learning rate scheduling, 
          and checkpointing.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          This module implements the complete training pipeline for BardGPT. It handles 
          model initialization, optimization, logging, and checkpoint management.
        </p>

        <h2 id="train-model">train_model()</h2>

        <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
          <span className="text-primary">def</span> train_model(config: TransformerConfig) -&gt; nn.Module
        </div>

        <p>
          Main entry point for training. Creates a model, trains it on the Tiny Shakespeare 
          dataset, and returns the trained model.
        </p>

        <ApiTable parameters={trainModelParams} title="Parameters" />

        <p><strong>Returns:</strong> <code>nn.Module</code> — Trained transformer model</p>

        <CodeBlock
          filename="transformer/train.py"
          language="python"
          code={`def train_model(config: TransformerConfig) -> nn.Module:
    """
    Train a decoder-only transformer model.
    
    Features:
    - AdamW optimizer with weight decay
    - Cosine learning rate schedule with linear warmup
    - Gradient clipping for stability
    - Periodic checkpointing
    - TensorBoard logging
    - Training and validation loss tracking
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Model initialization
    model = DecoderOnlyTransformerModel(config).to(device)
    print_model_summary(model)
    
    # Data loading
    data_loader = DataLoader("data/tiny_shakespeare.txt", config)
    print(f"Vocabulary size: {len(data_loader.stoi)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate scheduler
    def get_lr(step):
        warmup_steps = 100
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (config.steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    
    # TensorBoard logging
    writer = SummaryWriter()
    
    # Training loop
    model.train()
    training_losses = []
    validation_losses = []
    
    for step in range(config.steps):
        # Get batch
        x, y = data_loader.get_batch("train")
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        training_losses.append(loss.item())
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
        
        if step % 100 == 0:
            print(f"Step {step}/{config.steps} | Loss: {loss.item():.4f}")
        
        # Validation
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = data_loader.get_batch("val")
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, config.vocab_size),
                    val_y.view(-1)
                )
            validation_losses.append(val_loss.item())
            writer.add_scalar("Loss/val", val_loss.item(), step)
            print(f"  Validation loss: {val_loss.item():.4f}")
            model.train()
        
        # Checkpointing
        if step > 0 and step % 1000 == 0:
            save_checkpoint(model, config, step)
    
    # Save final checkpoint
    save_checkpoint(model, config, config.steps)
    
    # Save loss history
    with open("training_loss.json", "w") as f:
        json.dump(training_losses, f)
    with open("validation_loss.json", "w") as f:
        json.dump(validation_losses, f)
    
    writer.close()
    return model`}
        />

        <h2 id="load-data">load_data()</h2>

        <p>
          Data loading is handled by the <code>DataLoader</code> class from 
          <code>transformer.utils</code>. See the 
          <a href="/docs/api/utils#dataloader">DataLoader documentation</a> for details.
        </p>

        <CodeBlock
          language="python"
          code={`# Data is loaded automatically in train_model():
data_loader = DataLoader("data/tiny_shakespeare.txt", config)

# Access vocabulary mappings:
data_loader.stoi  # {"a": 0, "b": 1, ...}
data_loader.itos  # {0: "a", 1: "b", ...}

# Get training batch:
x, y = data_loader.get_batch("train")`}
        />

        <h2 id="training-loop-details">Training Loop Details</h2>

        <h3>Gradient Clipping</h3>
        <p>
          Gradients are clipped to a maximum norm of 1.0 to prevent exploding gradients:
        </p>
        <CodeBlock
          language="python"
          code={`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`}
        />

        <h3>Learning Rate Warmup</h3>
        <p>
          The learning rate linearly increases from 0 to the base LR over the first 
          100 steps, then follows a cosine decay:
        </p>
        <CodeBlock
          language="python"
          code={`# Warmup: linear increase for first 100 steps
# Decay: cosine decay from step 100 to config.steps

warmup_steps = 100
if step < warmup_steps:
    lr_mult = step / warmup_steps
else:
    progress = (step - warmup_steps) / (config.steps - warmup_steps)
    lr_mult = 0.5 * (1 + math.cos(math.pi * progress))`}
        />

        <h3>Checkpointing</h3>
        <p>Checkpoints are saved every 1000 steps and at the end of training:</p>
        <CodeBlock
          language="python"
          code={`# Checkpoints saved to:
# checkpoints/model_step_1000.pt
# checkpoints/model_step_2000.pt
# ...
# checkpoints/model_step_5000.pt (final)`}
        />

        <h2 id="hyperparameters">Hyperparameters</h2>

        <p>Fixed training hyperparameters used in <code>train_model()</code>:</p>

        <div className="overflow-x-auto">
          <table className="api-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Learning Rate</td>
                <td><code>3e-4</code></td>
                <td>Base learning rate</td>
              </tr>
              <tr>
                <td>Betas</td>
                <td><code>(0.9, 0.95)</code></td>
                <td>Adam momentum parameters</td>
              </tr>
              <tr>
                <td>Weight Decay</td>
                <td><code>0.1</code></td>
                <td>L2 regularization strength</td>
              </tr>
              <tr>
                <td>Warmup Steps</td>
                <td><code>100</code></td>
                <td>Linear warmup duration</td>
              </tr>
              <tr>
                <td>Gradient Clip</td>
                <td><code>1.0</code></td>
                <td>Maximum gradient norm</td>
              </tr>
              <tr>
                <td>Batch Size</td>
                <td><code>64</code></td>
                <td>Sequences per batch</td>
              </tr>
            </tbody>
          </table>
        </div>

        <Callout type="tip" title="Customization">
          To modify these hyperparameters, edit the <code>train_model()</code> function 
          directly or create a custom training script that imports the model components.
        </Callout>
      </div>
    </DocLayout>
  );
}
