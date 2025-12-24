import { DocLayout } from "@/components/DocLayout";
import { Breadcrumb } from "@/components/Breadcrumb";
import { CodeBlock } from "@/components/CodeBlock";
import { ApiTable } from "@/components/ApiTable";
import { Callout } from "@/components/Callout";

const toc = [
  { id: "overview", text: "Overview", level: 2 },
  { id: "training-loop", text: "Training Loop", level: 2 },
  { id: "data-loading", text: "Data Loading", level: 2 },
  { id: "learning-rate-schedule", text: "Learning Rate Schedule", level: 2 },
  { id: "gradient-clipping", text: "Gradient Clipping", level: 2 },
  { id: "checkpointing", text: "Checkpointing", level: 2 },
  { id: "monitoring", text: "Monitoring with TensorBoard", level: 2 },
  { id: "early-stopping", text: "Early Stopping", level: 2 },
];

export default function Training() {
  return (
    <DocLayout toc={toc}>
      <Breadcrumb items={[{ label: "Docs", href: "/docs" }, { label: "Training" }]} />
      
      <div className="doc-prose">
        <h1>Training</h1>
        
        <p className="text-lg text-muted-foreground">
          Comprehensive guide to training BardGPT models, including the training loop, 
          learning rate scheduling, and best practices.
        </p>

        <h2 id="overview">Overview</h2>
        <p>
          BardGPT's training pipeline is implemented in <code>transformer/train.py</code>. 
          The training process uses:
        </p>
        <ul>
          <li>AdamW optimizer with weight decay</li>
          <li>Cosine learning rate schedule with linear warmup</li>
          <li>Gradient clipping for training stability</li>
          <li>Automatic checkpointing at regular intervals</li>
          <li>TensorBoard logging for monitoring</li>
        </ul>

        <h2 id="training-loop">Training Loop</h2>
        <p>
          The main training function <code>train_model()</code> orchestrates the entire 
          training process:
        </p>

        <CodeBlock
          filename="transformer/train.py"
          language="python"
          code={`def train_model(config: TransformerConfig) -> nn.Module:
    """
    Train a decoder-only transformer model.
    
    Args:
        config: TransformerConfig with all hyperparameters
    
    Returns:
        Trained model
    """
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformerModel(config).to(device)
    
    # Data
    data_loader = DataLoader("data/tiny_shakespeare.txt", config)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=config.steps
    )
    
    # Training loop
    model.train()
    for step in range(config.steps):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging and checkpointing
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
        
        if step % 1000 == 0:
            save_checkpoint(model, config, step)
    
    return model`}
        />

        <h2 id="data-loading">Data Loading</h2>
        <p>
          The <code>DataLoader</code> class handles text preprocessing and batch generation:
        </p>

        <CodeBlock
          language="python"
          code={`class DataLoader:
    def __init__(self, filepath: str, config: TransformerConfig):
        with open(filepath, 'r') as f:
            text = f.read()
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        data = torch.tensor([self.stoi[c] for c in text])
        
        # Train/val split (90/10)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        self.seq_length = config.seq_length
        self.batch_size = 64
    
    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        
        # Random starting positions
        ix = torch.randint(len(data) - self.seq_length, (self.batch_size,))
        
        # Input and target sequences
        x = torch.stack([data[i:i+self.seq_length] for i in ix])
        y = torch.stack([data[i+1:i+self.seq_length+1] for i in ix])
        
        return x, y`}
        />

        <Callout type="info" title="Sliding Window">
          The data loader uses sliding window batching where each training example 
          is a random contiguous chunk of the text. The target sequence is shifted 
          by one position, creating the autoregressive training signal.
        </Callout>

        <h2 id="learning-rate-schedule">Learning Rate Schedule</h2>
        <p>
          BardGPT uses a cosine learning rate schedule with linear warmup, which has 
          been shown to improve training stability and final performance:
        </p>

        <CodeBlock
          language="python"
          code={`def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with linear warmup and cosine decay.
    
    Learning rate profile:
    - Steps 0 to num_warmup_steps: linear increase from 0 to lr
    - Steps num_warmup_steps to num_training_steps: cosine decay to 0
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps)
        progress /= float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)`}
        />

        <h2 id="gradient-clipping">Gradient Clipping</h2>
        <p>
          Gradient clipping prevents exploding gradients during training:
        </p>

        <CodeBlock
          language="python"
          code={`# Clip gradients to max norm of 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`}
        />

        <p>
          This is especially important during the early stages of training when 
          gradients can be unstable.
        </p>

        <h2 id="checkpointing">Checkpointing</h2>
        <p>
          Checkpoints are saved periodically and include both model weights and configuration:
        </p>

        <CodeBlock
          language="python"
          code={`def save_checkpoint(model, config, step, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step
    }
    torch.save(checkpoint, f"{path}/model_step_{step}.pt")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    config = checkpoint["config"]
    model = DecoderOnlyTransformerModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config`}
        />

        <h2 id="monitoring">Monitoring with TensorBoard</h2>
        <p>
          Training progress can be monitored using TensorBoard:
        </p>

        <CodeBlock
          language="bash"
          code={`# Start TensorBoard
tensorboard --logdir=runs

# View at http://localhost:6006`}
        />

        <p>Logged metrics include:</p>
        <ul>
          <li>Training loss per step</li>
          <li>Validation loss (evaluated periodically)</li>
          <li>Learning rate schedule</li>
          <li>Gradient norms</li>
        </ul>

        <h2 id="early-stopping">Early Stopping</h2>
        <p>
          While not enabled by default, you can implement early stopping based on 
          validation loss:
        </p>

        <CodeBlock
          language="python"
          code={`best_val_loss = float('inf')
patience = 5
patience_counter = 0

for step in range(config.steps):
    # ... training code ...
    
    if step % eval_interval == 0:
        val_loss = evaluate(model, data_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, config, step, "checkpoints/best")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at step {step}")
            break`}
        />
      </div>
    </DocLayout>
  );
}
