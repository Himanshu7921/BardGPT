import torch
import os
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    """
    Configuration settings for the Transformer model architecture.

    Attributes:
        vocab_size (int): Size of the character dictionary.
        d_model (int): The number of expected features in the encoder/decoder inputs.
        d_ff (int): The dimension of the feedforward network model.
        n_layers (int): Number of sub-decoder-layers in the transformer.
        n_heads (int): Numbers of heads in the multiheadattention models.
        dropout (float): The dropout value.
        seq_length (int): Length of sequences used during training.
        max_seq_length (int): The absolute maximum position for embeddings.
    """
    vocab_size: int = 65 # we can get this value by loading the data and get len(chars) | self.chars = sorted(list(set(text)))
    d_model: int = 384
    d_ff: int = 4 * d_model # Original Paper used d_ff as 4 * d_model
    n_layers: int = 6
    n_heads: int = 6
    dropout: float = 0.2
    seq_length:int = 128
    max_seq_length: int = 256
    steps = 200000
    use_fixed_positional_embeddings = True

def causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
    """
    Creates a causal (lower triangular) attention mask
    Shape: (1, 1, seq_length, seq_length) = (1, 1, T, T)
    """
    return torch.tril(
        torch.ones(seq_length, seq_length, device=device)
    ).unsqueeze(0).unsqueeze(0)

def load_checkpoint(
    path: str,
    model_class,
    device: torch.device,
    optimizer = None,
    scheduler = None
):
    """
    Loads a model and its training state from a saved checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model_class (Type): The class to instantiate the model from.
        device (torch.device): The device (CPU/GPU) to load the model onto.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

    Returns:
        tuple: (model, optimizer, scheduler, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]

    model = model_class(
        max_seq_length=config.max_seq_length,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.n_heads,
        use_fixed_positional_embeddings = config.use_fixed_positional_embeddings,
        dropout=config.dropout,
        N=config.n_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[✓] Model loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("[✓] Optimizer state restored")
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("[✓] Scheduler state restored")

    model.eval()
    return model, optimizer, scheduler, checkpoint


import os
import torch

def save_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    config=None,
    step: int = 0,
    path: str = "checkpoint.pt"
):
    """
    Saves the model state and training metdata to a file.

    Args:
        model (torch.nn.Module): The model instance to be saved.
        optimizer (torch.optim.Optimizer, optional): The current optimizer state.
        schedular (torch.optim.lr_schedular, optional): The learning rate schedular.
        config (TransformerConfig, optional): The configuration used to build the model.
        step (int): The current training step/epoch number.
        path (str): The destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"[✓] Model checkpoint saved to {path}")


def print_model_summary(config: TransformerConfig, model):
    """
    Prints a formatted report of the model architecture and parameters.

    Args:
        config (TransformerConfig): The configuration object for the model.
        model (torch.nn.Module): The instantiated Transformer model.
    """
    print("=" * 144)
    print("                                                 Decoder-Only Transformer Model Summary")
    print("=" * 144)

    print("\n[Model Type]")
    print("  • Decoder-only Transformer (GPT-style)")

    print("\n[Vocabulary]")
    print(f"  • Vocabulary size        : {config.vocab_size}")

    print("\n[Sequence Lengths]")
    print(f"  • Training seq_length    : {config.seq_length}")
    print(f"  • Max context length     : {config.max_seq_length}")

    print("\n[Embedding]")
    print(f"  • Embedding dimension    : {config.d_model}")
    print(f"  • Positional Embedding Type     : {'Fixed-Sinusoidal' if config.use_fixed_positional_embeddings else 'Learned'} Positional Embedding")

    print("\n[Transformer Architecture]")
    print(f"  • Number of layers       : {config.n_layers}")
    print(f"  • Number of heads        : {config.n_heads}")
    print(f"  • Head dimension         : {config.d_model // config.n_heads}")
    print(f"  • FeedForward dim (d_ff) : {config.d_ff}")

    print("\n[Regularization]")
    print(f"  • Dropout                : {config.dropout}")

    print("\n[Training Steps]")
    print(f"  • Steps:         : {config.steps}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[Parameters]")
    print(f"  • Total parameters       : {total_params:,}")
    print(f"  • Trainable parameters   : {trainable_params:,}")

    print("\n[Attention Details]")
    print("  • Attention type         : Masked self-attention (causal)")
    print("  • Positional encoding    : Fixed or Learned (configurable)")

    print("=" * 144)

def checkpoint_name(prefix="step", step=0):
    """
    Generates a standardized filename for saving model checkpoints.

    Args:
        prefix (str): The starting string for the filename.
        step (int): The current training step to be included in the name.

    Returns:
        str: A formatted string such as 'step_0000100.pt'.  
    """
    return f"{prefix}_{step:07d}.pt"


class DataLoader:
    """
    A simple character-level data loader for the text files.

    This class handles reading the dataset, creating a character-to-integer mapping, 
    and generating batches for training or validation.
    """
    def __init__(self, dataset_address: str = "tiny_shakespeare.txt", *, seq_length: int):
        """
        Initializes the DataLoader with a specific dataset.

        Args:
            dataset_address (str): The local path to the text file.
            seq_length (int): The window size for each training sequences.
        """
        text = open(dataset_address, 'r', encoding='utf-8').read()

        self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_length = seq_length  # sequence length

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split='train', batch_size = 64, *, device: torch.device):
        """
        Retrieves a random batch of inputs (X) and targets (Y).

        Args:
            split (str): Either 'train' or 'val' to select the data subset.
            batch_size (int): Number of independent sequences per batch.
            device (torch.device): Device to which the returned tensors are moved.

        Returns:
            tuple: (X, Y) where X is the context and Y is the target (next character).
        """
        assert split in ("train", "val")
        source = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(source) - self.seq_length - 1, (batch_size,))
        X = torch.stack([source[i:i + self.seq_length] for i in ix])
        Y = torch.stack([source[i+1:i + self.seq_length + 1] for i in ix])
        return X.to(device), Y.to(device)


def top_k_logits(logits: torch.Tensor, k: int):
    """
    Filters a distribution of logits using top-k sampling.

    Args:
        logits (torch.Tensor): The output scores from the model's last layer.
        k (int): The number of highest probability vocabulary tokens to keep.

    Returns:
        torch.Tensor: Logits with values outside the top-k set to negative infinity.
    """
    k = min(k, logits.size(-1))
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)

def top_p_logits(logits: torch.Tensor, p: float):
    """
    Performs Nucleus (Top-P) filtering on the logits.

    This limits the sampling to the smallest set of tokens whose cumulative probability exceeds the threshold p.

    Args:
        logits (torch.Tensor): The output scores from the model.
        p (float): The cumulative probability threshold (0 to 1).

    Returns:
        torch.Tensor: The filtered logits with non-selected values set to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)

    cum_probs = torch.cumsum(probs, dim=-1)

    keep = cum_probs <= p
    keep[..., 0] = True

    masked_logits = sorted_logits.masked_fill(~keep, -1e9)

    new_logits = torch.full_like(logits, -1e9)
    new_logits.scatter_(1, sorted_indices, masked_logits)

    return new_logits


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    device,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    """
    Generates a sequence of tokens from a starting context.

    Args:
        model (torch.nn.Module): The trained transformer model.
        idx (torch.Tensor): Starting sequence of token indices (shape [B,T]).
        max_new_tokens (int): How many tokens to generate.
        device (torch.device): Device to perform computation on.
        temperature (float): Scaling factor for the logits (higher = more random).
        top_k (int, optional): If set, only sample from the top k tokens.
        top_p (float, optional): If set, perform Nucleus sampling.

    Returns:
        torch.Tensor: The sequence including the newly generated tokens. 
    """
    model.eval()
    # max_T = model.embeddings.pos_embedding.max_seq_length // 2 # 256//2 = 128
    max_T = TransformerConfig.seq_length # 256//2 = 128

    for _ in range(max_new_tokens):

        # sliding window
        idx_cond = idx[:, -max_T:]

        T = idx_cond.size(1)
        mask = causal_mask(T, device)

        logits = model(idx_cond, mask)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        if top_p is not None:
            logits = top_p_logits(logits, top_p)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


