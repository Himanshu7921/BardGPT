import torch
import torch.nn as nn
from utils import TransformerConfig, causal_mask, DataLoader, generate, print_model_summary, save_checkpoint, load_checkpoint, checkpoint_name
from model import DecoderOnlyTransformerModel
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.utils.tensorboard import SummaryWriter



def load_data(address="tiny_shakespeare.txt"):
    config = TransformerConfig()
    loader = DataLoader(dataset_address=address, seq_length=config.seq_length)
    return loader, config

def train_model(model: DecoderOnlyTransformerModel,
            dataloader: DataLoader,
            batch_size: int,
            steps: int,
            scheduler,
            optimizer: torch.optim,
            loss_fn: torch.nn.CrossEntropyLoss,
            print_interval: int,
            config: TransformerConfig,
            device: torch.device):
    model.train()
    best_val_loss = float("inf")
    bad_steps = 0
    patience = 5
    writer = SummaryWriter(log_dir="runs/transformer_training")
    
    # Train the Model and Return it
    for step in range(steps):
        x, y = dataloader.get_batch(
            split = "train",
            batch_size =  batch_size,
            device = device
        )
        B, T = x.shape
        mask = causal_mask(seq_length = T,
                    device = device)

        logits = model(x, mask) # (B, T, vocab_size)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) # logits.view(-1, logits.shape[-1]) --> (B*T, vocab_size)
        # Reason: torch.nn.CrossEntropyLoss expects,
        #         -> logits: (N, C)
        #         -> Target: (N)
        #           - N: Number of classification Examples
        #           - C: Number of classes
        # Model outputs, (B, T, C) --> needs to be flatten as (B*T, C) and y from (B, T) to (B*T) this is what y.view(-1) does

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()
        writer.add_scalar("Loss/train", loss.item(), step)

        if step % print_interval == 0 and step != 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = dataloader.get_batch(split = "val", batch_size = batch_size, device = device)

                logits = model(x_val, causal_mask(x_val.shape[1], device))
                val_loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_val.reshape(-1))

                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("Loss/val", val_loss.item(), step)
                print(f"Step {step}/{steps} | Training Loss {loss.item():.4f} | Validation Loss: {val_loss:.4f}")

                # Early stop check, and save the best Generalized weights so far
                if val_loss < best_val_loss:
                    path = f"checkpoints/{checkpoint_name(prefix='best_val', step=step)}"
                    best_val_loss = val_loss
                    bad_steps = 0
                    # Save the best Generalized Model: path = 'checkpoints/best_val_{step}.pt'
                    save_checkpoint(
                        model=model,
                        optimizer=None,
                        scheduler=None,
                        config=config,
                        step=step,
                        path = path
                    )
                    print(f"Saved checkpoint at Setp: {step} | path = {path}")
                else:
                    bad_steps += 1
                    print(f"[Early Stop] Validation loss increased {bad_steps} times")

                if bad_steps >= patience:
                    print(f"\nEarly stopping triggered at step {step}.")
                    path = f"checkpoints/{checkpoint_name(prefix='early_stop', step=step)}"
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        step=step,
                        path = path
                    )
                    print(f"Saved checkpoint at Setp: {step} | path = {path}")
                    writer.close()
                    return model
                # --------------------------------
            model.train()
        
        if step % 5000 == 0 and step != 0:
            path = f"checkpoints/{checkpoint_name(step=step)}"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                step=step,
                path = path
            )
            print(f"Saved checkpoint at Setp: {step} | path = {path}")

    writer.close()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, config = load_data()

    model = DecoderOnlyTransformerModel(
        max_seq_length=config.max_seq_length,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.n_heads,
        use_fixed_positional_embeddings=True,
        dropout=config.dropout,
        N=config.n_layers,
    ).to(device)



    print_model_summary(config = config, model = model)

    steps = config.steps # Load the Training Steps directly from Config File
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay = 0.1)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup + CosineLR Decay for Transformers
    warmup_steps = 2000
    min_lr_scale = 3e-5 / 3e-4  # = 0.1

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps

        cosine_progress = (step - warmup_steps) / (steps - warmup_steps)
        return min_lr_scale + 0.5 * (1 - min_lr_scale) * (1 + math.cos(math.pi * cosine_progress))
    
    # Tie with Scheduler
    scheduler = LambdaLR(optimizer, lr_lambda)
    print("> Training on Device: ", device)
    model = train_model(
        model = model,
        dataloader = dataloader,
        scheduler = scheduler,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = device,
        steps = steps,
        batch_size = 64,
        config = config,
        print_interval = 1000,
    )

    # Save the Trained Model
    path = f"checkpoints/{checkpoint_name(prefix='final', step = steps)}"
    save_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=steps,
        path = path
    )
    print(f"Final Saved checkpoint After Training at path = {path}")
    # ----------------------------------- Generation from the Model ------------------------------------------------------
    start_text = "ROMEO:"
    start_ids = torch.tensor(
        [[dataloader.stoi[c] for c in start_text]],
        dtype=torch.long,
        device=device
    )

    generated_ids_gpt_style = generate(
        model=model,
        idx=start_ids,
        max_new_tokens=300,
        device=device,
        temperature=0.8,
        top_k=None,
        top_p=0.9,
    )

    # decode back to text
    gpt_style_generated_text = "".join([dataloader.itos[i.item()] for i in generated_ids_gpt_style[0]])
    print()
    print("---------------------------------------------------------- GPT Style Sampling: -----------------------------------------------------------------")
    print()
    print(gpt_style_generated_text)
    print()

    clean_generated_ids = generate(
        model=model,
        idx=start_ids,
        max_new_tokens=300,
        device=device,
        temperature=0.7,
        top_k = 40,
        top_p = None,
    )

    clean_generated_text = "".join([dataloader.itos[i.item()] for i in clean_generated_ids[0]])
    print("---------------------------------------------------------- Clean Generated Text ----------------------------------------------------------------")
    print()
    print(clean_generated_text)
    print()

    chaotic_generated_ids = generate(
        model=model,
        idx=start_ids,
        max_new_tokens=300,
        device=device,
        temperature=1.1,
        top_k=10,
        top_p=0.95,
    )

    generated_text = "".join([dataloader.itos[i.item()] for i in chaotic_generated_ids[0]])
    print("---------------------------------------------------------- Chaotic Generation ------------------------------------------------------------------")
    print()
    print(generated_text)
    print()