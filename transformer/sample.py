import torch
import torch.nn as nn
from utils import TransformerConfig, causal_mask, DataLoader, generate, print_model_summary, save_checkpoint, load_checkpoint, checkpoint_name
from model import DecoderOnlyTransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data(address="tiny_shakespeare.txt"):
    config = TransformerConfig()
    loader = DataLoader(dataset_address=address, seq_length=config.seq_length)
    return loader, config

dataloader, config = load_data()
steps = config.steps # # Load the Training Steps directly from Config File
path = f"checkpoints/{checkpoint_name(prefix='final', step = steps)}"
path = "checkpoints/best_val_0072000.pt"
loaded_model, _, _, _ = load_checkpoint(
    path = path,
    model_class=DecoderOnlyTransformerModel,
    device=device,
)

loaded_model.eval()
for i in range(5):
    print(f"sample: {i+1}/{5}")
    start_text = "ROMEO:"
    start_ids = torch.tensor(
        [[dataloader.stoi[c] for c in start_text]],
        dtype=torch.long,
        device=device
    )

    generated_ids_gpt_style_loaded_model = generate(
            model = loaded_model,
            idx = start_ids,
            max_new_tokens = 1000,
            device = device,
            temperature = 0.7,
            top_k = 50,
            top_p = None,
    )

    # decode back to text
    print("device", device)
    gpt_style_generated_text_loaded_model = "".join([dataloader.itos[i.item()] for i in generated_ids_gpt_style_loaded_model[0]])
    print("GPT Style Sampling from Loaded Model: ")
    print(gpt_style_generated_text_loaded_model)

    print("-"*40)