# help_func.py
from tqdm import tqdm

import torch
import torch.nn.functional as F

# import importlib
# importlib.reload(ds_prep)

# import inspect
# print(inspect.getsource(ds_prep.oasst1_df))
# from model.LoRA import LoRALinear
#
# with torch.no_grad():
#     for name, module in model.named_modules():
#         if isinstance(module, LoRALinear):
#             print(name, module.A.abs().mean().item(), module.B.abs().mean().item())

def save_checkpoint(
        model,
        optimizer,
        path: str ,
        epoch: int,
        step: int,
        device: str = None
) -> None:
    """Save model and optimizer checkpoint."""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "device": device or str(next(model.parameters()).device)
    }
    torch.save(checkpoint, path)


def load_checkpoint(
        model,
        optimizer,
        path: str,
        map_location: str = None,

):
    """Load checkpoint and return state."""
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"])

    start_epoch = checkpoint.get("epoch", 0)
    start_step = checkpoint.get("step", 0)
    device = checkpoint.get("device", "cpu")

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return model, optimizer, start_epoch, start_step, device

def leyers_with_grad(model):
    for name, p in model.named_parameters():
        print(f"{name} -> {p.requires_grad}")

def temp_predict(model, prompt, tokenizer, device, temperature=1.0, max_new_tokens=50):
    model.eval()

    prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
    decoded = tokenizer.decode(input_ids[0])
    return decoded.split("<|assistant|>")[-1]


def greedy_predict(model, prompt, tokenizer,device):
    model.eval()
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(50):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    input_ids = input_ids[:, (input_ids[0] == assistant_token_id).nonzero(as_tuple=True)[0][0] + 1:]
    return tokenizer.decode(input_ids[0])

def inspect_one_example(dataloader, tokenizer, IGNORE_INDEX):
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"][2]
    labels = batch["labels"][2]

    decoded_input = tokenizer.decode(input_ids.tolist())

    print("=== INPUT TEXT ===")
    print(decoded_input)

    print("\n=== TOKENS ===")
    print(input_ids.tolist())

    print("\n=== LABELS ===")
    print(labels.tolist())

    print("\n=== TOKEN | LABEL ===")
    for token_id, label_id in zip(input_ids.tolist(), labels.tolist()):
        token = tokenizer.decode([token_id])
        label = tokenizer.decode([label_id]) if label_id != IGNORE_INDEX else "IGN"

        print(f"{repr(token):>10} -> {label}")

def inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx, INDEX):
    """Детальная проверка сэмпла"""

    input_ids = batch['input_ids'][sample_idx]
    labels = batch['labels'][sample_idx]
    attention_mask = batch['attention_mask'][sample_idx]

    valid_len = attention_mask.sum().item()

    print(f"\n{'=' * 70}")
    print(f"Batch {batch_idx}, Sample {sample_idx}")
    print(f"{'=' * 70}")

    print(f"{'Pos':<4} {'Token':<25} {'Trainable':<12} {'Token ID':<8}")
    print("-" * 70)

    for i in range(valid_len):
        token_id = input_ids[i].item()
        token = tokenizer.decode([token_id]).replace('\n', '\\n')
        is_trainable = (labels[i] != INDEX).item()

        trainable_mark = "✅ LEARN" if is_trainable else "❌ IGNORE"

        if token in ["<|user|>", "<|assistant|>", "<|endoftext|>", "\\n"]:
            trainable_mark = f"🔷 {trainable_mark}"

        print(f"{i:<4} {token:<25} {trainable_mark:<12} {token_id:<8}")

        if i > 0 and is_trainable and (labels[i - 1] == INDEX).item():
            print(f"     ↑ Beginning of training ↑")

    total_trainable = (labels[:valid_len] != INDEX).sum().item()
    print(f"\n📊 Statistics:")
    print(f"  Total tokens: {valid_len}")
    print(f"  Trainable tokens: {total_trainable}")
    print(f"  Percentage: {total_trainable / valid_len * 100:.1f}%")

def check_one_batch(dataloader, tokenizer, INDEX):
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n{'#' * 70}")
        print(f"BATCH {batch_idx}")
        print(f"{'#' * 70}")

        for sample_idx in range(batch['input_ids'].shape[0]):
            inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx, INDEX)

        if batch_idx >= 0:
            break