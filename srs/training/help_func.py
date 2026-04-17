# help_func.py
from tqdm import tqdm

import torch
import os

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

# Check if LoRA is enabled and scaling is reasonable
# for name, module in model.named_modules():
#     if hasattr(module, 'enabled'):
#         print(f"{name}: enabled={module.enabled}, r={module.r}, alpha={module.alpha}, scaling={module.scaling}")
#
# for name, p in model.named_parameters():
#     if p.requires_grad:
#         print(name, p.grad.abs().mean() if p.grad is not None else None)
#
# for name, module in model.named_modules():
#     if hasattr(module, "A"):
#         print(name, module.A.abs().mean().item(), module.B.abs().mean().item())
#
# print(type(model.trf_blocks[0].att.W_query))
#
# batch = next(iter(dataloader_func))
# input_ids = batch["input_ids"].to(device)
# labels = batch["labels"].to(device)
# logits = model(input_ids)
# loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
# print("Initial batch loss:", loss.item())
# for name, param in model.named_parameters():
#     if "A" in name or "B" in name:
#         param.requires_grad = True
#         print(f"{name} -> {param.requires_grad}")
#     else:
#         param.requires_grad = False
#
# from model.LoRA import LoRALinear
# for module in model.modules():
#     if isinstance(module, LoRALinear):
#         module.A.requires_grad = True
#         module.B.requires_grad = True
#
# for name, p in model.named_parameters():
#     if p.requires_grad:
#         print(name)
#
# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Trainable params:", trainable)
# for p in model.parameters():
#     if p.requires_grad:
#         print(p)

# =========================
# Checkpoint
# =========================
def load_checkpoint(model, optimizer, path, device):
    start_epoch, start_step = 0, 0

    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint.get("epoch", 0)
        start_step = checkpoint.get("step", 0)

        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    # move optimizer tensors to device
    for state in optimizer.state.values():
        for k, v in state.items():
            state[k] = move_to_device(v, device)

    return start_epoch, start_step


def save_checkpoint(model, optimizer, epoch, step, path, device):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "device": device
    }, path)



def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj

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

    return start_epoch, start_step, device

def leyers_with_grad(model):
    for name, p in model.named_parameters():
        print(f"{name} -> {p.requires_grad}")

    print(sum(p.numel() for name, p in model.named_parameters() if p.requires_grad))


def inspect_one_example(dataloader, tokenizer, IGNORE_INDEX):
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

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