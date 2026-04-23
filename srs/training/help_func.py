# help_func.py
from tqdm import tqdm

import torch
import os
import numpy as np
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup
from srs.LLM_classification.config import Config
from model.GPT_full_model import GPT2Manager
import dataset.Dataset_dataloader as DL


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_schedular(optimizer, dataloader, epoch):
    num_training_steps = len(dataloader) * epoch
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def get_tokenizer():
    if os.path.exists(Config.TOKENIZER_PATH):
        tokenizer = GPT2Tokenizer.from_pretrained(Config.TOKENIZER_PATH)

    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(Config.TOKENIZER_PATH)

    tokenizer.eos_token = Config.EOS_TOKEN
    tokenizer.add_special_tokens({
        "additional_special_tokens": [Config.USER_TOKEN, Config.ASSIST_TOKEN]
    })
    return tokenizer

def get_dataloader(tokenizer, dataset, batch_size, shuffle=False):

    return DL.create_correct_dataloader(
        tokenizer=tokenizer,
        texts=dataset["text"].tolist(),
        batch_size=batch_size,
        max_length=512,
        stride=256,
        shuffle=shuffle
    )


def get_model(tokenizer, lora = True, r = 8, alpha = 8, dropout = 0.05):
    manager = GPT2Manager(use_lora=lora, r=r, alpha=alpha, dropout=dropout)

    return manager.get_model(tokenizer=tokenizer)
# =========================
# Checkpoint
# =========================
def load_checkpoint(model, optimizer, scheduler, path, device):
    start_epoch, start_step, loss_for_save = 0, 0, np.inf

    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        start_step = checkpoint.get("step", 0)
        loss_for_save = checkpoint.get("loss_for_save", np.inf)

        print(f"Resuming from epoch {start_epoch}, step {start_step}, loss {loss_for_save}")

    # move optimizer tensors to device
    for state in optimizer.state.values():
        for k, v in state.items():
            state[k] = move_to_device(v, device)

    return start_epoch, start_step, loss_for_save


def save_checkpoint(model, optimizer, scheduler, epoch, step, path, device, loss):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "device": device,
        "loss": loss
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
        if p.requires_grad:
            print(f"{name} -> {p.requires_grad}")
    print(f"Total trained param: {sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)}")


def inspect_one_example(dataloader, tokenizer, IGNORE_INDEX):
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    masks = batch["attention_mask"][0]
    decoded_input = tokenizer.decode(input_ids.tolist())

    print("=== INPUT TEXT ===")
    print(decoded_input)

    print("\n=== TOKENS ===")
    print(input_ids.tolist())

    print("\n=== LABELS ===")
    print(labels.tolist())

    print("\n=== MASK ===")
    print(masks.tolist())

    print("\n=== TOKEN | LABEL | MASK ===")
    for token_id, label_id, mask in zip(input_ids.tolist(), labels.tolist(), masks.tolist()):
        token = tokenizer.decode([token_id])
        label = tokenizer.decode([label_id]) if label_id != IGNORE_INDEX else "IGN"

        print(f"{repr(token):>10} -> {label} -> {mask}")

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