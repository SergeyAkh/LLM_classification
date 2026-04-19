# training.py
import math
import os

import numpy as np
import torch
from keras.src.ops import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from model.LoRA import LoRALinear
from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as ds_prep

from model.GPT_full_model import GPT2Manager
from srs.training.help_func import move_to_device, inspect_one_example, leyers_with_grad
from srs.inference.help import temp_predict
import importlib
import srs.LLM_classification.config as GPT
importlib.reload(ds_prep)

# =========================
# Setup
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.eos_token = Config.EOS_TOKEN
    tokenizer.add_special_tokens({
        "additional_special_tokens": [Config.USER_TOKEN, Config.ASSIST_TOKEN]
    })
    return tokenizer

def get_schedular(optimizer, dataloader, epoch):
    num_training_steps = len(dataloader) * epoch
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def get_dataloader(tokenizer, dataset, shuffle=False):

    return DL.create_correct_dataloader(
        tokenizer=tokenizer,
        texts=dataset["text"].tolist(),
        batch_size=4,
        max_length=512,
        stride=256,
        shuffle=shuffle
    )

def get_model(tokenizer, lora = True, r = 8, alpha = 8, dropout = 0.05):
    manager = GPT2Manager(use_lora=lora, r=r, alpha=alpha, dropout=dropout)

    return manager.get_model(tokenizer=tokenizer)

def load_checkpoint(model, optimizer, scheduler, path, device):
    start_epoch, start_step = 0, 0

    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0)
        start_step = checkpoint.get("step", 0)

        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    # move optimizer tensors to device
    for state in optimizer.state.values():
        for k, v in state.items():
            state[k] = move_to_device(v, device)

    return start_epoch, start_step

def save_checkpoint(model, optimizer, scheduler, epoch, step, path, device):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "device": device
    }, path)

def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
    scheduler=None,
    is_train=True,
    start_step=0,
    epoch=0,
    start_epoch=0,
    ignore_index=-100,
    SAVE = 1000,
    CHECKPOINT_PATH = None,
):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_tokens = 0

    # disable grad in eval
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for step, batch in enumerate(tqdm(dataloader)):

            # skip logic (only relevant for training resume)
            if is_train and epoch == start_epoch and step <= start_step:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            num_tokens = (labels != ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if is_train:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if step % SAVE == 0 and step > 0:
                    save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                    epoch=epoch, step=step, path=CHECKPOINT_PATH, device=device)
                    print(f"Checkpoint saved at current epoch: {epoch}, step: {step}, loss {avg_loss:.4f},"
                          f"Learning rate: {scheduler.get_last_lr()[0]}"
                          )
                if scheduler is not None:
                    scheduler.step()


        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    return avg_loss


# =========================
# Training Loop
# =========================
def train():
    EPOCHS = 20
    SAVE_EVERY = 1000

    training, val = ds_prep.get_data_preprocessed(Config, split_ratio=0.9)
    training["text"] = training["text"] + "<|endoftext|>"
    val["text"] = val["text"] + "<|endoftext|>"

    device = get_device()
    tokenizer = get_tokenizer()
    dataloader_tr = get_dataloader(tokenizer, training)
    dataloader_val = get_dataloader(tokenizer, val)

    model = get_model(tokenizer, lora=True,
                      r=Config.r,
                      alpha=Config.alpha,
                      dropout=Config.dropout).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-2, weight_decay=0.01, eps=1e-8
    )

    scheduler = get_schedular(optimizer, dataloader_tr, EPOCHS)
    # scheduler = None
    criterion = CrossEntropyLoss(ignore_index=Config.IGNORE_INDEX)

    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"

    start_epoch, start_step = load_checkpoint(
        model= model, optimizer=optimizer,
        scheduler=scheduler, path = CHECKPOINT_PATH, device = device
    )

    prompt = "Hi Assistant, how are you?"

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {epoch+1} ===")

        train_loss = run_epoch(
            model,
            dataloader_tr,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            is_train=True,
            start_step=start_step,
            epoch=epoch,
            start_epoch=start_epoch,
            ignore_index=Config.IGNORE_INDEX,
            SAVE=SAVE_EVERY,
            CHECKPOINT_PATH=CHECKPOINT_PATH
        )

        val_loss = run_epoch(
            model,
            dataloader_val,
            criterion,
            device,
            is_train=False,
            ignore_index=Config.IGNORE_INDEX
        )

        print(f"Loss on validation: {val_loss:.4f}, perplexity: {math.exp(val_loss)}, perplexity: {math.exp(val_loss)}")
        print(temp_predict(model, prompt, tokenizer, device, max_new_tokens=100, temperature=0.8))

        save_checkpoint(model = model, optimizer = optimizer,
                        scheduler = scheduler, epoch = epoch + 1,
                        step=0, path = CHECKPOINT_PATH, device=device)

        print(f"Model saved after whole epoch: {epoch}, loss: {train_loss:.4f}, perplexity: {math.exp(train_loss)},"
              f"Learning rate: {scheduler.get_last_lr()[0]}"
              )


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train()
