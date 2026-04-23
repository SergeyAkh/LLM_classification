# training.py
import math
import os

import numpy as np
import torch

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from transformers import GPT2Tokenizer, get_cosine_schedule_with_warmup

from model.LoRA import LoRALinear
from srs.LLM_classification.config import Config

import dataset.get_prep_data as ds_prep

from srs.training.help_func import (get_device, get_schedular, save_checkpoint,
                                    load_checkpoint, get_tokenizer, get_model,
                                    inspect_one_example, leyers_with_grad, get_dataloader)
from srs.inference.help import temp_predict
import importlib
import srs.training.help_func as GPT
importlib.reload(GPT)

# =========================
# Setup
# =========================

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
    save_results = None,
    loss_for_save = None
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
                optimizer.zero_grad()

            logits = model(input_ids)
            print("logits", logits.shape)
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
                    if avg_loss < loss_for_save:
                        loss_for_save = avg_loss
                        if save_results:
                            save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                        epoch=epoch, step=step, path=CHECKPOINT_PATH, device=device, loss=avg_loss)
                            print(f"Checkpoint saved at current epoch: {epoch}, step: {step}, loss {avg_loss:.4f}, perplexity: {math.exp(avg_loss)},"
                                  f"Learning rate: {scheduler.get_last_lr()[0]}"
                                  )
                if scheduler is not None:
                    scheduler.step()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    return avg_loss, loss_for_save

# =========================
# Training Loop
# =========================
def train():
    EPOCHS = 2
    SAVE_EVERY = 1000

    training, val = ds_prep.get_data_preprocessed(Config, split_ratio=0.9)
    # val = training[20:22].reset_index(drop=True)
    # training = training[:20].reset_index(drop=True)
    # print(training.loc[0, "text"])
    # print(len(training["text"][8]))
    device = get_device()
    tokenizer = get_tokenizer()

    dataloader_tr = get_dataloader(tokenizer, dataset = training, batch_size=16, shuffle=False)
    dataloader_val = get_dataloader(tokenizer, dataset = val, batch_size=16, shuffle=False)

    model = get_model(tokenizer, lora=True,
                      r=Config.r,
                      alpha=Config.alpha,
                      dropout=Config.dropout).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01
    )

    scheduler = get_schedular(optimizer, dataloader_tr, EPOCHS)
    # scheduler = None
    criterion = CrossEntropyLoss(ignore_index=Config.IGNORE_INDEX)

    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"

    start_epoch, start_step, loss_for_save = load_checkpoint(
        model= model, optimizer=optimizer,
        scheduler=scheduler, path = CHECKPOINT_PATH, device = device
    )

    prompt = "My name is Sergey"

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {epoch+1} ===")

        train_loss, loss_for_save = run_epoch(
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
            CHECKPOINT_PATH=CHECKPOINT_PATH,
            save_results=True,
            loss_for_save=loss_for_save
        )

        val_loss, loss_for_save_val = run_epoch(
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
                        step=0, path = CHECKPOINT_PATH, device=device, loss=loss_for_save)

        print(f"Model saved after whole epoch: {epoch}, loss: {train_loss:.4f}, perplexity: {math.exp(train_loss)},"
              f"Learning rate: {scheduler.get_last_lr()[0]}"
              )


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train()
