# training.py
import math
import os
import random

# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0'
import torch

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import dataloader

from tqdm import tqdm
from srs.LLM_classification.config import Config

import dataset.get_prep_data as ds_prep

from srs.training.help_func import (get_device, get_schedular, save_checkpoint,
                                    load_checkpoint, get_tokenizer, get_model,
                                    inspect_one_example, leyers_with_grad, get_dataloader)
from srs.inference.help import temp_predict
# import importlib
# import srs.training.help_func as GPT
# importlib.reload(GPT)
# import inspect
# print(inspect.getsource(ds_prep.oasst1_df))
# =========================
# Setup
# =========================

def run_epoch(
    model_name,
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
    loss_for_save = None,
    accum_steps = 1,
    tokenizer = None,
    prompt = None,

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
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if is_train:
                if step % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

            # with torch.autocast(device_type="mps", dtype=torch.float16):
            logits = model(input_ids=input_ids,
                               attention_mask=attention_mask
                               )
            if model_name == "gpt2":
                output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
                loss = output.loss
                logits = output.logits
            else:
                logits = logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()
            loss = criterion(
                    logits.reshape(-1, logits.size(-1)),  # reshape > view (safer)
                    labels.reshape(-1)
                    )

            num_tokens = (labels != ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if is_train:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

                loss = loss / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    if scheduler is not None:
                        lr_to_print = scheduler.get_lr()[0]
                        scheduler.step()
                    else:
                        lr_to_print = optimizer.param_groups[0]["lr"]

                if step % SAVE == 0 and step > 0:
                    print(temp_predict(model_name = model_name, model = model, prompt = prompt, tokenizer = tokenizer,
                                       device = device, max_new_tokens=100, temperature=0.5))
                    if avg_loss < loss_for_save:
                        loss_for_save = avg_loss
                        print(
                            f"Results: current epoch: {epoch}, step: {step}, loss {avg_loss:.4f}, simple_loss: {loss.item()}, "
                            f"perplexity: {math.exp(avg_loss)},{math.exp(loss.item())}"
                            f"Learning rate: {lr_to_print}"
                            )
                        if save_results:
                            save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler,
                                            epoch=epoch, step=step, path=CHECKPOINT_PATH, device=device, loss=avg_loss)
                            print(f"Checkpoint saved")

                    else:
                        print(f"Not saved: epoch: {epoch}, step: {step}, loss {avg_loss:.4f}, simple_loss: {loss.item()}, perplexity: {math.exp(avg_loss)},"
                                      f"Learning rate: {lr_to_print}")

                if step % 4:
                    torch.mps.empty_cache()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0

    return avg_loss, loss_for_save

# =========================
# Training Loop
# =========================
def train():
    model_name = "custom_gpt2"
    EPOCHS = 1
    SAVE_EVERY = 100
    accum_steps = 2
    batch_size = 2
    max_length = 256
    stride = 256
    training, val = ds_prep.get_data_preprocessed(Config, split_ratio=0.9, language="en")
    training["text"] = training["text"].str.split("'<|endoftext|>'", regex=True).str[0]
    training["text"] = training["text"].str.replace(r"<\|\s*$", "<|endoftext|>", regex=True)

    val["text"] = val["text"].str.split("'<|endoftext|>'", regex=True).str[0]
    val["text"] = val["text"].str.replace(r"<\|\s*$", "<|endoftext|>", regex=True)

    val = training[1000:1010]
    training = training[:100]
    print(training.loc[1, "text"])


    device = get_device()
    device = torch.device("cpu")
    tokenizer = get_tokenizer()

    dataloader_tr = get_dataloader(tokenizer, dataset = training,
                                   batch_size=batch_size, max_length = max_length, stride = stride, shuffle=False)
    dataloader_val = get_dataloader(tokenizer, dataset = val,
                                    batch_size=batch_size, max_length = max_length, stride = stride, shuffle=False)

    inspect_one_example(dataloader_tr, tokenizer, Config.IGNORE_INDEX)

    model = get_model(model_name,tokenizer, lora=False,
                      r=Config.r,
                      alpha=Config.alpha,
                      dropout=0).to(device)

    leyers_with_grad(model)

    model.train()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.00001
    )
    criterion = CrossEntropyLoss(ignore_index=Config.IGNORE_INDEX)

    scheduler = get_schedular(optimizer, dataloader_tr, EPOCHS, accum_steps)

    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}.pt"

    start_epoch, start_step, loss_for_save = load_checkpoint(
        model= model, optimizer=optimizer,
        scheduler=scheduler, path = CHECKPOINT_PATH, device = device
    )

    val = val.reset_index(drop=True)
    num = random.randint(0, len(val))
    answer = val["text"][num].split("<|user|>")[1].split("<|assistant|>")[1].strip()
    prompt = val["text"][num].split("<|user|>")[1].split("<|assistant|>")[0].strip()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n=== Epoch {epoch+1} ===")

        train_loss, loss_for_save = run_epoch(
            model_name,
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
            loss_for_save=loss_for_save,
            accum_steps = accum_steps,
            tokenizer = tokenizer,
            prompt = prompt
        )

        val_loss, loss_for_save_val = run_epoch(
            model_name,
            model,
            dataloader_val,
            criterion,
            device,
            is_train=False,
            ignore_index=Config.IGNORE_INDEX
        )

        print(f"Loss on validation: {val_loss:.4f}, loss on train: {train_loss}, perplexity: {math.exp(val_loss)}, perplexity: {math.exp(val_loss)}")
        print(f"Question: {prompt}")
        print(f"True answer: {answer}")
        print(f"predictions:")
        print(temp_predict(model_name,model, prompt, tokenizer, device,
                           max_new_tokens=100, temperature=0.3))


        save_checkpoint(model = model, optimizer = optimizer,
                            scheduler = scheduler, epoch = epoch + 1,
                            step=0, path = CHECKPOINT_PATH, device=device, loss=loss_for_save)

        print(f"Model saved after whole epoch: {epoch}, loss: {train_loss:.4f}, perplexity: {math.exp(train_loss)},"
                  # f"Learning rate: {scheduler.get_last_lr()[0]}"
                  )


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train()
