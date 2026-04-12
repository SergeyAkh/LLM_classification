# training.py

import os
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2Tokenizer

from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as ds_prep

from model.GPT_full_model import GPT2Manager
from srs.training.help_func import move_to_device
from srs.inference.help import temp_predict

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

def get_dataloader(tokenizer, num_examples = None, shuffle=False):
    df_all = ds_prep.get_data_preprocessed(Config)
    if num_examples is not None:
        df_all = df_all[:num_examples]
    return DL.create_correct_dataloader(
        tokenizer=tokenizer,
        texts=df_all["text"].tolist(),
        batch_size=4,
        max_length=512,
        stride=256,
        shuffle=shuffle
    )

def get_model(tokenizer, lora = True, r = 8, alpha = 8):
    manager = GPT2Manager(use_lora=lora, r=r, alpha=alpha)
    model = manager.get_model(tokenizer=tokenizer)
    return model

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

# =========================
# Training Loop
# =========================
def train():
    device = get_device()
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokenizer)

    model = get_model(tokenizer, lora=True, r=8, alpha=8).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    criterion = CrossEntropyLoss(ignore_index=Config.IGNORE_INDEX)

    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"

    start_epoch, start_step = load_checkpoint(
        model, optimizer, CHECKPOINT_PATH, device
    )

    EPOCHS = 100
    SAVE_EVERY = 1000

    prompt = "Who is Julius Caesar? And how did he died?"

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        print(f"\n=== Epoch {epoch+1} ===")

        total_loss = 0
        total_tokens = 0
        for step, batch in enumerate(tqdm(dataloader)):

            if epoch == start_epoch and step <= start_step:
                continue
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            num_tokens = (labels[0] != Config.IGNORE_INDEX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            avg_loss = total_loss / total_tokens

            if step % SAVE_EVERY == 0 and step > 0:
                save_checkpoint(model, optimizer, epoch, step, CHECKPOINT_PATH, device)
                print(f"Checkpoint saved at current epoch: {epoch + 1}, step: {step}, loss {avg_loss:.4f}")

        print(temp_predict(model, prompt, tokenizer, device, max_new_tokens=100, temperature=0.8))

        avg_loss = total_loss / len(dataloader)
        save_checkpoint(model, optimizer, epoch + 1, 0, CHECKPOINT_PATH, device)

        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    train()
