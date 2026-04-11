# dataset_pipeline.py

from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as ds_prep
from transformers import GPT2Tokenizer

import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from model.GPT_full_model import GPT2Manager
import os

import importlib
import model.GPT_full_model as hf
importlib.reload(hf)
from srs.training.help_func import (greedy_predict, temp_predict,
                                    leyers_with_grad, move_to_device)
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '1.0'
df_all = ds_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})

LEARNING_RATE = 1e-3

IGNORE_INDEX = -100
batch_size = 4
max_length = 512
stride = 256
shuffle = False


new_df = df_all["text"][:200].tolist()

dataloader_func = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=
    # df_all["text"].tolist(),
new_df,
    batch_size=batch_size,
    max_length=max_length,
    stride=stride,
    shuffle=shuffle
)


if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")


manager = GPT2Manager(use_lora=True, r=8,alpha=8)
model = manager.get_model(tokenizer=tokenizer)


optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

start_epoch = 0
start_step = 0
EPOCHS = 100
SAVE_EVERY_STEPS = 10

prompt = "Write a poem about AI"
CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")

    # model, optimizer, start_epoch, start_step, device = load_checkpoint()
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch",0)
    start_step = checkpoint.get("step", 0)
    device = checkpoint.get("device","mps")
    print(f"Resuming from epoch {start_epoch}, step {start_step}")


model = model.to(device)
for param_state in optimizer.state.values():
    for k in param_state:
        param_state[k] = move_to_device(param_state[k], device)

batch = next(iter(dataloader_func))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)
logits = model(input_ids)
loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
total_loss = 0
num_tokens = (labels[0] != -100).sum().item()
total_loss += loss.item() * num_tokens
avg_loss = total_loss / total_tokens
print(loss.item())
print(sum((labels[0] != -100)))


for epoch in range(start_epoch, EPOCHS):
    model.train()
    print(f"\n=== Epoch {epoch+1} ===")

    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader_func)):
        if epoch == start_epoch and step < start_step:
            continue
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # forward
        logits = model(input_ids)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer.step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        # if step % 500 == 0 and step > 0:
        #     torch.mps.empty_cache()
        # -----------------------------
        # Save checkpoint every N steps
        # -----------------------------
        print(step, step == len(dataloader_func))
        if (step % SAVE_EVERY_STEPS == 0 and step > 0) or step == len(dataloader_func):
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1 if step == len(dataloader_func) else epoch,
                "step": step,
                "device": device
            }, CHECKPOINT_PATH)
            loss_checkpoint = total_loss / len(dataloader_func)
            print(f"Checkpoint saved at step {step}, loss: {loss_checkpoint:.4f}")
    print(temp_predict(model, prompt, tokenizer, device, max_new_tokens=100,temperature=0.8))
    avg_loss = total_loss / len(dataloader_func)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")






