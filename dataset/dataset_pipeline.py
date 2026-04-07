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
import srs.training.help_func as hf
importlib.reload(hf)
from srs.training.help_func import greedy_predict, temp_predict
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '1.0'
df_all = ds_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})

def dataloader(data, tokenizer_func, batch_size, max_length, stride, shuffle):
    dataloader_func = DL.create_correct_dataloader(
        tokenizer=tokenizer_func,
        texts=data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle
    )
    return dataloader_func

LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0
IGNORE_INDEX = -100
batch_size = 4
max_length = 512
stride = 256
shuffle = False

df_grather = df_all[df_all["text"].str.len() > 1024]
new_df = df_all["text"][-5:].tolist()

dataloader_func = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=df_all["text"].tolist(),
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

manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss(ignore_index=-100)
start_epoch = 0
start_step = 0
EPOCHS = 1
SAVE_EVERY_STEPS = 500
accum_steps = 4

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH +  "checkpoint.pt"
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    start_step = checkpoint["step"]
    device = checkpoint["device"]
    print(f"Resuming from epoch {start_epoch}, step {start_step}")

model.train()
global_step = start_step

for epoch in range(start_epoch, EPOCHS):
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
        loss = loss / accum_steps
        # backward
        loss.backward()
        global_step += 1
        if (step + 1) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # optimizer.step()
        # optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()

        if step % 20 == 0:
            torch.mps.empty_cache()

        # -----------------------------
        # Save checkpoint every N steps
        # -----------------------------
        if global_step % SAVE_EVERY_STEPS == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "device": device
            }, CHECKPOINT_PATH)
            print(f"Checkpoint saved at step {global_step}")

    avg_loss = total_loss / len(dataloader_func)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": start_epoch + epoch + 1,
        "device": device
    }, Config.MODEL_WEIGHTS_PATH + "checkpoint.pt")
# Save model weights
torch.save(model.state_dict(), Config.MODEL_WEIGHTS_PATH + "model.pt")

# (optional but recommended) save optimizer state
torch.save(optimizer.state_dict(), Config.MODEL_WEIGHTS_PATH + "optimizer.pt")

prompt = "The structure of an atom:"

print(temp_predict(model, prompt, tokenizer, device, temperature = 0.8))

print(tokenizer.add_special_tokens())