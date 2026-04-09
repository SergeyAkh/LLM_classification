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
from srs.training.help_func import greedy_predict, temp_predict, leyers_with_grad
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


new_df = df_all["text"][:3].tolist()

dataloader_func = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=
    df_all["text"].tolist(),
# new_df,
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
model = model.to(device)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

start_epoch = 0
start_step = 0
EPOCHS = 100
SAVE_EVERY_STEPS = 500


leyers_with_grad(model)

for name, param in model.named_parameters():
    if "A" in name or "B" in name:
        param.requires_grad = True
        print(f"{name} -> {param.requires_grad}")
    else:
        param.requires_grad = False

        # print(f"{name} -> {param.requires_grad}")

from model.LoRA import LoRALinear
for module in model.modules():
    if isinstance(module, LoRALinear):
        module.A.requires_grad = True
        module.B.requires_grad = True

for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", trainable)
for p in model.parameters():
    if p.requires_grad:
        print(p)

model.train()

from model.LoRA import LoRALinear
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(name, module.A.abs().mean().item(), module.B.abs().mean().item())

for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
        if torch.isnan(module.A).any() or torch.isnan(module.B).any():
            print("NaN detected in", name)

batch = next(iter(dataloader_func))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)
logits = model(input_ids)
loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
print("Initial batch loss:", loss.item())

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch",0)
    start_step = checkpoint.get("step", 0)
    device = checkpoint.get("device","mps")
    print(f"Resuming from epoch {start_epoch}, step {start_step}")

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

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer.step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

        if step % 500 == 0 and step > 0:
            torch.mps.empty_cache()
        # -----------------------------
        # Save checkpoint every N steps
        # -----------------------------
        if step % SAVE_EVERY_STEPS == 0 and step > 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "device": device
            }, CHECKPOINT_PATH)
            loss_checkpoint = total_loss / len(dataloader_func)
            print(f"Checkpoint saved at step {step}, loss: {loss_checkpoint:.4f}")

    avg_loss = total_loss / len(dataloader_func)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    step = 0
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch + 1,
        "step": step,
        "device": device
    }, CHECKPOINT_PATH)

prompt = "Capital of France?"

print(temp_predict(model, prompt, tokenizer, device, temperature = 0.8))

print(tokenizer.add_special_tokens())





