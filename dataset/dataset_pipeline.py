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

import importlib
importlib.reload(DL)

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
batch_size = 2
max_length = 512
stride = 256
shuffle = False

data_preproc =         ["""<|user|> Give three tips for staying healthy.
<|assistant|> 1. Eat a balanced and nutritious diet...
2. Engage in regular physical activity...
3. Get enough sleep...<|user|> What about mental health?
<|assistant|> Mental health is equally important! Practice mindfulness, maintain social connections, and seek help when needed.<|endoftext|>"""
    ]

dataloader_func = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=data_preproc,
    batch_size=batch_size,
    max_length=max_length,
    stride=stride,
    shuffle=shuffle
)


batch = next(iter(dataloader_func))

input_ids = batch["input_ids"][0]
labels = batch["labels"][0]
# print(input_ids)
# print(input_ids[:-1])
# print(input_ids[1:])
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




device = torch.device("cpu")
manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)

model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss(ignore_index=-100)

EPOCHS = 100

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1} ===")

    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader_func)):

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # forward
        logits = model(input_ids)

        # reshape для CrossEntropy
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (стабильность)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        # 🔍 debug каждые N шагов
        if step % 10 == 0:
            print(f"step {step} | loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

model.eval()

prompt = "<|user|> Give tips\n<|assistant|>"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    for _ in range(50):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

print(tokenizer.decode(input_ids[0]))