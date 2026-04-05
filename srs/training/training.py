# training.py
from dataset.dataset_pipeline import dataloader, tokenizer, df_all
from srs.training.help_func import predict


import dataset.dataset_pipeline as ds_pipeline
import importlib
importlib.reload(ds_pipeline)

from torch.nn import CrossEntropyLoss
from model.GPT_full_model import GPT2Manager
import torch
from tqdm import tqdm
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'

if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")


LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0
IGNORE_INDEX = -100
batch_size = 2
max_length = 512
stride = 256
shuffle = False

data_preproc = df_all["text"].tolist()

data_preproc =         ["""<|user|> Give three tips for staying healthy.
<|assistant|> 1. Eat a balanced and nutritious diet...
2. Engage in regular physical activity...
3. Get enough sleep...<|user|> What about mental health?
<|assistant|> Mental health is equally important! Practice mindfulness, maintain social connections, and seek help when needed.<|endoftext|>"""
    ]
dataloader = dataloader(data_preproc, tokenizer, batch_size, max_length, stride, shuffle)

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











test_prompt = ["""<|user|>Give three tips for staying healthy<|assistant|>"""]

manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)

loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

batch = next(iter(dataloader))
input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

logits = model(in_idx=input_ids)
pred_ids = torch.argmax(logits, dim=-1)

print(tokenizer.decode(input_ids))










EPOCHS = 101
loss_history = []
step_history = []
global_step = 0
model.train()
for epoch in range(EPOCHS):
    torch.mps.empty_cache()
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'=' * 60}")

    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(in_idx=input_ids)  # [batch_size, seq_len, vocab_size]

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
            labels.view(-1)  # [batch_size * seq_len]
        )
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                pred_text = predict(model, test_prompt, tokenizer)
                print(pred_text[0])

        optimizer.step()

        current_loss = loss.item()
        epoch_loss += current_loss
        global_step += 1

        loss_history.append(current_loss)
        step_history.append(global_step)

        avg_loss = epoch_loss / (batch_idx + 1)

        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
        })
        break

    # Статистика эпохи
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch + 1} completed!")
    print(f"   Average loss: {avg_epoch_loss:.4f}")








