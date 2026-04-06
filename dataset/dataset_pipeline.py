# dataset_pipeline.py
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import iterate

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
import srs.training.help_func as hf
importlib.reload(hf)
from srs.training.help_func import greedy_predict, temp_predict

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


new_df = df_all["text"][-5:].tolist()

dataloader_func = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=df_all["text"].tolist(),
    batch_size=batch_size,
    max_length=max_length,
    stride=stride,
    shuffle=shuffle
)


batch = next(iter(dataloader_func))

input_ids = batch["input_ids"][2]
labels = batch["labels"][2]

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



if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")


manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)


model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss(ignore_index=-100)

EPOCHS = 1

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1} ===")

    total_loss = 0

    for step, batch in enumerate(tqdm(dataloader_func)):

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # forward
        logits = model(input_ids)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        total_loss += loss.item()

        if step % 20 == 0:
            torch.mps.empty_cache()
        # if step % 10 == 0:
        #     print(f"step {step} | loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader_func)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")


prompt = "The structure of an atom:"

print(temp_predict(model, prompt, tokenizer, device, temperature = 0.8))

print(tokenizer.add_special_tokens())