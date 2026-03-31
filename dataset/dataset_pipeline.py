from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as DS_prep
from transformers import GPT2Tokenizer

import torch
from torch.nn import CrossEntropyLoss
from model.GPT_full_model import GPT2Manager

import dataset.Dataset_dataloader as MODEL
import importlib
importlib.reload(MODEL)

import inspect
print(inspect.getsource(MODEL))

df_all = DS_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})

tokenizer.pad_token = tokenizer.eos_token

dataloader = DL.create_dataloader_chat(
    tokenizer=tokenizer,
    texts=df_all["text"].tolist(),
    batch_size=2,
    max_length=32,
    stride=16
)



encoded = tokenizer.encode("<|assistant|>")
print("Encoded:", encoded)

tokens = tokenizer.convert_ids_to_tokens(encoded)
print("Tokens:", tokens)


if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")

manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)

IGNORE_INDEX = -100
loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch = next(iter(dataloader))

input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

logits = model(in_idx=input_ids)


for step in range(200):  # small overfit test
    logits = model(input_ids)

    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),  # [batch*seq_len, vocab]
        labels.reshape(-1)  # [batch*seq_len]
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print("Loss:", loss.item())


target_tokens = [
    labels.reshape(-1)[i].item()
    for i in range(len(labels.reshape(-1)))
    if labels.reshape(-1)[i] != -100
]

logits.reshape(-1, logits.size(-1)).shape

print(labels.reshape(-1).item())
print("TARGET:")
print(tokenizer.decode(target_tokens))

pred_ids = torch.argmax(logits, dim=-1)
print(pred_ids.shape)
print("MODEL OUTPUT:")
print(tokenizer.decode(pred_ids[1]))



for batch in dataloader:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)


    # forward pass
    logits = model(in_idx=input_ids)

    # HuggingFace models return loss if labels provided
    loss = loss_fn(
    logits.view(-1, logits.size(-1)),  # [batch*seq_len, vocab_size]
    labels.view(-1)                     # [batch*seq_len]
)

    # or manual:
    # logits = outputs.logits
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
