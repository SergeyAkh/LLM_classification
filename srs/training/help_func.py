# help_func.py
from tqdm import tqdm

import torch
import torch.nn.functional as F

# import importlib
# importlib.reload(ds_prep)

# import inspect
# print(inspect.getsource(ds_prep.oasst1_df))


def temp_predict(model, prompt, tokenizer, device, temperature):
    model.eval()
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(50):
            logits = model(input_ids)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
    input_ids = input_ids[:, (input_ids[0] == assistant_token_id).nonzero(as_tuple=True)[0][0] + 1:]
    return tokenizer.decode(input_ids[0])

def greedy_predict(model, prompt, tokenizer,device):
    model.eval()
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(50):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    input_ids = input_ids[:, (input_ids[0] == assistant_token_id).nonzero(as_tuple=True)[0][0] + 1:]
    return tokenizer.decode(input_ids[0])



def inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx, INDEX):
    """Детальная проверка сэмпла"""

    input_ids = batch['input_ids'][sample_idx]
    labels = batch['labels'][sample_idx]
    attention_mask = batch['attention_mask'][sample_idx]

    valid_len = attention_mask.sum().item()

    print(f"\n{'=' * 70}")
    print(f"Batch {batch_idx}, Sample {sample_idx}")
    print(f"{'=' * 70}")

    print(f"{'Pos':<4} {'Token':<25} {'Trainable':<12} {'Token ID':<8}")
    print("-" * 70)

    for i in range(valid_len):
        token_id = input_ids[i].item()
        token = tokenizer.decode([token_id]).replace('\n', '\\n')
        is_trainable = (labels[i] != INDEX).item()

        trainable_mark = "✅ LEARN" if is_trainable else "❌ IGNORE"

        if token in ["<|user|>", "<|assistant|>", "<|endoftext|>", "\\n"]:
            trainable_mark = f"🔷 {trainable_mark}"

        print(f"{i:<4} {token:<25} {trainable_mark:<12} {token_id:<8}")

        if i > 0 and is_trainable and (labels[i - 1] == INDEX).item():
            print(f"     ↑ Beginning of training ↑")

    total_trainable = (labels[:valid_len] != INDEX).sum().item()
    print(f"\n📊 Statistics:")
    print(f"  Total tokens: {valid_len}")
    print(f"  Trainable tokens: {total_trainable}")
    print(f"  Percentage: {total_trainable / valid_len * 100:.1f}%")

def check_one_batch(dataloader, tokenizer, INDEX):
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n{'#' * 70}")
        print(f"BATCH {batch_idx}")
        print(f"{'#' * 70}")

        for sample_idx in range(batch['input_ids'].shape[0]):
            inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx, INDEX)

        if batch_idx >= 0:
            break