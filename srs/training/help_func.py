# help_func.py
from tqdm import tqdm

import torch
import torch.nn.functional as F

# import importlib
# importlib.reload(ds_prep)

# import inspect
# print(inspect.getsource(ds_prep.oasst1_df))

def predict(model, texts, tokenizer):
    predictions = []
    for text_idx, text in enumerate(texts):
        full_tokens = tokenizer.encode(
            text,
            allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"},
            add_special_tokens=False
        )

        full_tokens = torch.tensor(full_tokens).unsqueeze(0)
        logits = model(in_idx=full_tokens)
        pred_ids = torch.argmax(logits, dim=-1)
        pred = tokenizer.decode(pred_ids)
        predictions.append(pred)
    return predictions



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