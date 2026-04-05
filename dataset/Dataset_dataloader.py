# Dataset_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

IGNORE_INDEX = -100

class CorrectChatDataset(Dataset):

    def __init__(self, texts: List[str], tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.labels = []

        self.assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        print(f"Processing {len(texts)} conversations...")

        for text_idx, text in enumerate(texts):
            # if text_idx % 100 == 0:
            #     print(f"  Processing {text_idx}/{len(texts)}")

            self._process_conversation(text, tokenizer, max_length, stride)

        print(f"✅ Dataset created: {len(self.input_ids)} samples")

    def _process_conversation(self, text: str, tokenizer, max_length: int, stride: int):

        full_tokens = tokenizer.encode(
            text,
            allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"},
            add_special_tokens=False
        )

        if len(full_tokens) < 2:
            return

        assistant_positions = []
        for i, token_id in enumerate(full_tokens):
            if token_id == self.assistant_token_id:
                assistant_positions.append(i)

        if not assistant_positions:
            return

        seq_len = len(full_tokens)

        for start in range(0, seq_len - 1, stride):
            end = min(start + max_length, seq_len)

            if end - start < 2:
                continue

            chunk_tokens = full_tokens[start:end]

            input_tokens = chunk_tokens[:-1]
            target_tokens = chunk_tokens[1:]

            # 🔥 НОВАЯ логика labels
            labels = [IGNORE_INDEX] * len(target_tokens)

            current_role = None
            print("here")
            for i, token_id in enumerate(input_tokens):

                if token_id == self.assistant_token_id:
                    current_role = "assistant"
                    continue

                elif token_id == self.user_token_id:
                    current_role = "user"
                    continue
                else:
                    if current_role == "assistant":
                        if target_tokens[i] in [self.user_token_id]:
                            continue
                        labels[i] = target_tokens[i]

            # если вообще нет обучающих токенов — пропускаем
            if all(l == IGNORE_INDEX for l in labels):
                continue

            # ✅ append остаётся как был
            min_len = min(len(input_tokens), len(labels))
            if min_len > 0:
                self.input_ids.append(
                    torch.tensor(input_tokens[:min_len], dtype=torch.long)
                )
                self.labels.append(
                    torch.tensor(labels[:min_len], dtype=torch.long)
                )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }


def create_correct_dataloader(
        tokenizer,
        texts: List[str],
        batch_size: int = 2,
        max_length: int = 512,
        stride: int = 256,
        shuffle: bool = True
):

    dataset = CorrectChatDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, lbls in zip(input_ids, labels):
            current_len = len(ids)
            padding_len = max_len - current_len

            padded_input_ids.append(torch.cat([
                ids,
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ]))

            padded_labels.append(torch.cat([
                lbls,
                torch.full((padding_len,), IGNORE_INDEX, dtype=torch.long)
            ]))

            attention_masks.append(torch.cat([
                torch.ones(current_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ]))

        return {
            'input_ids': torch.stack(padded_input_ids),
            'labels': torch.stack(padded_labels),
            'attention_mask': torch.stack(attention_masks)
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader



