import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import List, Tuple
import re
import logging

IGNORE_INDEX = -100

from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch, tokenizer):
#     input_ids = [item["input_ids"] for item in batch]
#     labels = [item["labels"] for item in batch]
#
#     # pad sequences to the max length in this batch
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#     labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#
#     return {
#         "input_ids": input_ids,
#         "labels": labels
#     }

# class ChatDataset(Dataset):
#     def __init__(self, texts, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.labels = []
#
#         eos_token = "<|endoftext|>"
#         assistant_token = "<|assistant|>"
#         user_token = "<|user|>"
#
#         assistant_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
#
#         for text in texts:  # ✅ iterate over conversations
#             token_ids = tokenizer.encode(
#                 text,
#                 allowed_special={eos_token, user_token, assistant_token}
#             )
#
#             # 🔥 handle short conversations
#             if len(token_ids) <= max_length:
#                 input_chunk = token_ids[:-1]
#                 target_chunk = token_ids[1:]
#                 chunks = [(input_chunk, target_chunk)]
#             else:
#                 chunks = []
#                 for i in range(0, len(token_ids), stride):
#                     input_chunk = token_ids[i:i + max_length]
#
#                     if len(input_chunk) < 2:
#                         continue
#
#                     # 🔥 enforce hard limit
#                     input_chunk = input_chunk[:max_length]
#
#                     target_chunk = token_ids[i + 1:i + max_length + 1]
#
#                     min_len = min(len(input_chunk), len(target_chunk))
#                     input_chunk = input_chunk[:min_len]
#                     target_chunk = target_chunk[:min_len]
#
#                     chunks.append((input_chunk, target_chunk))
#
#             # 🔥 process chunks
#             for input_chunk, target_chunk in chunks:
#                 labels = target_chunk.copy()
#
#                 # find last assistant token
#                 last_assistant_idx = None
#                 for j in range(len(input_chunk)):
#                     if input_chunk[j] == assistant_token_id:
#                         last_assistant_idx = j
#
#                 if last_assistant_idx is not None:
#                     labels[:last_assistant_idx] = [IGNORE_INDEX] * last_assistant_idx
#                 else:
#                     continue
#
#                 self.input_ids.append(torch.tensor(input_chunk))
#                 self.labels.append(torch.tensor(labels))
#
#     def __len__(self):
#         return len(self.input_ids)
#
#     def __getitem__(self, idx):
#         return {
#             "input_ids": self.input_ids[idx],
#             "labels": self.labels[idx]
#         }

class CorrectChatDataset(Dataset):
    """Правильная версия с корректным маскированием"""

    def __init__(self, texts: List[str], tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.labels = []

        self.assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.user_token_id = tokenizer.convert_tokens_to_ids("<|user|>")
        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        print(f"Processing {len(texts)} conversations...")

        for text_idx, text in enumerate(texts):
            if text_idx % 100 == 0:
                print(f"  Processing {text_idx}/{len(texts)}")

            self._process_conversation(text, tokenizer, max_length, stride)

        print(f"✅ Dataset created: {len(self.input_ids)} samples")

    def _process_conversation(self, text: str, tokenizer, max_length: int, stride: int):
        """Обрабатывает диалог с правильным маскированием"""

        # Токенизируем весь диалог
        full_tokens = tokenizer.encode(
            text,
            allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"},
            add_special_tokens=False
        )

        if len(full_tokens) < 2:
            return

        # Находим все позиции <|assistant|>
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

            # Берем чанк
            chunk_tokens = full_tokens[start:end]

            # Создаем input и target
            input_tokens = chunk_tokens[:-1]
            target_tokens = chunk_tokens[1:]

            # Создаем labels
            labels = target_tokens.copy()

            # Находим позиции <|assistant|> внутри чанка
            chunk_assistant_positions = [
                pos - start for pos in assistant_positions
                if start <= pos < end
            ]

            if not chunk_assistant_positions:
                # Нет ассистента в этом чанке - пропускаем
                continue

            # Находим последний <|assistant|> в чанке
            last_assistant_idx = max(chunk_assistant_positions)

            # Маскируем все токены ДО последнего <|assistant|>
            # И САМ <|assistant|> тоже маскируем
            if last_assistant_idx < len(labels):
                # Маскируем все токены до и включая <|assistant|>
                labels[:last_assistant_idx] = [IGNORE_INDEX] * (last_assistant_idx + 1)

            # Дополнительно маскируем токены переноса строки, если нужно
            # (опционально - можно оставить для обучения форматированию)

            # Убеждаемся, что длины совпадают
            min_len = min(len(input_tokens), len(labels))
            if min_len > 0:
                self.input_ids.append(torch.tensor(input_tokens[:min_len], dtype=torch.long))
                self.labels.append(torch.tensor(labels[:min_len], dtype=torch.long))

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
    """Создает DataLoader с правильным маскированием"""

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

# def create_dataloader_chat(tokenizer, texts, batch_size=4, max_length=1024,
#                            stride=512, shuffle=True):
#
#     dataset = ChatDataset(texts, tokenizer, max_length, stride)
#
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=False,
#         collate_fn=partial(collate_fn, tokenizer=tokenizer)
#     )

