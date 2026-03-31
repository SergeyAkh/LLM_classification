import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

IGNORE_INDEX = -100

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # pad sequences to the max length in this batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "labels": labels
    }

class ChatDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, stride):
        self.input_ids = []
        self.labels = []

        eos_token = "<|endoftext|>"
        assistant_token = "<|assistant|>"
        user_token = "<|user|>"

        assistant_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
        user_token_id = tokenizer.convert_tokens_to_ids(user_token)

        for text in texts:  # ✅ iterate over conversations
            token_ids = tokenizer.encode(
                text,
                allowed_special={eos_token, user_token, assistant_token}
            )

            # 🔥 handle short conversations
            if len(token_ids) <= max_length:
                input_chunk = token_ids[:-1]
                target_chunk = token_ids[1:]
                chunks = [(input_chunk, target_chunk)]
            else:
                chunks = []
                for i in range(0, len(token_ids), stride):
                    input_chunk = token_ids[i:i + max_length]

                    if len(input_chunk) < 2:
                        continue

                    # 🔥 enforce hard limit
                    input_chunk = input_chunk[:max_length]

                    target_chunk = token_ids[i + 1:i + max_length + 1]

                    min_len = min(len(input_chunk), len(target_chunk))
                    input_chunk = input_chunk[:min_len]
                    target_chunk = target_chunk[:min_len]

                    chunks.append((input_chunk, target_chunk))

            # 🔥 process chunks
            for input_chunk, target_chunk in chunks:
                labels = target_chunk.copy()

                in_assistant = False
                has_valid_label = False

                for j in range(len(input_chunk)):
                    token_id = input_chunk[j]

                    if token_id == assistant_token_id:
                        in_assistant = True
                        labels[j] = IGNORE_INDEX
                        continue

                    if token_id == user_token_id:
                        in_assistant = False
                        labels[j] = IGNORE_INDEX
                        continue

                    if not in_assistant:
                        labels[j] = IGNORE_INDEX
                    else:
                        has_valid_label = True  # 🔥 we actually train here

                # 🔥 use THIS instead of has_assistant
                if not has_valid_label:
                    continue

                self.input_ids.append(torch.tensor(input_chunk))
                self.labels.append(torch.tensor(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

def create_dataloader_chat(tokenizer, texts, batch_size=4, max_length=1024,
                           stride=512, shuffle=True):

    dataset = ChatDataset(texts, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

