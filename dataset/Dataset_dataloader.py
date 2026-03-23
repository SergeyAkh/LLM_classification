from torch.utils.data import Dataset, DataLoader
import torch

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, pad_token_id=50256):
        self.input_ids = []
        self.target_ids = []
        eos_token = "<|endoftext|>"
        full_text = eos_token.join(txt.astype(str).tolist()) + eos_token
        # Tokenize the entire text
        token_ids = tokenizer.encode(full_text, allowed_special={eos_token})
        # Use a sliding window to chunk the book into overlapping sequences of max_length

        if len(token_ids) < max_length:
            input_chunk = token_ids + [pad_token_id] * (max_length - len(token_ids))
            target_chunk = input_chunk[1:] + [pad_token_id]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        else:
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1:i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class SpamDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=None, pad_token_id=50256):
        self.data = data
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

def create_dataloader_v1(tokenizer, txt, batch_size=4, max_length=None,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0, proc_type=None):

    # Create dataset
    if proc_type == "generate_next_word":
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    elif proc_type == "classification":
        dataset = SpamDataset(txt, tokenizer, max_length=None, pad_token_id=50256)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader