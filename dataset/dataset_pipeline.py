import os
import pandas as pd
from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as DS_prep
from transformers import GPT2Tokenizer




df_all = DS_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})



dataloader = DL.create_dataloader_chat(
    tokenizer=tokenizer,
    texts=df_all["text"].tolist(),
    batch_size=2,
    max_length=32,
    stride=16
)



