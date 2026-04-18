import torch
import torch.nn.functional as F

import os
import torch
from transformers import GPT2Tokenizer

from srs.LLM_classification.config import Config
from model.GPT_full_model import GPT2Manager

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.eos_token = Config.EOS_TOKEN
    tokenizer.add_special_tokens({
        "additional_special_tokens": [Config.USER_TOKEN, Config.ASSIST_TOKEN]
    })
    return tokenizer

def get_model(tokenizer, lora = True, r = 8, alpha = 8, dropout = 0.5):
    manager = GPT2Manager(use_lora=lora, r=r, alpha=alpha, dropout=dropout)
    model = manager.get_model(tokenizer=tokenizer, inference=True)
    return model

def build_prompt(history):
    prompt = ""
    for role, text in history:
        prompt += f"<|{role}|> {text}\n"
    prompt += "<|assistant|>"
    print(f"Prompt: {prompt}")
    return prompt

def temp_predict(model, prompt, tokenizer, device, temperature=1.0, max_new_tokens=50):
    model.eval()

    prompt = f"<|user|> {prompt} <|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    # new_tokens = input_ids[0][original_length:]
    # response = tokenizer.decode(new_tokens)
    decoded = tokenizer.decode(input_ids[0])
    decoded = decoded.split("<|assistant|>")[-1].strip()
    return decoded.split("<|endoftext|>")[0]


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