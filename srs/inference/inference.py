# inference.py

from srs.inference.help import *
from srs.training.help_func import (get_device, get_tokenizer, get_model, load_checkpoint)
from torch.optim import AdamW
device = get_device()
device = torch.device("mps")
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.1
TOP_K = 50

# ---- Load tokenizer ----
tokenizer = get_tokenizer()

# special tokens
ASSISTANT_TOKEN = Config.ASSIST_TOKEN
USER_TOKEN = Config.USER_TOKEN
model_name = "custom_gpt2"
# ---- Load model ----
model = get_model(model_name, tokenizer, lora=True, r=16,
                  alpha=128, dropout=0.1).to(device)
model_type = Config.MODEL_TYPE

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])

model.eval()

def build_prompt(history):
    prompt = ""
    for role, text in history:
        if role == "user":
            prompt += f" {USER_TOKEN*2} {text}"
        else:
            prompt += f" {ASSISTANT_TOKEN*2} {text}"

    prompt += f" {ASSISTANT_TOKEN*2}"

    return prompt

def clear_history(history):
    """Clear the entire conversation history."""
    history.clear()
    print("History has been cleared!")

def main():
    print("Chat started. Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            clear_history(history)
            print("Conversation history erased. Starting fresh!\n")
            continue

        history.append(("user", user_input))
        # print(f"history: {history}")

        prompt = build_prompt(history)
        # prompt = user_input
        # print(f"prompt: {prompt}")

        answer = temp_predict(model_name, model, prompt, tokenizer, device, max_new_tokens = MAX_NEW_TOKENS, temperature=TEMPERATURE)

        print(f"Bot: {answer}\n")

        history.append(("assistant", answer))

# prompt: <|user|> <|user|> Who wrote Hamlet? <|assistant|><|assistant|>
if __name__ == "__main__":
    main()