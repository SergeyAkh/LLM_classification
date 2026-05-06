# inference.py

from srs.inference.help import *
from srs.training.help_func import (get_device, get_tokenizer, get_model, load_checkpoint)
from torch.optim import AdamW
device = get_device()
device = torch.device("cpu")
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.3
TOP_K = 50

# ---- Load tokenizer ----
tokenizer = get_tokenizer()

# special tokens
ASSISTANT_TOKEN = Config.ASSIST_TOKEN
USER_TOKEN = Config.USER_TOKEN
model_name = "custom_gpt2"
# ---- Load model ----
model = get_model(model_name, tokenizer, lora=False,
                  r=Config.r,
                  alpha=Config.alpha,
                  dropout=0).to(device)

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

start_epoch, start_step, loss_for_save = load_checkpoint(
    model=model, optimizer=optimizer,
    scheduler=None, path=CHECKPOINT_PATH, device=device
)


model.eval()



def generate(model, input_ids):
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[:, -1, :] / TEMPERATURE

        # top-k sampling
        top_k_values, top_k_indices = torch.topk(next_token_logits, TOP_K)
        probs = torch.softmax(top_k_values, dim=-1)
        next_token = top_k_indices[0, torch.multinomial(probs, 1)]

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # stop if assistant finished (you can customize this)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return input_ids

def build_prompt(history):
    prompt = ""
    for role, text in history:
        if role == "user":
            prompt += f"{USER_TOKEN} {text}\n"
        else:
            prompt += f"{ASSISTANT_TOKEN} {text}\n"

    prompt += f"{ASSISTANT_TOKEN} "

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

        prompt = build_prompt(history)

        answer = temp_predict(model_name, model, prompt, tokenizer, device, max_new_tokens = MAX_NEW_TOKENS, temperature=TEMPERATURE)
        print(f"Bot: {answer}\n")

        history.append(("assistant", answer))


if __name__ == "__main__":
    main()