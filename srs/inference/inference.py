# inference.py

from srs.inference.help import *

device = get_device()

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50

# ---- Load tokenizer ----
tokenizer = get_tokenizer()

# special tokens
ASSISTANT_TOKEN = Config.ASSIST_TOKEN
USER_TOKEN = Config.USER_TOKEN

# ---- Load model ----
model = get_model(tokenizer, lora=False, r=16, alpha=16).to(device)
CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / "checkpoint_LoRA.pt"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
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


def main():
    print("Chat started. Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        history.append(("user", user_input))

        prompt = build_prompt(history)

        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # output_ids = generate(model, input_ids)
        # decoded = tokenizer.decode(output_ids[0])
        # extract only last assistant answer
        # answer = decoded.split(ASSISTANT_TOKEN)[-1].strip()


        answer = temp_predict(model, prompt, tokenizer, device, temperature=TEMPERATURE)
        print(f"Bot: {answer}\n")

        history.append(("assistant", answer))


if __name__ == "__main__":
    main()