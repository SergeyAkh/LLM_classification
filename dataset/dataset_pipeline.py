from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as ds_prep
from transformers import GPT2Tokenizer
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from model.GPT_full_model import GPT2Manager

import dataset.Dataset_dataloader as MODEL
import importlib
importlib.reload(ds_prep)

import inspect
print(inspect.getsource(ds_prep.oasst1_df))

IGNORE_INDEX = -100


EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
GRADIENT_CLIP = 1.0

df_all = ds_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})



conversation =         ["""<|user|> Give three tips for staying healthy.
<|assistant|> 1. Eat a balanced and nutritious diet...
2. Engage in regular physical activity...
3. Get enough sleep...
<|endoftext|>""","""<|user|> What about mental health?
<|assistant|> Mental health is equally important! Practice mindfulness, maintain social connections, and seek help when needed.
<|endoftext|>"""
    ]


dataloader = DL.create_correct_dataloader(
    tokenizer=tokenizer,
    texts=
    # conversation,
    df_all["text"].tolist(),
    batch_size=2,           # ← batch_size передается сюда
    max_length=512,          # Для теста, лучше увеличить до 512
    stride=256,              # Для теста, лучше увеличить до 256
    shuffle=True
)


def inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx):
    """Детальная проверка сэмпла"""

    input_ids = batch['input_ids'][sample_idx]
    labels = batch['labels'][sample_idx]
    attention_mask = batch['attention_mask'][sample_idx]

    valid_len = attention_mask.sum().item()

    print(f"\n{'=' * 70}")
    print(f"Batch {batch_idx}, Sample {sample_idx}")
    print(f"{'=' * 70}")

    # Показываем каждый токен
    print(f"{'Pos':<4} {'Token':<25} {'Trainable':<12} {'Token ID':<8}")
    print("-" * 70)

    for i in range(valid_len):
        token_id = input_ids[i].item()
        token = tokenizer.decode([token_id]).replace('\n', '\\n')
        is_trainable = (labels[i] != IGNORE_INDEX).item()

        trainable_mark = "✅ LEARN" if is_trainable else "❌ IGNORE"

        # Особо отмечаем специальные токены
        if token in ["<|user|>", "<|assistant|>", "<|endoftext|>", "\\n"]:
            trainable_mark = f"🔷 {trainable_mark}"

        print(f"{i:<4} {token:<25} {trainable_mark:<12} {token_id:<8}")

        # Показываем где начинается обучение
        if i > 0 and is_trainable and (labels[i - 1] == IGNORE_INDEX).item():
            print(f"     ↑ НАЧАЛО ОБУЧЕНИЯ ↑")

    # Статистика
    total_trainable = (labels[:valid_len] != IGNORE_INDEX).sum().item()
    print(f"\n📊 Statistics:")
    print(f"  Total tokens: {valid_len}")
    print(f"  Trainable tokens: {total_trainable}")
    print(f"  Percentage: {total_trainable / valid_len * 100:.1f}%")


# Проверяем
for batch_idx, batch in enumerate(dataloader):
    print(f"\n{'#' * 70}")
    print(f"BATCH {batch_idx}")
    print(f"{'#' * 70}")

    for sample_idx in range(batch['input_ids'].shape[0]):
        inspect_sample_detailed(batch, tokenizer, batch_idx, sample_idx)

    if batch_idx >= 0:  # Проверяем только первый батч
        break



def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Простой линейный scheduler с warmup"""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



if torch.cuda.is_available():
   device = torch.device("cuda")
elif torch.backends.mps.is_available():
   device = torch.device("mps")
else:
   device = torch.device("cpu")

manager = GPT2Manager()
model = manager.get_model(tokenizer=tokenizer)
model = model.to(device)


loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Подготовка scheduler
total_steps = len(dataloader) * EPOCHS
scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)

loss_history = []
step_history = []
global_step = 0

model.train()
for epoch in range(EPOCHS):
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'=' * 60}")

    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        # Переносим данные на устройство
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # attention_mask используйте если ваша модель его поддерживает
        # attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        logits = model(in_idx=input_ids)  # [batch_size, seq_len, vocab_size]

        # Вычисляем loss
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
            labels.view(-1)  # [batch_size * seq_len]
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (важно для стабильности)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)

        optimizer.step()
        scheduler.step()

        # Статистика
        current_loss = loss.item()
        epoch_loss += current_loss
        global_step += 1

        # Сохраняем историю
        loss_history.append(current_loss)
        step_history.append(global_step)

        # Обновляем прогресс бар
        avg_loss = epoch_loss / (batch_idx + 1)
        current_lr = scheduler.get_last_lr()[0]

        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Периодическая генерация для контроля
        # if batch_idx % 50 == 0 and batch_idx > 0:
        #     model.eval()
        #     with torch.no_grad():
        #         # Берем первый пример из батча
        #         sample_input = input_ids[0:1]  # [1, seq_len]
        #
        #         # Генерация (если у вашей модели есть метод generate)
        #         # Если нет, делаем простую автогрессивную генерацию
        #         generated = generate_text(
        #             model, tokenizer, sample_input,
        #             max_new_tokens=50, temperature=0.7
        #         )
        #
        #         print(f"\n{'=' * 50}")
        #         print(f"Step {global_step} - Sample generation:")
        #         print(f"Input: {tokenizer.decode(sample_input[0], skip_special_tokens=True)[:100]}...")
        #         print(f"Output: {generated}...")
        #         print(f"{'=' * 50}\n")
        #
        #     model.train()

    # Статистика эпохи
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"\n✅ Epoch {epoch + 1} completed!")
    print(f"   Average loss: {avg_epoch_loss:.4f}")
    print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")

    # Сохраняем чекпоинт после каждой эпохи
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, f'checkpoint_epoch_{epoch + 1}.pt')

print("\n" + "=" * 70)
print("🎉 Training completed!")
print("=" * 70)

# 📈 Визуализация обучения
plt.figure(figsize=(10, 6))
plt.plot(step_history, loss_history, alpha=0.6, label='Loss per step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)

# Добавляем сглаженную кривую
window = 20
if len(loss_history) > window:
    smoothed = [sum(loss_history[i:i + window]) / window for i in range(0, len(loss_history) - window, window)]
    plt.plot(range(window, len(loss_history), window)[:len(smoothed)], smoothed, 'r-', linewidth=2, label='Smoothed')

plt.legend()
plt.savefig('training_loss.png', dpi=100)
plt.show()