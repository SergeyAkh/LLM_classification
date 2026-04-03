# dataset_pipeline.py
from srs.LLM_classification.config import Config
import dataset.Dataset_dataloader as DL
import dataset.get_prep_data as ds_prep
from transformers import GPT2Tokenizer

df_all = ds_prep.get_data_preprocessed(Config)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
})

def dataloader(data, tokenizer_func, batch_size, max_length, stride, shuffle):
    dataloader_func = DL.create_correct_dataloader(
        tokenizer=tokenizer_func,
        texts=data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle
    )
    return dataloader_func

#

#
#
# total_steps = 50
#
# loss_history = []
# step_history = []
# global_step = 0
#


#
#
#     # Сохраняем чекпоинт после каждой эпохи
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': avg_epoch_loss,
#     }, f'checkpoint_epoch_{epoch + 1}.pt')
#
# print("\n" + "=" * 70)
# print("🎉 Training completed!")
# print("=" * 70)
#
# text = generate(
#     model,
#     tokenizer,
#     "<|user|> Give three tips for staying healthy <|assistant|>",
#     device=device,
#     max_new_tokens=150,
#     temperature=0.9,
#     top_k=40,
#     top_p=0.95
# )
# print(text)