import urllib.request
import zipfile
import os
import ssl
from pathlib import Path
from srs.LLM_classification.config import Config
import pandas as pd
from datasets import load_dataset
import importlib
import srs.LLM_classification.config as cfg

importlib.reload(cfg)

def alpaca_df(Config) -> pd.DataFrame:
    path = os.path.join(Config.DATA_RAW_PATH, 'alpaca_data_cleaned.json')
    if os.path.exists(path):
        df = pd.read_json(path)
    else:
        df = pd.read_json(Config.alpaca_ds)
        df.to_json(path)
    return df

alp_df = alpaca_df(Config)




oasst1_df = load_dataset("OpenAssistant/oasst1")

oasst1_train = pd.DataFrame(oasst1_df["train"])



df_filtered = oasst1_train[oasst1_train['role'].isin(['prompter', 'assistant'])].copy()
df_filtered = df_filtered.sort_values(['message_id', 'created_date'])
pairs = []

for tree_id, tree_group in df_filtered.groupby('message_id'):
    print(tree_group)

for tree_id, tree_group in df_filtered.groupby('message_id'):
    # Создаем словарь для быстрого доступа
    messages_dict = {}
    for _, row in tree_group.iterrows():
        messages_dict[row['message_id']] = row

print(messages_dict)










def create_user_assistant_pairs(df):
    """
    Создает пары (user message, assistant response)
    с учетом иерархии диалогов
    """
    # Оставляем только сообщения от пользователя и ассистента
    df_filtered = df[df['role'].isin(['prompter', 'assistant'])].copy()

    # Сортируем по времени
    df_filtered = df_filtered.sort_values(['message_id', 'created_date'])

    # Создаем список пар
    pairs = []

    # Группируем по деревьям диалогов
    for tree_id, tree_group in df_filtered.groupby('message_id'):
        # Создаем словарь для быстрого доступа
        messages_dict = {}
        for _, row in tree_group.iterrows():
            messages_dict[row['message_id']] = row

        # Ищем пары (ответ ассистента → его родительское сообщение)
        for _, row in tree_group.iterrows():
            if row['role'] == 'assistant' and pd.notna(row['parent_id']):
                parent_id = row['parent_id']

                # Проверяем, что родитель существует и это сообщение пользователя
                if parent_id in messages_dict:
                    parent_msg = messages_dict[parent_id]

                    # Проверяем, что родитель - пользователь (или можно включать другие роли)
                    if parent_msg['role'] == 'prompter':
                        pairs.append({
                            'prompt': parent_msg['text'],
                            'response': row['text'],
                            'message_id': tree_id,
                            'prompt_id': parent_id,
                            'response_id': row['message_id']
                        })

    return pd.DataFrame(pairs)


# Создаем датасет пар
pairs_df = create_user_assistant_pairs(oasst1_train)
print(f"Создано {len(pairs_df)} пар пользователь-ассистент")
print(f"Уникальных деревьев: {pairs_df['message_id'].nunique()}")
print(f"\nПример пары:")
print(f"Пользователь: {pairs_df.iloc[0]['prompt'][:200]}...")
print(f"Ассистент: {pairs_df.iloc[0]['response'][:200]}...")