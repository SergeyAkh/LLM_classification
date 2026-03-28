import urllib.request
import zipfile
import os
import ssl
from pathlib import Path
import math
from srs.LLM_classification.config import Config
import pandas as pd
from datasets import load_dataset
import importlib
import srs.LLM_classification.config as cfg
from collections import defaultdict

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

def oasst1_df(Config) -> pd.DataFrame:
    path_tr = os.path.join(Config.DATA_RAW_PATH, 'oasst1_train.json')
    path_val = os.path.join(Config.DATA_RAW_PATH, 'oasst1_val.json')
    if os.path.exists(path_val) & os.path.exists(path_tr):
        df_tr = pd.read_json(path_tr, lines=True)
        df_val = pd.read_json(path_val, lines=True)

    else:
        oasst1_df = load_dataset(Config.oasst_ds)
        df_tr = oasst1_df["train"]
        df_val = oasst1_df["validation"]
        oasst1_df["train"].to_json(os.path.join(Config.DATA_RAW_PATH, 'oasst1_train.json'))
        oasst1_df["validation"].to_json(os.path.join(Config.DATA_RAW_PATH, 'oasst1_val.json'))
    df_tr = df_tr[["message_id", "parent_id", "text", "role"]]
    df_val = df_val[["message_id", "parent_id", "text", "role"]]

    return df_tr, df_val

def is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def build_conv(node, path):
    path = path + [node]

    if node["message_id"] not in children:
        return [path]

    convs = []
    for child in children[node["message_id"]]:
        convs.extend(build_conv(child, path))

    return convs


df_tr, df_val = oasst1_df(Config)

children = defaultdict(list)

for _, row in df_tr.iterrows():
    children[row["parent_id"]].append(row)


roots = [row for _, row in df_tr.iterrows() if is_nan(row["parent_id"])]

all_convs = []

for root in roots:
    all_convs.extend(build_conv(root, []))


def format_conv(conv):
    text = ""

    for msg in conv:
        role = msg["role"]
        content = msg["text"].strip()

        if role == "prompter":
            text += f"<|user|> {content}\n"
        elif role == "assistant":
            text += f"<|assistant|> {content}\n"

    return text

texts = [format_conv(conv) for conv in all_convs]

for i in children.keys():
    print(i)
