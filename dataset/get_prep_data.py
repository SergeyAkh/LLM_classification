import os
import math
import pandas as pd
from datasets import load_dataset
from collections import defaultdict


def alpaca_df(config) -> pd.DataFrame:

    path = os.path.join(config.DATA_RAW_PATH, 'alpaca_data_cleaned.json')
    if os.path.exists(path):

        df = pd.read_json(path)
    else:
        os.makedirs(config.DATA_RAW_PATH, exist_ok=True)
        df = pd.read_json(config.alpaca_ds)
        df.to_json(path)


    df['input'] = df['input'].fillna('')
    df['user_text'] = df.apply(
        lambda x: f"<|user|> {x['instruction']}" + (f" {x['input']}" if x['input'] else ""),
        axis=1
    )

    df['full_text'] = (
            df['user_text'] +
            " <|assistant|> " +
            df['output'] +
            " <|endoftext|>"
    )
    df["text"] = df[['full_text']]
    df_new = df.drop(["instruction", "input","user_text","full_text","output"], axis=1)
    return df_new


def oasst1_df(config) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_tr = os.path.join(config.DATA_RAW_PATH, 'oasst1_train.json')
    path_val = os.path.join(config.DATA_RAW_PATH, 'oasst1_val.json')

    cols = ["message_id", "parent_id", "text", "role", "message_tree_id", "rank", "created_date"]

    if (not os.path.exists(path_tr)) or (not os.path.exists(path_val)):
        os.makedirs(config.DATA_RAW_PATH, exist_ok=True)

        ds = load_dataset(config.oasst_ds)

        df_tr = ds["train"].to_pandas()
        df_val = ds["validation"].to_pandas()

        df_tr.to_json(path_tr, orient="records", lines=True)
        df_val.to_json(path_val, orient="records", lines=True)

    else:
        df_tr = pd.read_json(path_tr, lines=True)
        df_val = pd.read_json(path_val, lines=True)

    if not isinstance(df_tr, pd.DataFrame):
        df_tr = df_tr.to_pandas()

    if not isinstance(df_val, pd.DataFrame):
        df_val = df_val.to_pandas()

    cols_tr = [c for c in cols if c in df_tr.columns]
    cols_val = [c for c in cols if c in df_val.columns]

    df_tr = df_tr[cols_tr]
    df_val = df_val[cols_val]

    return df_tr, df_val

def preprop_oasst(preprop_oasst):
    id_to_row = preprop_oasst.set_index("message_id").to_dict("index")
    children = preprop_oasst.groupby("parent_id")["message_id"].apply(list).to_dict()

    all_ids = set(preprop_oasst["message_id"])
    parent_ids = set(preprop_oasst["parent_id"].dropna())

    leaf_ids = list(all_ids - parent_ids)

    def build_thread(leaf_id):
        thread = []
        current = leaf_id

        while not pd.isna(current):
            if current not in id_to_row:
                break  # broken link in data

            row = id_to_row[current]
            thread.append(row)
            current = row["parent_id"]

        thread.reverse()
        return thread

    threads = [build_thread(leaf_id) for leaf_id in leaf_ids]

    def is_valid_thread(thread):
        if len(thread) < 2:
            return False

        # must end with assistant
        if thread[-1].get("role") != "assistant":
            return False

        return True

    threads = [t for t in threads if is_valid_thread(t)]
    threads = [t for t in threads if all(m["text"] for m in t)]
    def threads_to_text_df(threads):
        data = []

        for i, thread in enumerate(threads):
            lines = []

            for msg in thread:
                role = msg.get("role", "user")  # fallback if missing

                if role == "prompter":
                    lines.append(f"<|user|> {msg['text']}")
                elif role == "assistant":
                    lines.append(f"<|assistant|> {msg['text']}")
                else:
                    lines.append(msg["text"])

            if len(lines) >= 2:
                data.append({
                    "id": i,
                    "text": "\n".join(lines)
                })

        return pd.DataFrame(data)

    treads = threads_to_text_df(threads)
    return treads

def get_data_preprocessed(config, split_ratio = 0.9):
    if os.path.exists(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv")) & os.path.exists(os.path.join(config.PREPROC_DS, "Preprocessed_val_data.csv")):
        train = pd.read_csv(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv"))
        val = pd.read_csv(os.path.join(config.PREPROC_DS, "Preprocessed_val_data.csv"))
    else:
        os.makedirs(config.PREPROC_DS, exist_ok=True)

        # alp_df = alpaca_df(config)

        df_tr, df_val = oasst1_df(config)

        oasst_all = pd.concat([df_tr, df_val], ignore_index=True)

        oasst_df = preprop_oasst(oasst_all)

        df_all = pd.concat([oasst_df], ignore_index=True)

        # shuffle
        df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

        split_idx = int(len(df_all) * split_ratio)

        train = df_all[:split_idx]
        val = df_all[split_idx:]
        # save
        train.to_csv(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv"), index=False)
        val.to_csv(os.path.join(config.PREPROC_DS, "Preprocessed_val_data.csv"), index=False)

    return train, val
