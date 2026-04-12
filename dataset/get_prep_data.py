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

def is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def build_tree(tree_df):
    nodes = {}
    children = defaultdict(list)

    for _, row in tree_df.iterrows():
        node = {
            "id": row["message_id"],
            "parent_id": row["parent_id"],
            "text": row["text"],
            "role": row["role"]
        }
        nodes[node["id"]] = node
        children[node["parent_id"]].append(node["id"])

    # attach children
    for parent_id, child_ids in children.items():
        if parent_id in nodes:
            nodes[parent_id]["children"] = child_ids

    return nodes

# =========================
# 3. Get leaf nodes
# =========================
def get_leaf_nodes(nodes):
    return [
        node_id for node_id, node in nodes.items()
        if "children" not in node or len(node["children"]) == 0
    ]

# =========================
# 4. Build full path (root → leaf)
# =========================
def build_path(nodes, node_id):
    path = []

    while node_id in nodes:
        node = nodes[node_id]
        path.append((node["role"], node["text"]))
        node_id = node["parent_id"]

    return list(reversed(path))

# =========================
# 5. Format for GPT training
# =========================
def format_conversation(path):
    text = ""

    for role, content in path:
        if not content.strip():
            continue

        prefix = "<|user|>" if role == "prompter" else "<|assistant|>"
        text += f"{prefix} {content}\n"

    return text.strip() + " <|endoftext|>"

# =========================
# 6. Filter bad data (optional but recommended)
# =========================
def is_valid_path(path):
    if len(path) < 2:
        return False
    return all(len(text.strip()) > 3 for _, text in path)



def preprop_oasst(df):

    prompters = df[df["role"] == "prompter"]
    assistants_na = df[(df["role"] == "assistant")&(df["rank"].isna())]
    assistants_notna = df[(df["role"] == "assistant")&(~df["rank"].isna())]
    # -----------------------------
    # 2. Keep best assistant per parent
    # -----------------------------

    test = assistants_notna.groupby("parent_id", group_keys=True).apply(lambda x: x.loc[[x["rank"].idxmax(),x["rank"].idxmin()]]).reset_index()
    test = test.sort_values(['parent_id', 'rank'])

    result = (
        test.sort_values(['parent_id', 'rank'])  # keep correct order
            .groupby(['parent_id', 'role', 'message_tree_id'], as_index=False)
            .agg({
                'text': ' '.join,
                'message_id': 'first'
            })
    )

    assistants_na = (
        assistants_na
        .groupby(['parent_id', 'role', 'message_tree_id'], as_index=False)
        .agg({
            'text': ' '.join,
            'message_id': 'first'
        })
    )

    assistant = pd.concat([assistants_na, result], ignore_index=True)

    # -----------------------------
    # 3. Keep only prompts that have answers
    # -----------------------------
    valid_parent_ids = set(assistant["parent_id"])

    prompters = prompters[prompters["message_id"].isin(valid_parent_ids)]
    prompters = (
        prompters.groupby(["message_id","role","message_tree_id"])['text']
        .apply(lambda x: ' '.join(x))  # or use special separator
        .reset_index()
    )
    # -----------------------------
    # 4. Combine back
    # -----------------------------
    df_filtered = pd.concat([prompters, assistant], ignore_index=True)

    # -----------------------------
    # 5. Optional: remove broken chains
    # (keep only nodes whose parent exists OR root)
    # -----------------------------
    valid_ids = set(df_filtered["message_id"])

    df_filtered = df_filtered[
        df_filtered["parent_id"].isin(valid_ids) |
        ~df_filtered["parent_id"].isin(df_filtered["message_id"])
    ]

    df_filtered = df_filtered.reset_index(drop=True)

    # =========================
    # 7. Main pipeline
    # =========================
    all_sequences = []

    for tree_id, tree_df in df_filtered.groupby("message_tree_id"):
        nodes = build_tree(tree_df)
        leaf_nodes = get_leaf_nodes(nodes)

        for leaf in leaf_nodes:
            path = build_path(nodes, leaf)

            if is_valid_path(path):
                formatted = format_conversation(path)
                all_sequences.append(formatted)

    oasst_flat = pd.DataFrame({'text': all_sequences})
    return oasst_flat

def get_data_preprocessed(config) -> pd.DataFrame:
    if os.path.exists(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv")):
        df_all = pd.read_csv(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv"))
    else:
        os.makedirs(config.PREPROC_DS, exist_ok=True)

        alp_df = alpaca_df(config)

        df_tr, df_val = oasst1_df(config)

        oasst_all = pd.concat([df_tr, df_val], ignore_index=True)

        oasst_df = preprop_oasst(oasst_all)

        df_all = pd.concat([alp_df, oasst_df], ignore_index=True)

        # shuffle
        df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

        # save
        df_all.to_csv(os.path.join(config.PREPROC_DS, "Preprocessed_data.csv"), index=False)

    return df_all