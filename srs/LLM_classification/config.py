from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Config:
    DEBUG = True
    APP_NAME = "My App"

    #dataset_server
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"

    # paths
    DATA_RAW_PATH = "data/raw"
    MODEL_PATH = "weights/model.pt"

    # model params
    GPT_WEIGHTS_PATH = "model/gpt_trained_weights"
    MODEL_NAME = "gpt2"
    MODEL_TYPE = "gpt2-small (124M)"
    BASE_CONFIG  = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
    }
    MODEL_CONFIG = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
