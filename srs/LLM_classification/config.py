from pathlib import Path
import sys

# Определяем PROJECT_ROOT для разных сред выполнения
if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
    # Интерактивная консоль или Jupyter
    PROJECT_ROOT = Path.cwd().parent
elif '__file__' in globals():
    # Запуск как скрипт
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    # Fallback
    PROJECT_ROOT = Path.cwd().parent

class Config:
    # Class attributes (доступны через Config.ATTR)
    DEBUG = True
    APP_NAME = "My App"

    # dataset_server
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"

    alpaca_ds = "hf://datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json"
    oasst_ds = "OpenAssistant/oasst1"
    PROJECT_ROOT = PROJECT_ROOT
    # paths - используем PROJECT_ROOT для полных путей
    DATA_RAW_PATH = PROJECT_ROOT / "dataset" / "data" / "raw"
    MODEL_PATH = PROJECT_ROOT / "weights" / "model.pt"

    # model params
    GPT_WEIGHTS_PATH = PROJECT_ROOT / "gpt_trained_weights"
    MODEL_NAME = "gpt2"
    MODEL_TYPE = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": True
    }

    MODEL_CONFIG = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    @classmethod
    def get_model_config(cls):

        config = cls.BASE_CONFIG.copy()
        config.update(cls.MODEL_CONFIG[cls.MODEL_TYPE])
        return config
