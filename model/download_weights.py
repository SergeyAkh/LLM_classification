import os
import requests  # Make sure requests is installed
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import torch
from srs.LLM_classification.config import Config
from Load_model import GPT2Loader

path = Config.GPT_WEIGHTS_PATH
MODEL_TYPE = Config.MODEL_TYPE
BASE_CONFIG = Config.BASE_CONFIG
model_configs = Config.MODEL_CONFIG
BASE_CONFIG.update(model_configs[MODEL_TYPE])

model_size = MODEL_TYPE.split(" ")[-1].lstrip("(").rstrip(")")

loader = GPT2Loader(model_size=model_size, models_dir=path)
settings, params = loader.download_and_load()  # downloads & loads model

# gpt is your PyTorch GPT-2 model
# loader.load_weights_into_gpt(gpt)  # assigns weights