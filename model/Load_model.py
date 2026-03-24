import os
import json
import numpy as np
import torch
import tensorflow as tf
import requests
from tqdm import tqdm

class GPT2Loader:
    ALLOWED_SIZES = ("124M", "355M", "774M", "1558M")

    def __init__(self, model_size="124M", models_dir="gpt2"):
        if model_size not in self.ALLOWED_SIZES:
            raise ValueError(f"Model size not in {self.ALLOWED_SIZES}")
        self.model_size = model_size
        self.models_dir = models_dir
        self.model_dir = os.path.join(models_dir, model_size)
        self.settings = None
        self.params = None

    # ----------------- Public Methods -----------------
    def download_and_load(self):
        self._download_files()
        self._load_params()
        return self.settings, self.params

    def load_weights_into_gpt(self, gpt):
        if self.params is None:
            raise ValueError("Model parameters are not loaded. Call download_and_load() first.")
        self._assign_weights(gpt)

    # ----------------- Internal Methods -----------------
    def _download_files(self):
        os.makedirs(self.model_dir, exist_ok=True)
        base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        filenames = [
            "checkpoint", "encoder.json", "hparams.json",
            "model.ckpt.data-00000-of-00001", "model.ckpt.index",
            "model.ckpt.meta", "vocab.bpe"
        ]
        for filename in filenames:
            url = os.path.join(base_url, self.model_size, filename)
            dest = os.path.join(self.model_dir, filename)
            self._download_file(url, dest)

    def _download_file(self, url, destination):
        try:
            response = requests.get(url, stream=True, verify=False)
            total_size = int(response.headers.get("content-length", 0))

            if os.path.exists(destination) and os.path.getsize(destination) == total_size:
                print(f"File already exists and is up-to-date: {destination}")
                return

            block_size = 1024
            with tqdm(total=total_size, unit="iB", unit_scale=True, desc=os.path.basename(url)) as t:
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(block_size):
                        t.update(len(chunk))
                        f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")

    def _load_params(self):
        tf_ckpt_path = tf.train.latest_checkpoint(self.model_dir)
        self.settings = json.load(open(os.path.join(self.model_dir, "hparams.json")))
        self.params = self._load_gpt2_params_from_tf_ckpt(tf_ckpt_path, self.settings)

    def _load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}
        for name, _ in tf.train.list_variables(ckpt_path):
            array = np.squeeze(tf.train.load_variable(ckpt_path, name))
            parts = name.split("/")[1:]  # skip 'model/'
            target = params
            if parts[0].startswith("h"):
                layer_num = int(parts[0][1:])
                target = params["blocks"][layer_num]
            for key in parts[1:-1]:
                target = target.setdefault(key, {})
            target[parts[-1]] = array
        return params

    def _assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    def _assign_weights(self, gpt):
        p = self.params
        gpt.pos_emb.weight = self._assign(gpt.pos_emb.weight, p['wpe'])
        gpt.tok_emb.weight = self._assign(gpt.tok_emb.weight, p['wte'])

        for b, block in enumerate(p["blocks"]):
            # Attention weights
            q_w, k_w, v_w = np.split(block["attn"]["c_attn"]["w"], 3, axis=-1)
            q_b, k_b, v_b = np.split(block["attn"]["c_attn"]["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = self._assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = self._assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = self._assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
            gpt.trf_blocks[b].att.W_query.bias = self._assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = self._assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = self._assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
            gpt.trf_blocks[b].att.out_proj.weight = self._assign(gpt.trf_blocks[b].att.out_proj.weight, block["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = self._assign(gpt.trf_blocks[b].att.out_proj.bias, block["attn"]["c_proj"]["b"])
            gpt.trf_blocks[b].ff.layers[0].weight = self._assign(gpt.trf_blocks[b].ff.layers[0].weight, block["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = self._assign(gpt.trf_blocks[b].ff.layers[0].bias, block["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = self._assign(gpt.trf_blocks[b].ff.layers[2].weight, block["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = self._assign(gpt.trf_blocks[b].ff.layers[2].bias, block["mlp"]["c_proj"]["b"])
            gpt.trf_blocks[b].norm1.scale = self._assign(gpt.trf_blocks[b].norm1.scale, block["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = self._assign(gpt.trf_blocks[b].norm1.shift, block["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = self._assign(gpt.trf_blocks[b].norm2.scale, block["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = self._assign(gpt.trf_blocks[b].norm2.shift, block["ln_2"]["b"])

        gpt.final_norm.scale = self._assign(gpt.final_norm.scale, p["g"])
        gpt.final_norm.shift = self._assign(gpt.final_norm.shift, p["b"])
        gpt.out_head.weight = self._assign(gpt.out_head.weight, p["wte"])