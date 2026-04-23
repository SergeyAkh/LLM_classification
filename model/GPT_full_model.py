from os import abort
from srs.LLM_classification.config import Config
from model.Load_model import GPT2Loader
from model.GPT_manual_architecture import GPTModel, MultiHeadAttention, FeedForward
from model.LoRA import LoRALinear
import torch.nn as nn

# import importlib
# import model.GPT_manual_architecture as hf
# importlib.reload(hf)
class GPT2Manager:
    """
    Handles GPT-2 model initialization:
    - Reads config
    - Downloads & loads weights
    - Builds PyTorch GPT model
    """

    def __init__(self, use_lora=True, r=8, alpha = 1, dropout=0.05):
        # Load configuration
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.use_lora = use_lora
        self.path = Config.GPT_WEIGHTS_PATH
        self.model_type = Config.MODEL_TYPE
        self.base_config = Config.BASE_CONFIG
        self.model_configs = Config.MODEL_CONFIG
        self.base_config.update(self.model_configs[self.model_type])

        # Extract model size (e.g., "124M", "355M")
        self.model_size = self.model_type.split(" ")[-1].lstrip("(").rstrip(")")

        # Initialize placeholders
        self.loader = None
        self.settings = None
        self.params = None
        self.model = None

    def resize_token_embeddings(self, new_vocab_size):
        old_vocab_size, emb_dim = self.model.tok_emb.weight.shape

        if new_vocab_size <= old_vocab_size:
            return

        device = self.model.tok_emb.weight.device

        # --- tok_emb ---
        new_tok_emb = nn.Embedding(new_vocab_size, emb_dim).to(device)
        new_tok_emb.weight.data[:old_vocab_size] = self.model.tok_emb.weight.data
        nn.init.normal_(new_tok_emb.weight.data[old_vocab_size:], mean=0.0, std=0.02)

        self.model.tok_emb = new_tok_emb

        # --- out_head ---
        new_out_head = nn.Linear(emb_dim, new_vocab_size, bias=False).to(device)
        new_out_head.weight.data[:old_vocab_size] = self.model.out_head.weight.data
        nn.init.normal_(new_out_head.weight.data[old_vocab_size:], mean=0.0, std=0.02)

        self.model.out_head = new_out_head

        # --- weight tying ---
        self.model.out_head.weight = self.model.tok_emb.weight

    def freeze_except_lora(self, model):
        for name, param in model.named_parameters():
            if "A" in name or "B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def apply_lora(self, model, enabled=True):

        for module in model.modules():
            if isinstance(module, MultiHeadAttention):
                module.W_query = LoRALinear(
                    module.W_query, r=self.r, alpha=self.alpha, dropout=self.dropout
                )
                module.W_query.enabled = enabled

                module.W_value = LoRALinear(
                    module.W_value, r=self.r, alpha=self.alpha, dropout=self.dropout
                )
                module.W_value.enabled = enabled

                module.W_key = LoRALinear(
                    module.W_key, r=self.r, alpha=self.alpha, dropout=self.dropout
                )
                module.W_key.enabled = enabled

                module.out_proj = LoRALinear(
                    module.out_proj, r=self.r, alpha=self.alpha, dropout=self.dropout
                )
                module.out_proj.enabled = enabled

            elif isinstance(module, FeedForward):
                for i, layer in enumerate(module.layers):
                    if isinstance(layer, nn.Linear):
                        module.layers[i] = LoRALinear(layer, r=self.r, alpha=self.alpha, dropout=self.dropout)
                        module.layers[i].enabled = enabled

    def prepare_model(self, tokenizer, inference = None):
        self.loader = GPT2Loader(model_size=self.model_size, models_dir=self.path)

        self.settings, self.params = self.loader.download_and_load()

        self.model = GPTModel(self.base_config)

        if inference is None:

            self.loader.load_weights_into_gpt(self.model)

        if self.use_lora:
            self.apply_lora(self.model, enabled=True)
            self.freeze_except_lora(self.model)

        # 🔥 New tokens
        if tokenizer is not None:

            new_vocab_size = len(tokenizer)

            self.resize_token_embeddings(new_vocab_size)
            if self.use_lora:
                self.freeze_except_lora(self.model)
            # self.base_config["vocab_size"] = new_vocab_size
        else:
            abort()
        return self.model

    def get_model(self, tokenizer=None, inference = None):
        """
        Returns the model; prepare it if not yet loaded.
        """

        if self.model is None and inference is None:
            return self.prepare_model(tokenizer)
        else:
            return self.prepare_model(tokenizer, inference)