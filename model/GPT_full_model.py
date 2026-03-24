from srs.LLM_classification.config import Config
from Load_model import GPT2Loader
from GPT_manual_architecture import GPTModel


class GPT2Manager:
    """
    Handles GPT-2 model initialization:
    - Reads config
    - Downloads & loads weights
    - Builds PyTorch GPT model
    """

    def __init__(self):
        # Load configuration
        self.path = Config.GPT_WEIGHTS_PATH
        self.model_type = Config.MODEL_TYPE
        self.base_config = Config.BASE_CONFIG.copy()  # avoid mutating original
        self.model_configs = Config.MODEL_CONFIG
        self.base_config.update(self.model_configs[self.model_type])

        # Extract model size (e.g., "124M", "355M")
        self.model_size = self.model_type.split(" ")[-1].lstrip("(").rstrip(")")

        # Initialize placeholders
        self.loader = None
        self.settings = None
        self.params = None
        self.model = None

    def prepare_model(self):
        """
        Download/load GPT-2 weights and build the PyTorch GPT model.
        """
        # Initialize loader
        self.loader = GPT2Loader(model_size=self.model_size, models_dir=self.path)

        # Download and load GPT-2 parameters
        self.settings, self.params = self.loader.download_and_load()

        # Create GPT PyTorch model
        self.model = GPTModel.GPTModel(self.base_config)

        # Assign weights
        self.loader.load_weights_into_gpt(self.model)

        return self.model  # returns the ready-to-use PyTorch model

    def get_model(self):
        """
        Returns the model; prepare it if not yet loaded.
        """
        if self.model is None:
            return self.prepare_model()
        return self.model

manager = GPT2Manager()

# Get the PyTorch GPT model (downloads + loads weights automatically)
augment_model = manager.get_model()