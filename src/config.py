from dataclasses import dataclass
from pathlib import Path
import os
import torch


@dataclass
class TrainConfig:
    data_dir: Path = Path("/kaggle/input/competitions/name-that-llm")
    output_dir: Path = Path("/kaggle/working/outputs")
    submission_path: Path = Path("/kaggle/working/submission.parquet")
    model_path: str | None = None

    # Sequence & LoRA hyperparameters
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training hyperparameters (per_device_batch_size=2 x 2 GPUs x 4 accum = 16 effective)
    val_size: float = 0.1
    seed: int = 42
    learning_rate: float = 2e-4
    train_batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    weight_decay: float = 0.01
    logging_steps: int = 20

    def resolve_model_path(self) -> str:
        """Locate the attached Hugging Face Gemma 3 directory in /kaggle/input."""
        if self.model_path and Path(self.model_path).exists():
            return self.model_path

        for root, _, files in os.walk("/kaggle/input"):
            if "config.json" in files and "model.safetensors.index.json" in files:
                self.model_path = root
                return root

        raise FileNotFoundError(
            "Could not find Gemma 3 Hugging Face files (config.json) under /kaggle/input."
        )

    @property
    def use_bf16(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    @property
    def dtype(self) -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if self.use_bf16 else torch.float16