import os
import torch
import peft.import_utils
import peft.tuners.lora.torchao
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from src.config import TrainConfig


def patch_peft_torchao() -> None:
    """Bypass PEFT's torchao version check when running on offline Kaggle images."""
    peft.import_utils.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.is_torchao_available = lambda: False


def load_tokenizer_and_lora_model(
    cfg: TrainConfig,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[PreTrainedTokenizerBase, PeftModel]:
    """Load local Gemma 3 checkpoint onto the current DDP rank and wrap with LoRA."""
    patch_peft_torchao()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    model_path = cfg.resolve_model_path()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        dtype=cfg.dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(base_model, lora_config)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    return tokenizer, model