from pathlib import Path
import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase, Trainer

from src.config import TrainConfig
from src.dataset import LLMNameDataset


def generate_submission(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    cfg: TrainConfig,
    id2label: dict[int, str],
    text_col: str,
) -> Path | None:
    """Run distributed inference on test.csv and write submission.parquet on rank 0."""
    test_df = pd.read_csv(cfg.data_dir / "test.csv")
    test_dataset = LLMNameDataset(
        df=test_df,
        tokenizer=tokenizer,
        text_col=text_col,
        max_length=cfg.max_length,
        label2id=None,
    )

    preds = trainer.predict(test_dataset)

    if trainer.is_world_process_zero():
        probs = torch.softmax(torch.tensor(preds.predictions, dtype=torch.float32), dim=-1).numpy()
        sub = pd.read_parquet(cfg.data_dir / "sample_submission.parquet")
        for idx, label_name in id2label.items():
            if label_name in sub.columns:
                sub[label_name] = probs[:, idx]

        sub.to_parquet(cfg.submission_path, index=False)
        print(f"Saved {cfg.submission_path} with shape {sub.shape}")
        return cfg.submission_path

    return None