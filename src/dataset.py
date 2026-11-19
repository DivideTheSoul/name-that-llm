from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def build_label_maps(train_df: pd.DataFrame) -> tuple[dict[int, str], dict[str, int]]:
    """Create deterministic id2label and label2id dictionaries from training labels."""
    labels = sorted(train_df["model"].unique().tolist())
    id2label = dict(enumerate(labels))
    label2id = {label: idx for idx, label in id2label.items()}
    return id2label, label2id


def detect_text_column(df: pd.DataFrame) -> str:
    """Identify the input text column in train/test DataFrames."""
    for candidate in ("text", "response", "output", "prompt"):
        if candidate in df.columns:
            return candidate
    non_target_cols = [c for c in df.columns if c not in ("id", "model")]
    return non_target_cols[0]


class LLMNameDataset(Dataset):
    """PyTorch Dataset for LLM response classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_col: str,
        max_length: int = 512,
        label2id: dict[str, int] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            str(row[self.text_col]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.label2id is not None:
            item["labels"] = torch.tensor(self.label2id[row["model"]], dtype=torch.long)

        return item


def prepare_datasets(
    data_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[LLMNameDataset, LLMNameDataset, dict[int, str], dict[str, int], str]:
    """Load train.csv, create stratified train/validation splits, and return Datasets."""
    train_df = pd.read_csv(data_dir / "train.csv")  #
    id2label, label2id = build_label_maps(train_df)
    text_col = detect_text_column(train_df)

    train_split, val_split = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["model"],
    )

    train_dataset = LLMNameDataset(train_split, tokenizer, text_col, max_length, label2id)
    val_dataset = LLMNameDataset(val_split, tokenizer, text_col, max_length, label2id)
    return train_dataset, val_dataset, id2label, label2id, text_col