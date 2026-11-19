from pathlib import Path
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import TrainConfig
from src.dataset import prepare_datasets
from src.model import load_tokenizer_and_lora_model
from src.predict import generate_submission


def run_pipeline(cfg: TrainConfig | None = None) -> Path | None:
    """Execute multi-GPU DDP training, evaluation, and submission generation."""
    cfg = cfg or TrainConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.resolve_model_path()
    raw_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if raw_tokenizer.pad_token is None:
        raw_tokenizer.pad_token = raw_tokenizer.eos_token

    train_dataset, val_dataset, id2label, label2id, text_col = prepare_datasets(
        data_dir=cfg.data_dir,
        tokenizer=raw_tokenizer,
        max_length=cfg.max_length,
        val_size=cfg.val_size,
        seed=cfg.seed,
    )

    tokenizer, model = load_tokenizer_and_lora_model(cfg, id2label, label2id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        bf16=cfg.use_bf16,
        fp16=not cfg.use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if trainer.is_world_process_zero():
        model.print_trainable_parameters()

    trainer.train()

    if trainer.is_world_process_zero():
        adapter_dir = cfg.output_dir / "best_lora_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    return generate_submission(
        trainer=trainer,
        tokenizer=tokenizer,
        cfg=cfg,
        id2label=id2label,
        text_col=text_col,
    )


if __name__ == "__main__":
    run_pipeline()