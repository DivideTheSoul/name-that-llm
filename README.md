# Name That LLM — Gemma-3 LoRA Classifier

Modular PyTorch & Hugging Face PEFT pipeline for classifying LLM-generated responses using **Gemma 3 (4B)** with LoRA sequence classification adapters.

## Architecture
- `src/config.py`: Hardware-aware precision (`bf16`/`fp16`) and offline path discovery.
- `src/dataset.py`: Stratified train/validation splits and tokenization pipeline.
- `src/model.py`: Offline `AutoModelForSequenceClassification` + LoRA adapter initialization.
- `src/train.py`: End-to-end training loop and adapter checkpointing.
- `src/predict.py`: Batched test inference and `submission.parquet` formatting.