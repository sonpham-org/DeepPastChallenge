#!/usr/bin/env python3
"""
Local training script for ByT5-base on RTX 4090 (24GB VRAM).
Trains Akkadian → English translation model using all available data sources.

Usage:
    python train_local.py                    # Full training (all data)
    python train_local.py --stage1-only      # Only stage 1 (general Akkadian)
    python train_local.py --stage2-only      # Only stage 2 (OA specialization)
    python train_local.py --epochs 20        # Custom epoch count
    python train_local.py --resume ckpt/     # Resume from checkpoint
"""

import argparse
import os
import re
import math
import json
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = BASE_DIR / "external_data"

DEFAULT_CONFIG = {
    "model_name": "google/byt5-base",
    "output_dir": str(BASE_DIR / "models" / "byt5-base-local"),
    "max_source_length": 512,
    "max_target_length": 512,
    # Training hyperparameters (optimized for RTX 4090 24GB)
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 8,  # effective batch = 32
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "warmup_steps": 2000,
    "num_train_epochs_stage1": 5,
    "num_train_epochs_stage2": 15,
    "label_smoothing_factor": 0.1,
    "fp16": True,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 50,
    "save_total_limit": 10,
    "seed": 42,
    "bidirectional": True,
    "dropout": 0.15,
}


# ============================================================
# Data Loading
# ============================================================
def preprocess_transliteration(text):
    """Normalize transliteration text."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"(\.{3,}|…+|……)", "<big_gap>", text)
    text = re.sub(r"(xx+|\s+x\s+)", "<gap>", text)
    return text


def load_competition_data():
    """Load competition training data (1561 document-level pairs)."""
    path = DATA_DIR / "train.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return pd.DataFrame(columns=["source", "target"])
    df = pd.read_csv(path)
    pairs = []
    for _, row in df.iterrows():
        src = preprocess_transliteration(row.get("transliteration", ""))
        tgt = str(row.get("translation", "")).strip()
        if src and tgt:
            pairs.append({"source": src, "target": tgt})
    print(f"  Competition data: {len(pairs)} pairs")
    return pd.DataFrame(pairs)


def load_sentence_data():
    """Load sentence-level aligned data from Sentences_Oare."""
    path = DATA_DIR / "Sentences_Oare_FirstWord_LinNum.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return pd.DataFrame(columns=["source", "target"])
    df = pd.read_csv(path)
    pairs = []
    for _, row in df.iterrows():
        # Sentence data has translation but not full transliteration
        # We use it for the translations and match by text_uuid to train.csv
        tgt = str(row.get("translation", "")).strip()
        if tgt and len(tgt) > 5:
            # Use first_word_spelling as partial source hint
            src_hint = str(row.get("first_word_spelling", "")).strip()
            if src_hint:
                pairs.append({"source": src_hint, "target": tgt})
    print(f"  Sentence-level data: {len(pairs)} pairs (partial source)")
    return pd.DataFrame(pairs)


def load_akkademia_data():
    """Load Akkademia NMT parallel data (50K pairs)."""
    tr_path = EXTERNAL_DIR / "akkademia" / "NMT_input" / "train.tr"
    en_path = EXTERNAL_DIR / "akkademia" / "NMT_input" / "train.en"
    if not tr_path.exists() or not en_path.exists():
        print(f"  WARNING: Akkademia data not found, skipping")
        return pd.DataFrame(columns=["source", "target"])
    with open(tr_path, "r", encoding="utf-8") as f:
        sources = f.readlines()
    with open(en_path, "r", encoding="utf-8") as f:
        targets = f.readlines()
    pairs = []
    for src, tgt in zip(sources, targets):
        src = preprocess_transliteration(src.strip())
        tgt = tgt.strip()
        if src and tgt and len(src) > 3 and len(tgt) > 3:
            pairs.append({"source": src, "target": tgt})
    print(f"  Akkademia data: {len(pairs)} pairs")
    return pd.DataFrame(pairs)


def load_oracc_data():
    """Load ORACC parallel data (2117 pairs)."""
    path = EXTERNAL_DIR / "oracc" / "train.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return pd.DataFrame(columns=["source", "target"])
    df = pd.read_csv(path)
    pairs = []
    for _, row in df.iterrows():
        src = preprocess_transliteration(row.get("akkadian", ""))
        tgt = str(row.get("english", "")).strip()
        if src and tgt and len(src) > 3 and len(tgt) > 3:
            pairs.append({"source": src, "target": tgt})
    print(f"  ORACC data: {len(pairs)} pairs")
    return pd.DataFrame(pairs)


def add_bidirectional(df):
    """Add reverse direction (English → Akkadian) pairs."""
    reverse = df.copy()
    reverse["source"], reverse["target"] = df["target"].values, df["source"].values
    reverse["direction"] = "en2akk"
    df["direction"] = "akk2en"
    combined = pd.concat([df, reverse], ignore_index=True)
    print(f"  After bidirectional: {len(combined)} pairs")
    return combined


def prepare_all_data(config, stage="all"):
    """Load and combine all data sources."""
    print("\nLoading data sources:")

    if stage == "stage2":
        # Stage 2: only competition data (Old Assyrian specialization)
        comp_df = load_competition_data()
        if config["bidirectional"]:
            comp_df = add_bidirectional(comp_df)
        return comp_df

    # Stage 1: all available data
    dfs = []
    comp_df = load_competition_data()
    if len(comp_df) > 0:
        # Upsample competition data 3x in stage1 (it's the most relevant)
        dfs.extend([comp_df] * 3)

    akk_df = load_akkademia_data()
    if len(akk_df) > 0:
        dfs.append(akk_df)

    oracc_df = load_oracc_data()
    if len(oracc_df) > 0:
        dfs.append(oracc_df)

    if not dfs:
        raise ValueError("No training data found! Check data directories.")

    combined = pd.concat(dfs, ignore_index=True)

    if config["bidirectional"]:
        combined = add_bidirectional(combined)

    # Shuffle
    combined = combined.sample(frac=1, random_state=config["seed"]).reset_index(drop=True)
    print(f"\nTotal training pairs: {len(combined)}")
    return combined


# ============================================================
# Dataset
# ============================================================
class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.sources = []
        self.targets = []
        for _, row in df.iterrows():
            direction = row.get("direction", "akk2en")
            if direction == "en2akk":
                prefix = "translate English to Akkadian: "
            else:
                prefix = "translate Akkadian to English: "
            self.sources.append(prefix + str(row["source"]))
            self.targets.append(str(row["target"]))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
        )

        return {
            "input_ids": source_encoding["input_ids"],
            "attention_mask": source_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }


# ============================================================
# Evaluation
# ============================================================
def create_compute_metrics(tokenizer):
    """Create metrics computation function."""
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        chrf_result = chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            word_order=2,
        )

        bleu_score = bleu_result["score"]
        chrf_score = chrf_result["score"]
        combined = math.sqrt(max(bleu_score, 0.01) * max(chrf_score, 0.01))

        return {
            "bleu": bleu_score,
            "chrfpp": chrf_score,
            "combined": combined,
        }

    return compute_metrics


# ============================================================
# Training
# ============================================================
def train(config, stage="all"):
    """Run training."""
    print(f"\n{'='*60}")
    print(f"  ByT5-base Local Training - {stage.upper()}")
    print(f"{'='*60}")
    print(f"Config: {json.dumps({k: str(v) for k, v in config.items()}, indent=2)}")

    # Load tokenizer and model
    print(f"\nLoading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    # Set dropout
    if config["dropout"] != 0.1:
        for module in model.modules():
            if hasattr(module, "dropout"):
                if hasattr(module.dropout, "p"):
                    module.dropout.p = config["dropout"]

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # Load data
    train_df = prepare_all_data(config, stage=stage)

    # Split 5% for validation
    val_size = min(500, int(len(train_df) * 0.05))
    val_df = train_df.sample(n=val_size, random_state=config["seed"])
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = TranslationDataset(
        train_df, tokenizer, config["max_source_length"], config["max_target_length"]
    )
    val_dataset = TranslationDataset(
        val_df, tokenizer, config["max_source_length"], config["max_target_length"]
    )

    # Determine epochs
    if stage == "stage1":
        num_epochs = config["num_train_epochs_stage1"]
    elif stage == "stage2":
        num_epochs = config["num_train_epochs_stage2"]
    else:
        num_epochs = config["num_train_epochs_stage1"]

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"{stage}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        label_smoothing_factor=config["label_smoothing_factor"],
        fp16=config["fp16"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=config["max_target_length"],
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="combined",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=4,
        seed=config["seed"],
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=config["max_source_length"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=create_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    print(f"\nStarting training ({num_epochs} epochs)...")
    if config.get("resume_from"):
        trainer.train(resume_from_checkpoint=config["resume_from"])
    else:
        trainer.train()

    # Save best model
    best_dir = os.path.join(output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"\nBest model saved to: {best_dir}")

    return best_dir


def two_stage_training(config):
    """Run two-stage training pipeline."""
    # Stage 1: General Akkadian (all data)
    print("\n" + "=" * 60)
    print("  STAGE 1: General Akkadian Training")
    print("=" * 60)
    stage1_model_path = train(config, stage="stage1")

    # Stage 2: Old Assyrian specialization (competition data only)
    print("\n" + "=" * 60)
    print("  STAGE 2: Old Assyrian Specialization")
    print("=" * 60)
    config_stage2 = config.copy()
    config_stage2["model_name"] = stage1_model_path
    config_stage2["learning_rate"] = 3e-5  # Lower LR for fine-tuning
    config_stage2["warmup_steps"] = 500
    stage2_model_path = train(config_stage2, stage="stage2")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Stage 1 model: {stage1_model_path}")
    print(f"  Stage 2 model: {stage2_model_path}")
    print(f"{'='*60}")
    return stage2_model_path


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ByT5-base local training for Akkadian translation")
    parser.add_argument("--stage1-only", action="store_true", help="Only run stage 1")
    parser.add_argument("--stage2-only", action="store_true", help="Only run stage 2")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint directory")
    parser.add_argument("--model", type=str, default="google/byt5-base", help="Base model name/path")
    parser.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model
    config["per_device_train_batch_size"] = args.batch_size
    config["gradient_accumulation_steps"] = args.grad_accum
    config["bidirectional"] = not args.no_bidirectional
    config["seed"] = args.seed

    if args.epochs:
        config["num_train_epochs_stage1"] = args.epochs
        config["num_train_epochs_stage2"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    if args.resume:
        config["resume_from"] = args.resume
    if args.output_dir:
        config["output_dir"] = args.output_dir

    if args.stage1_only:
        train(config, stage="stage1")
    elif args.stage2_only:
        train(config, stage="stage2")
    else:
        two_stage_training(config)


if __name__ == "__main__":
    main()
