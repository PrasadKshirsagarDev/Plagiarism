#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, random, argparse, math
from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# ----------------------
# Helpers
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # expected keys: text, label
            if "text" in obj and "label" in obj and isinstance(obj["text"], str):
                rows.append({"text": obj["text"].strip(), "label": str(obj["label"]).strip()})
    if not rows:
        raise ValueError(f"No valid rows found in {path}. Need fields: 'text' and 'label'.")
    return rows

def normalize_label(s: str) -> str:
    s = s.strip().lower()
    # map common variants
    if s in ["ai", "machine", "generated", "llm", "bot"]:
        return "AI"
    if s in ["human", "hum", "manav", "मानव", "manually", "writer"]:
        return "HUMAN"
    # default heuristic: exact tokens "ai"/"human" preferred
    return "AI" if "ai" in s else ("HUMAN" if "human" in s or "मानव" in s else s.upper())

def build_label_maps(labels: List[str]):
    uniq = sorted(list({normalize_label(x) for x in labels}))
    # Ensure binary order is consistent if present
    if set(uniq) == {"AI", "HUMAN"}:
        uniq = ["AI", "HUMAN"]
    id2label = {i: lab for i, lab in enumerate(uniq)}
    label2id = {lab: i for i, lab in id2label.items()}
    return label2id, id2label

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

class WeightedTrainer(Trainer):
    """Add class weights to CE loss (helps with class imbalance)."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        )

    # Accept HF's extra arg to stay compatible across versions
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # remove labels so the model doesn't compute its own loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune XLM-R for Hindi AI-vs-HUMAN detection")
    ap.add_argument("--data", type=str, required=True, help="Path to merged_file.jsonl (text,label)")
    ap.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Backbone (e.g., xlm-roberta-base / xlm-roberta-large)")
    ap.add_argument("--output_dir", type=str, default="./hindi_ai_detector_xlmr")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # 1) Load data
    rows = read_jsonl(args.data)
    texts = [r["text"] for r in rows]
    labels_raw = [normalize_label(r["label"]) for r in rows]
    label2id, id2label = build_label_maps(labels_raw)

    # Sanity: must be exactly two classes ideally
    num_labels = len(label2id)
    print(f"Labels: {label2id}  (num_labels={num_labels})")
    y_all = np.array([label2id[l] for l in labels_raw], dtype=np.int64)

    # 2) Train/val/test split (stratified)
    X_tmp, X_test, y_tmp, y_test = train_test_split(texts, y_all, test_size=args.test_size, stratify=y_all, random_state=args.seed)
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=args.seed)

    # 3) Build HF datasets
    def to_list(text_list, y_list):
        return [{"text": t, "label": int(y)} for t, y in zip(text_list, y_list)]
    ds_train = Dataset.from_list(to_list(X_train, y_train))
    ds_val   = Dataset.from_list(to_list(X_val,   y_val))
    ds_test  = Dataset.from_list(to_list(X_test,  y_test))
    dsd = DatasetDict(train=ds_train, validation=ds_val, test=ds_test)

    # 4) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    dsd = dsd.map(tok_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 6) Class weights (for imbalance)
    classes = np.array(sorted(list(set(y_all))))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    print(f"class_weights (train): {class_weights}")

    # 7) Training args
    os.makedirs(args.output_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # 8) Train
    trainer.train()

    # 9) Eval on test set
    test_metrics = trainer.evaluate(dsd["test"])
    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Detailed report
    preds = trainer.predict(dsd["test"])
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n=== Classification report (test) ===")
    print(classification_report(dsd["test"]["label"], y_pred, target_names=[id2label[i] for i in range(num_labels)], digits=4))
    print("Confusion matrix:\n", confusion_matrix(dsd["test"]["label"], y_pred))

    # 10) Save best model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[✓] Saved best model to: {args.output_dir}")

    # 11) Tiny inference demo
    demo = [
        "भारत की अर्थव्यवस्था डिजिटल भुगतान के साथ तेज़ी से बढ़ रही है।",
        "कृपया जलवायु परिवर्तन पर एक औपचारिक अनुच्छेद तैयार करें जो नीति-निर्माताओं को सुझाए।"
    ]
    enc = tokenizer(demo, return_tensors="pt", truncation=True, padding=True, max_length=args.max_length)
    model.eval()
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    print("\n=== Inference demo ===")
    for txt, p in zip(demo, probs):
        top = int(np.argmax(p))
        print(f"PRED: {id2label[top]}  |  probs: { {id2label[i]: float(p[i]) for i in range(len(p))} }")

if __name__ == "__main__":
    main()
