# 3位解法: テーブルデータをテキスト情報に変換して日本語ModernBERTを学習・推論

投稿者: takaito  
最終更新: 2025/08/04

---

## アプローチ概要

カテゴリデータが非常に多いテーブルデータを**すべてテキスト（プロンプト）に変換**し、自然言語処理モデルの入力として扱う手法。細かい特徴量作成の時間を大幅に削減できる。

---

## テキスト（プロンプト）の例

```
コンテンツ・サービス・デザイン
2022年
非管理職(一般)
平均残業時間: 約30時間
最大残業時間: 約40時間
リテラシー_DX基礎(【DX基礎研修】DX概論)
リテラシー_DX基礎(【DX基礎研修】ディスラプター①)
2023年
非管理職(一般)
平均残業時間: 約40時間
最大残業時間: 約60時間
DX講演会(「デジタル革命の先にある新しい社会」)
2024年
非管理職(一般)
平均残業時間: 約40時間
最大残業時間: 約60時間
```

---

## 使用データ

自前で作成が必要なCSVファイル:

| ファイル名 | カラム |
|---|---|
| `original_train.csv` | 社員番号, category, prompt, labels |
| `original_test.csv` | 社員番号, category, prompt |

- `labels` は `target` カラムのリネーム
- `prompt` に上記のようなテキスト情報を格納

---

## 使用モデル

- **日本語ModernBERT** (SB Intuitions公開)
  - 共有コード: `sbintuitions/modernbert-ja-70m`（速度重視）
  - 自身の実験: `sbintuitions/modernbert-ja-310m`（精度重視、GPUメモリに余裕があれば推奨）
  - 最大8192トークンまで対応
- ModernBERTはBERT/RoBERTaなどのエンコーダ型Transformerを改善したモデルで、高速・高性能が特長

---

## 学習設定

| パラメータ | 値 |
|---|---|
| n_splits | 5 |
| max_length | 1024 |
| optimizer | adamw_torch |
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 4 |
| per_device_eval_batch_size | 8 |
| n_epochs | 10 |
| learning_rate | 2e-4 |
| warmup_steps | 20 |
| weight_decay | 0.01 |
| lr_scheduler_type | cosine |
| precision | bf16 |
| metric_for_best_model | AUC |

- **CV戦略**: GroupKFold（社員番号でグループ化、リーク防止）
- **評価指標**: AUC + Accuracy

---

## 学習・推論コード（全体）

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

@dataclass
class Config:
    ver = 0
    n_splits = 5
    output_dir: str = "output"
    model_name: str = 'modernbert-ja-70m'
    checkpoint: str = "sbintuitions/modernbert-ja-70m"
    max_length: int = 1024
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 8
    n_epochs: int = 10
    lr: float = 2e-4
    warmup_steps: int = 20
    seed = 2025

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        tokenized = self.tokenizer(batch["prompt"], max_length=self.max_length, truncation=True)
        return {**tokenized, "labels": batch["labels"]}

class TestCustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        tokenized = self.tokenizer(batch["prompt"], max_length=self.max_length, truncation=True)
        return {**tokenized}

encode = CustomTokenizer(tokenizer, max_length=config.max_length)
test_encode = TestCustomTokenizer(tokenizer, max_length=config.max_length)

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(logits).float().softmax(-1).numpy()[:, 1]
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(y_true=labels, y_pred=probs > 0.5)
    return {"auc": auc, "acc": acc}

train_df = pd.read_csv('./data/original_train.csv')
test_df = pd.read_csv('./data/original_test.csv')
submission_df = pd.read_csv('./data/sample_submission.csv')

test_ds = Dataset.from_pandas(test_df[['prompt']])
test_ds = test_ds.map(test_encode, batched=True)

folds = np.zeros(len(train_df))
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
kfold = GroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

for fold, (train_index, valid_index) in enumerate(kfold.split(train_df, train_df['labels'], train_df['社員番号'])):
    train = train_df.iloc[train_index]
    valid = train_df.iloc[valid_index]
    folds[valid_index] = fold + 1

    steps = 2 * config.n_splits * int(
        len(train) / 4 * (config.n_splits - 1) / config.n_splits
        / config.per_device_train_batch_size / config.gradient_accumulation_steps
    )
    training_args = TrainingArguments(
        output_dir=f'./models/{config.model_name}_VER{config.ver}_fold{fold+1}',
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy="steps",
        logging_steps=steps,
        do_eval=True,
        eval_steps=steps,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=steps,
        optim=config.optim_type,
        bf16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        metric_for_best_model="auc",
        greater_is_better=True,
        seed=config.seed,
        data_seed=config.seed,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    train_ds = Dataset.from_pandas(train[['prompt', 'labels']])
    valid_ds = Dataset.from_pandas(valid[['prompt', 'labels']])
    train_ds = train_ds.map(encode, batched=True)
    valid_ds = valid_ds.map(encode, batched=True)

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()

    logits = trainer.predict(valid_ds).predictions
    probs = torch.from_numpy(logits).float().softmax(-1).numpy()[:, 1]
    oof_preds[valid_index] = probs

    logits = trainer.predict(test_ds).predictions
    probs = torch.from_numpy(logits).float().softmax(-1).numpy()[:, 1]
    test_preds += probs / config.n_splits
```

---

## Tips・工夫点

### tips① 入力トークン長
- 日本語ModernBERTは最大**8192トークン**まで対応
- 系列長が長いほどメモリ消費が増えるため、勾配ブースティングで有効だった特徴量を参考に情報の取捨選択を推奨

### tips② 精度と速度のトレードオフ
- 速度重視: `bf16=True` / `torch_dtype=torch.bfloat16`
- 精度重視: `bf16=False` / `torch_dtype=torch.float32`
- 推奨戦略: まずbf16でハイパーパラメータ・入力特徴量の当たりをつけ、最終的にfloat32へ切り替える
- bfloat16で精度劣化が気になる場合は Kahan Summation（[optimi](https://github.com/warner-benjamin/optimi)）も選択肢

### tips③ モデルサイズ
- `modernbert-ja-70m`: 速度重視の初期実験向け
- `modernbert-ja-310m`: GPUメモリに余裕があれば推奨。十分高速にフルチューニング可能
- LLMのLoRAと異なり、BERTは**全パラメータのフルチューニング**が高速にできる点が短期コンペで有利

---

## スコア

- Private: **0.7134**（テキスト化パターンを変更したバージョン、最終サブのアンサンブルには未使用）

---

## まとめ

テーブルデータをテキスト変換してBERT系モデルに入力するアプローチは、複雑な特徴量エンジニアリングを省略しつつ高スコアを実現できる。短期コンペでのトライ&エラーのサイクルを高頻度で回す戦略として有効。
