# テーブルデータのテキスト変換 × 日本語ModernBERT アプローチ調査

出典: 本コンペ3位解法（takaito, 2025/08/04, Private: 0.7134）

---

## アプローチ概要

カテゴリデータが多いテーブルデータを**すべてテキスト（プロンプト）に変換**し、日本語ModernBERTの入力として扱う手法。
特徴量エンジニアリングを大幅に省略しながら高スコアを実現できる。

---

## プロンプト例

社員1人分のデータを以下のように時系列でテキスト化する:

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

- 年ごとにまとめ、役職・残業・受講情報を時系列に並べる
- 数値は「約30時間」のように自然言語的に表現

---

## 使用モデル

| モデル | 用途 |
|--------|------|
| `sbintuitions/modernbert-ja-70m` | 速度重視・初期実験向け |
| `sbintuitions/modernbert-ja-310m` | 精度重視・GPUメモリに余裕があれば推奨 |

- ModernBERT: BERT/RoBERTa系エンコーダを改善したモデル（高速・高性能）
- 最大8192トークン対応
- BERTはLLMのLoRAと異なり**全パラメータのフルチューニング**が高速にできる点が短期コンペで有利

---

## 推奨学習設定

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

- CV戦略: `GroupKFold`（社員番号でグループ化、リーク防止）

---

## 実装上のポイント

### 入力トークン長
- 最大8192トークン対応だが、メモリ消費に注意
- LightGBMで有効だった特徴量を参考に情報の取捨選択を推奨
- まず `max_length=1024` から試す

### 精度と速度のトレードオフ
- **速度重視**: `bf16=True` / `torch_dtype=torch.bfloat16`（ハイパーパラメータ探索フェーズ）
- **精度重視**: `bf16=False` / `torch_dtype=torch.float32`（最終提出フェーズ）
- bfloat16で精度劣化が気になる場合は Kahan Summation（[optimi](https://github.com/warner-benjamin/optimi)）も選択肢

### LightGBMとのアンサンブル
- ModernBERT単体のOOF予測値をLightGBMのスタッキング入力に使う、またはシンプルな加重平均でアンサンブルすることでさらなるスコア向上が期待できる

---

## スコア

- Private: **0.7134**（現ベースライン LB:0.6364 から大幅改善の余地あり）

---

## 必要データ形式

```python
# train: 社員番号, category, prompt, labels
# test:  社員番号, category, prompt
# labels は target のリネーム
```

---

## 参考コード

`sample_code/3th_place_solution/takaito.ipynb` および `sample_code/3th_place_solution/details.md` を参照。
