# ベースライン調査レポート

調査日: 2026-03-26
目的: ベースライン作成のための手法・実装方針調査

---

## コンペ概要（確認済み）

- **タスク**: 二値分類（target予測）
- **評価指標**: ROC-AUC
- **主キー**: `['社員番号', 'category']`
- **データ構造**:
  - train: 7,338行 / 1,223社員 / 6カテゴリ
  - test: 11,022行 / 1,837社員（trainと重複なし）
  - サブテーブル: udemy_activity, career, dx, hr, overtime, position_history
- **特徴**: 社員単位のサブテーブルを結合して特徴量を作る多テーブル構造

---

## 1. 類似コンペの解法

### Home Credit Default Risk (Kaggle)
- **URL**: https://github.com/kozodoi/Kaggle_Home_Credit
- **要約**: 複数テーブル（7テーブル）から社員/顧客単位に特徴量を集約し、LightGBMで二値分類。Mean/Sum/Std等の集計量を多数生成。
- **キーアイデア**: 各サブテーブルをgroupby集計→mainテーブルにjoin。ターゲットエンコーディング。
- **このコンペへの転用可能性**: **高** — 全く同じ多テーブル構造×二値分類×ROC-AUC
- **実装難易度**: 低〜中

### atmaCup #16 1st place solution
- **URL**: https://speakerdeck.com/unonao/atmacup-number-16-1st-place-solution-plus-qu-rizu-mifang-zhen-rifan-ri / https://github.com/unonao/atmacup-16
- **要約**: 共起関係・セッション内特徴量が重要。LightGBM + lambdarankを使用。ドメインシフト対策が肝。
- **キーアイデア**: ドメイン固有の共起行列特徴量、ユーザー行動のシーケンス集計
- **このコンペへの転用可能性**: **中** — 多テーブル集計の方針は参考になるが、推薦問題と異なる
- **実装難易度**: 中

---

## 2. ベースラインの設計方針

### 推奨アーキテクチャ: LightGBM + GroupKFold

EDAおよび類似コンペの知見から、以下の方針でベースラインを構築する。

#### 特徴量エンジニアリング（社員単位の集計）

| テーブル | 集計特徴量の候補 |
|---|---|
| `udemy_activity` | 総レクチャー数、コース数、完了率平均、マーク済み終了数、コースカテゴリのone-hot/count |
| `overtime` | 月別残業時間の平均・最大・合計・最近N月分 |
| `position_history` | 最新役職、役職変化数（昇進回数）、最新年、在籍期間 |
| `dx` | 研修受講回数、研修カテゴリのcount |
| `hr` | 人事施策利用回数、カテゴリ別利用回数 |
| `career` | アンケート回答を数値化したまま結合 |

#### モデル

```
LightGBM (binary classification, objective='binary', metric='auc')
```

#### 交差検証

- `category`ごとにtrain/testの社員が分かれている（重複なし） → **GroupKFold on 社員番号** または **StratifiedKFold on category×target**
- 推奨: `StratifiedKFold(n_splits=5, shuffle=True)` を基本とし、後で社員番号GroupKFoldとスコア比較

---

## 3. 関連論文・手法

### neptune.ai: Binary Classification Tips from 10 Kaggle Competitions
- **URL**: https://neptune.ai/blog/binary-classification-tips-and-tricks-from-kaggle
- **要約**: 10のコンペから得られた二値分類のTips集。集計特徴量、ターゲットエンコーディング、アンサンブル等を網羅。
- **キーアイデア**: ① Target encoding with CV、② 集計特徴量（mean/max/std/count）、③ 不均衡対策（class_weight）
- **このコンペへの転用可能性**: **高** — 汎用的なTipsとして活用できる
- **実装難易度**: 低

### Manual Feature Engineering for Home Credit (Medium)
- **URL**: https://medium.com/comet-ml/manual-feature-engineering-kaggle-home-credit-db1362d683c4
- **要約**: 複数テーブルからagg特徴量を手動生成するプロセスを詳述。merge→groupby→agg→join。
- **キーアイデア**: 各テーブルで「mean, max, min, sum, std, count, nunique」を一括生成
- **このコンペへの転用可能性**: **高** — 同じ多テーブル集計の問題
- **実装難易度**: 低

---

## 4. 不均衡対策

EDAより目的変数が不均衡。以下を検討:

- `class_weight` or `scale_pos_weight` in LightGBM
- ROC-AUCはそのままでも不均衡に強いが、学習の安定化に有効
- ベースラインでは `is_unbalance=True` を試す

---

## 推奨アクション（優先度順）

1. **[最優先] Udemyアクティビティの社員別集計特徴量を作成** — データ量が最大（539,164行）、信号源として最重要
2. **[高] 残業時間の集計特徴量** — 月次時系列データ → 平均・最大・直近トレンド
3. **[高] LightGBM ベースライン実装** — StratifiedKFold 5-fold、基本集計特徴量のみ
4. **[中] 職位履歴の特徴量** — 昇進有無、最新役職のone-hot、在籍年数
5. **[中] DX研修・HR施策のカウント特徴量** — 受講回数・カテゴリ別カウント
6. **[低] キャリアアンケート結合** — 375人分のみ。欠損ありの社員はNaN→後で対処

---

## 実装参考コード（骨格）

```python
# 特徴量集計の骨格
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- Udemy集計 ---
udemy_agg = df_udemy.groupby('社員番号').agg(
    udemy_lecture_count=('レクチャー / クイズ ID', 'count'),
    udemy_course_nunique=('コースID', 'nunique'),
    udemy_completion_mean=('推定完了率%', 'mean'),
    udemy_marked_done_sum=('マーク済み終了', 'sum'),
).reset_index()

# --- overtime集計 ---
ot_agg = df_overtime.groupby('社員番号').agg(
    ot_mean=('hours', 'mean'),
    ot_max=('hours', 'max'),
    ot_sum=('hours', 'sum'),
    ot_months=('hours', 'count'),
).reset_index()

# --- マージ ---
df = df_train.merge(udemy_agg, on='社員番号', how='left')
df = df.merge(ot_agg, on='社員番号', how='left')
# ... 他のテーブルも同様

# --- LightGBM CV ---
feature_cols = [c for c in df.columns if c not in ['社員番号', 'category', 'target']]
X = pd.get_dummies(df[feature_cols], columns=['category'])
y = df['target']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    # ... lgb.train
    pass
```
