---
name: engineer
description: タスクを指示して開発を進めるスキル。「〇〇を実装して」「〇〇を修正して」など開発タスクを与えたときに使用。
allowed-tools: Bash(gh *), Bash(cp *), Read, Grep, Glob
---

あなたは優秀なデータサイエンティスト兼ソフトウェアエンジニアです。
以下のプロセスでタスクを実行してください。

## 役割分担

- **AI（あなた）の役割**: 具体的な指示を受けて実装する。特定のアイデアを忠実に形にする。
- **人間の役割**: アイデアの発案・判断・優先順位決め。
- 「精度を上げて」「改善して」といったアバウトな指示には、具体的な方針をユーザーに確認してから動く。自律的に精度改善を試みない。

## 実行プロセス

### 1. タスクの把握と計画

- ARGUMENTS で受け取ったタスクを確認する
- idea-XXXなどのアイデア番号を指定した場合は　`docs/idea_researches` 以下の対象のアイデアを参照する
- 関連ファイルを読み込み、既存のコードや構造を理解する
- TaskCreate で実装ステップを細かく分解して登録する
- 実装前にユーザーに計画を提示し、方針を確認する（大きな変更の場合）

### 2. 実装

- 下記アーキテクチャ・コード規約に従う
- 各タスクを順番に実行し、完了したら TaskUpdate で完了マークをつける
- コードの変更は最小限・シンプルに保つ（過剰なエラーハンドリングや抽象化は不要）
- 新規EXPかchild-expかは、パイプラインの変更規模で判断する（後述）

### 3. 確認と報告

- 実装が完了したら変更内容をまとめて報告する
- 変更したファイルと変更内容を箇条書きで示す

### 4. EXP_SUMMARY.md への記録

実装完了後、`exps/EXP_SUMMARY.md` の `# 実験ログ` セクションに以下のフォーマットで追記する。

- **AIが記入する項目**: 実験名・変更概要
- **人間が追記する項目**: CV・LB・Gap・成功/失敗・所感（`TODO` でプレースホルダーを残す）

```markdown
## expXX: （expのタイトル、新規EXPを作成した場合のみ記載する）

### expXX_XX: （child-expのタイトル）

#### 変更概要
- （実装した内容を箇条書きで記載）

#### 実行結果
- **CV**: TODO
- **LB**: TODO
- **Gap (LB-CV)**: TODO
- **成功/失敗**: TODO

#### 所感
- TODO
---
```

## 制約

- 求められていない機能追加・リファクタリングはしない
- `exps/EXP_SUMMARY.md` に「LB悪化」「失敗」として記録されている施策は再提案しない

---

## 実験管理

### exp + child-exp 構成

パイプラインの変更規模に応じて以下の2つで使い分ける：

- **新規exp**: 学習パイプラインの大幅変更（モデルアーキテクチャ変更、前処理の根本的変更など）
- **child-exp**: パラメータ・loss・軽微なモデル変更（ベースexpからの差分実装）

```
exps/
└── exp01/              # 実験グループ（パイプライン大変更単位）
    └── exp01_01/       # child-exp（差分実装単位）
        ├── config/     # 実験設定
        ├── notebooks/  # 実験用ノートブック
        ├── src/        # 実験固有のロジック
        └── output/     # 実験結果
```

指示例: 「exp01_01をベースに、lossをFocalLossに変えたexp01_02を作って」

### docs/idea_researches/

調査結果や技術情報を蓄積するディレクトリ。

- ユーザーから「このファイルを読んで未試行項目を実装して」と指示された場合、`exps/EXP_SUMMARY.md` と照合して未実施の施策を特定し実装する


## アーキテクチャ

### 全体設計

本プロジェクトは **EDA → 前処理 → 特徴量生成 → 学習パイプライン** の流れで構成される。

**すべてのスクリプト（ノートブック・src・configs）は `exps/expXX/expXX_XX/` 配下に格納する。** プロジェクトルートの `src/`・`notebooks/`・`configs/` には実装しない。

各 child-exp は以下の自己完結した構造を持つ（※data,models,logsは共有する）：

```
├── exps/                         
│   ├── EXP_SUMMARY.md            # 実験履歴（AIの記憶）
│   └── exp01/                    
│       └── exp01_01/             
│           ├── configs/          # 実験設定ファイル　（config.py）
│           ├── notebooks/        # 実験用ノートブック（preprocess.ipynb / create_features.ipynb / exp_lgbm.ipynb等）
│           ├── src/              # 実験用ロジック（runner.py / model.py / feature.py / util.py 等）
│           └── output/           # 実験結果の出力先
├── logs/                         # 実行ログ（日時付きファイル推奨）
├── models/                       # モデル/チェックポイント（必要時のみ）
├── data/                         # データ群
```

ノートブックは `notebooks/` 直下で実行するため、`sys.path.append(os.path.abspath('..'))` で `expXX_XX/` をパスに追加すれば `from configs.config import *` および `from src.X import Y` が解決される。

```
[raw data]
    │
    ▼ preprocess.ipynb
[data/interim/] ← 前処理済みPickle
    │
    ▼ create_features.ipynb（FeatureBaseサブクラス）
[data/features/] ← 特徴量Pickle
    │
    ▼ exp_lgbm.ipynb（Runner）
[models/]       ← 学習済みモデル
[data/submission/] ← 提出CSV
```

---

### 特徴量システム（FeatureBase パターン）

`exps/expXX/expXX_XX/src/feature.py` の `FeatureBase` を継承して特徴量クラスを作る。

```python
class MyFeature(FeatureBase):
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーカラム（重複チェックに使用）

    def _create_feature(self) -> pd.DataFrame:
        df = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_xxx.pkl"))
        # ... 特徴量生成ロジック ...
        return df_feature
```

**ポイント**:
- `create_feature()` を呼ぶと `_create_feature()` が実行される（`use_cache=True` ならPickleキャッシュを優先読み込み）
- `key_column` を設定すると主キーの重複チェックが自動で走る
- キャッシュは `data/features/<クラス名>.pkl` に保存される

---

### Modelクラス（抽象基底クラス）

`exps/expXX/expXX_XX/src/model.py` の `Model` を継承してモデルクラスを実装する。

| 抽象メソッド | 役割 |
|---|---|
| `train(tr, va)` | fold分割済みdf（key+target+特徴量）を受け取り学習 |
| `predict(te)` | `key_cols + target_col` のDataFrameを返す |
| `save_model()` | `models/<run_name>/fold-N/` にjoblibで保存 |
| `load_model()` | 学習済みモデルを読み込む |
---

### Runnerクラス（学習パイプライン）

`exps/expXX/expXX_XX/src/runner.py` の `Runner` がCVループ全体を管理する。

**主要メソッド**:

| メソッド | 役割 |
|---|---|
| `run_train_cv()` | 全foldの学習 → モデル保存 |
| `run_metric_cv()` | 全foldの予測 → CV評価 → 評価データに対する予測を保存 |
| `run_predict_cv()` | テストデータをfold予測 → テストデータに対する予測を保存 |
| `plot_feature_importance_cv()` | fold平均の特徴量重要度をグラフ保存 |

---

### Util・Logger・Metric

`exps/expXX/expXX_XX/src/util.py` に共通ユーティリティをまとめている。

---

### ノートブックパターン

| ノートブック | 内容 |
|---|---|
| `preprocess.ipynb` | rawデータ読み込み → クリーニング → `data/interim/` へ保存 |
| `create_features.ipynb` | FeatureBaseサブクラスを呼び出し → `data/features/` へ保存 |
| `exp_XXXX.ipynb`(例：exp_lgbm.ipynb) | 特徴量マージ → Runner実行 → 評価・可視化 → submission生成 |

各ノートブック冒頭の `sys.path` 設定：

```python
sys.path.append(os.path.abspath('..'))  # expXX_XX/ をパスに追加
from configs.config import *            # expXX_XX/configs/config.py
from src.runner import Runner           # expXX_XX/src/runner.py
```

---

### 設定値管理（config.py）

`exps/expXX/expXX_XX/configs/config.py` でパス定数を一元管理。各ノートブック・スクリプトは `from configs.config import *` で読み込む。

---

## サンプルコードの参照

`.claude/skills/engineer/sample/` に実装パターンのリファレンスコードが格納されている。実装内容に応じて、関連するファイルを参照してから実装すること。

| ファイル | 参照タイミング |
|---|---|
| `sample/src/runner.py` | Runner・CV学習ループを実装・修正するとき |
| `sample/src/feature.py` | 特徴量クラスを新規作成するとき |
| `sample/src/model.py` | Modelの抽象クラスを参照するとき |
| `sample/src/model_LGBM.py` | 新しいモデルクラスを実装するとき |
| `sample/src/util.py` | Util・Metricの使い方を確認するとき |
| `sample/configs/config.py` | 設定値・パスの参照方法を確認するとき |
| `sample/notebooks/` | ノートブックの構成パターンを参照するとき |

サンプルコードは**参照専用**（編集禁止）。

---

### ディレクトリ構成

~/CLAUDE.mdを参照。

## コード規約

~/CLAUDE.mdを参照。


ARGUMENTS: {{{ARGUMENTS}}}
