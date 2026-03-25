# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# プロジェクト概要

このプロジェクトはデータ分析コンペ用の開発を実施します。
コンペの詳細は./doc/overview.mdに記載します。

# 役割
あなたは優秀なデータサイエンティスト兼ペアプロです。

# ディレクトリ構成

```
.
├── README.md                     # リポジトリ全体の簡易概要
├── requirements.txt              # 依存パッケージ
├── configs/
│   └── config.py                 # 設定値（定数/パス）を集中管理
├── data/
│   ├── raw/                      # 生データ（配布物・外部取得データ）
│   │   └── input/                # コンペ配布の入力ファイル類
│   ├── interim/                  # 中間生成物（フィルタ後/正規化後など）
│   ├── features/                 # 特徴量生成結果（数値・埋め込み等）
│   ├── figures/                  # 可視化出力（PNG/HTML等）
│   └── submission/               # 提出用CSVの生成先
├── docs/
│   ├── overview.md               # コンペ概要
│   ├── data_discription.md       # データセットの詳細説明
│   └── idea_researches/          # アイデア調査結果の蓄積
├── exps/                         # 実験管理ディレクトリ
│   ├── EXP_SUMMARY.md            # 実験履歴（AIの記憶）
│   └── exp01/                    # 実験グループ（パイプライン大変更単位）
│       └── exp01_01/             # child-exp（パラメータ・loss等の差分）
│           ├── config/           # 実験設定ファイル
│           ├── notebooks/        # 実験用ノートブック
│           ├── src/              # 実験用ロジック
│           └── output/           # 実験結果の出力先
├── logs/                         # 実行ログ（日時付きファイル推奨）
├── models/                       # モデル/チェックポイント（必要時のみ）
├── notebooks/                    # 本番用Notebook
├── sample_code/                  # 参考サンプル（他人のコードなど）
├── src/                          # 本番用モジュール
├── tmp/                          # 検証などで利用するファイルやコード群
└── CLAUDE.md                     # このリポジトリの開発/運用ガイド

```

# アーキテクチャ

詳細なアーキテクチャ（EDA・Model-Runner-Notebookパターン・特徴量システム・学習パイプライン）が必要な場合は `.claude/skills/engineer/SKILL.md` を参照。

# コード規約

### コメント

- docstringはGoogle styleで書く
- コメントは日本語で記述する
- コメントは#の数で階層化する。数が少ないほど上位階層となる。

### 変数名

- 以下のデータ型についてはprefixにデータ型を付与する。
    - df_: pandas Dataframe
    - pdf_: polars Dataframe
    - dict_: 辞書
    - list_: リスト
    - set_: セット

# 環境・実行方法

- パッケージ管理・スクリプト実行は `uv` を使用する（例: `uv run python script.py`）

# 禁止事項

- `data/raw/` 配下のファイルは直接編集しない（読み取り専用）
- `sample_code/` 配下のファイルは編集しない（参照専用）



