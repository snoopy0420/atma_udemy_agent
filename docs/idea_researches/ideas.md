# アイデア一覧

実装候補のアイデアを管理するファイル。
実装指示時はIDを指定する（例: 「idea-001を実装して」）。
詳細資料がある場合は `docs/idea_researches/details/` 配下のファイルを参照。

---

## 記載例

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-001 | 層化交差検証を導入する | validation_strategy.md | AI |

## 全体

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-001 | ベースライン | baseline_research.md | AI |

## 特徴量エンジニアリング

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-101 | 残業データの時系列特徴量（trend_slope, 直近N月平均, 変動係数, 最終値等） | advanced_feature_engineering_research.md | AI |
| idea-102 | Udemyアクティビティの詳細特徴量（active_days, binge_ratio, quiz_ratio, duration, recent3m等） | advanced_feature_engineering_research.md | AI |
| idea-103 | ターゲットエンコーディング（役職ラベル・コースカテゴリ・DX/HR施策カテゴリ） | advanced_feature_engineering_research.md | AI |
| idea-104 | テーブル間クロス特徴量（比率・積: udemy_count/overtime_month等） | advanced_feature_engineering_research.md | AI |
| idea-105 | 職位履歴の詳細特徴量（昇進からの経過年数, 役職ランク変化量, 1役職あたり在籍年数） | advanced_feature_engineering_research.md | AI |

## モデル・学習設定

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-201 | CatBoostを追加してLightGBMとアンサンブル（カテゴリ変数をそのまま投入） | advanced_feature_engineering_research.md | AI |
| idea-202 | Optuna によるLightGBMハイパーパラメータ最適化 | advanced_feature_engineering_research.md | AI |
| idea-203 | AutoGluon で上限スコアを確認（best_quality preset, roc_auc） | advanced_feature_engineering_research.md | AI |
| idea-204 | TabPFN v2でアンサンブル多様性を追加（小規模データ向け基盤モデル） | advanced_feature_engineering_research.md | AI |
| idea-205 | FT-Transformer / SAINTによるdeep learning（アンサンブル要素として） | advanced_feature_engineering_research.md | AI |
| idea-206 | テーブルデータを年次テキスト（役職・残業・受講情報）に変換し日本語ModernBERT（sbintuitions/modernbert-ja-70m or 310m）でfine-tuning（3位解法, Private:0.7134） | modernbert_text_conversion_research.md | 3位 |

## アンサンブル・後処理

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-301 | 擬似ラベリング（testの高確信サンプルをtrainに追加して再学習） | advanced_feature_engineering_research.md | AI |
| idea-302 | XGBoost + LightGBM + CatBoost のOOFスタッキング | advanced_feature_engineering_research.md | AI |
| idea-303 | ヒルクライミングアンサンブル（加重平均の重みを最適化） | advanced_feature_engineering_research.md | AI |

## バリデーション戦略

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-401 | - | - | - |

## その他

| ID | アイデア | 詳細資料 | 主体 |
|----|----------|----------|----------|
| idea-501 | - | - | - |
