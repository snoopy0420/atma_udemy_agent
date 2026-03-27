# 高度な特徴量エンジニアリング・モデリング調査レポート

調査日: 2026-03-26
目的: ベースライン(CV:0.610, LB:0.6364)からのスコア向上のための手法調査

---

## コンペ現状まとめ（参照用）

- **実装済み**: 全テーブルの基本集計（mean/max/std/sum/count）、category one-hot、最新役職、在籍年数、昇進回数
- **CV**: 0.610 / **LB**: 0.6364
- **モデル**: LightGBM + StratifiedGroupKFold(n_splits=5, group=社員番号)

---

## 1. 類似コンペの知見

### Home Credit Default Risk (Kaggle 1位解法)

- **URL**: https://www.kaggle.com/c/home-credit-default-risk / https://github.com/kozodoi/Kaggle_Home_Credit
- **要約**: 7テーブル構造×二値分類×ROC-AUCの多テーブルコンペ。1位解法では特徴量エンジニアリングがモデルチューニング・スタッキングより有効と明言。
- **キーアイデア**:
  - **比率特徴量**: あるテーブルの特徴量を別テーブルの特徴量で割った比率（例: credit/annuity ratio）
  - **近傍ターゲット平均**: 類似サンプルの目的変数の平均（KNNベース）
  - **時系列window集計**: 特定期間のみに限定したaggregation（直近N件、最初のN件など）
  - **Weighted Moving Average**: 時系列データへの加重移動平均
- **このコンペへの転用可能性**: **高** — 多テーブル×ROC-AUCの構造が一致
- **実装難易度**: 中

### WNS Analytics Wizard / Employee Promotion Prediction

- **URL**: https://www.kaggle.com/code/angps95/employee-promotion-prediction / https://www.kaggle.com/code/rsnayak/wns-analytics-wizard-2018-ml-hackathon
- **要約**: 社員データを使った昇進予測コンペ。ターゲットエンコーディングとカテゴリ変数の組み合わせ特徴量が重要。
- **キーアイデア**:
  - カテゴリ変数同士の組み合わせ（例: department × region）
  - 在籍年数と研修スコアの積など乗算・除算特徴量
- **このコンペへの転用可能性**: **高** — 社員単位の二値分類で構造類似
- **実装難易度**: 低〜中

---

## 2. 高度な特徴量エンジニアリング手法

### 2-1. 残業データの時系列特徴量

- **参考**: https://towardsdatascience.com/lazyprophet-time-series-forecasting-with-lightgbm-3745bafe5ce5/ / https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast

**実装可能な特徴量**:
```
overtime_trend_slope           # 線形回帰の傾き（増加/減少トレンド）
overtime_recent3m_mean         # 直近3ヶ月平均
overtime_recent6m_mean         # 直近6ヶ月平均
overtime_first3m_mean          # 最初の3ヶ月平均（入社直後の行動）
overtime_cv (std/mean)         # 変動係数（安定性）
overtime_month_count_nonzero   # 残業ゼロでない月数
overtime_max_consecutive_zero  # 残業ゼロが連続した最大月数
overtime_last_value            # 最終月の残業時間
overtime_yoy_change            # 前年比変化率
```

- **このコンペへの転用可能性**: **高** — overtime_work_by_monthは月次時系列データ
- **実装難易度**: 低〜中
- **期待効果**: 残業トレンドは行動変化を捉えるため信号が強い可能性

### 2-2. Udemyアクティビティの詳細特徴量

- **参考**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9359516/ (Predicting student performance using sequence classification)

**実装可能な特徴量**:
```
udemy_active_days              # 受講した日数（ユニーク日数）
udemy_active_months            # 受講した月数
udemy_sessions_per_month       # 月あたりセッション数
udemy_completion_trend         # 完了率の時系列トレンド（最近vs過去）
udemy_binge_ratio              # 同日複数レクチャー比率（一気観い）
udemy_course_diversity         # 受講コースカテゴリ数（nunique）
udemy_avg_lecture_duration     # 平均レクチャー所要時間（end - start）
udemy_quiz_ratio               # クイズ受講比率（クイズ数/全レクチャー数）
udemy_quiz_score_mean          # クイズ平均スコア（最終結果列）
udemy_marked_done_ratio        # マーク済み終了率（marked_done/count）
udemy_recent3m_count           # 直近3ヶ月のレクチャー数
udemy_course_completion_count  # コース単位での完了率が高いコース数
```

- **このコンペへの転用可能性**: **高** — Udemyデータは最も大規模（539K行）
- **実装難易度**: 中
- **期待効果**: 学習の「深さ」「継続性」「多様性」を捉えられる

### 2-3. 職位履歴の詳細特徴量

**実装可能な特徴量**:
```
position_has_promoted           # 昇進経験あり（bool）
position_years_since_promo      # 最後の昇進からの経過年数
position_rank_label_latest      # 最新役職のランク（数値エンコード）
position_rank_change_total      # 役職ランクの総変化量（正=昇進, 負=降格）
position_employment_type_change # 雇用形態変化あり（bool）
position_tenure_per_position    # 1役職あたりの平均在籍年数
```

- **実装難易度**: 低
- **期待効果**: 現在の役職だけでなく「キャリア軌跡」の情報を追加

### 2-4. テーブル間クロス特徴量

**実装可能な特徴量（ドメイン知識ベース）**:
```
udemy_count_per_overtime_month  # 残業月数あたりのUdemy受講数（多忙さに対する学習意欲）
dx_count_per_position_tenure    # 在籍年数あたりのDX研修受講数
hr_count_per_position_year      # 職位年数あたりのHR施策利用数
overtime_mean_per_position_year # 役職ごとの平均残業時間
udemy_diversity_x_dx_count      # Udemy多様性 × DX研修数（積特徴量）
```

- **参考**: Kaggle Grandmasters Playbook (NVIDIA) — 8カテゴリ列の組み合わせで28特徴量を生成
  - URL: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- **実装難易度**: 低
- **期待効果**: 中（単体では弱いが複数組み合わせで効く可能性）

### 2-5. ターゲットエンコーディング（カテゴリ変数）

- **参考**: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/

**対象カテゴリ変数**:
- 最新役職名（役職ラベル）
- Udemyコースカテゴリ（最多受講カテゴリなど）
- DX研修カテゴリ
- HR施策カテゴリ

**実装上の注意**:
- CVのfold内で fit_transform する（リーケージ防止）
- `sklearn.preprocessing.TargetEncoder` (sklearn 1.3+) または `category_encoders.TargetEncoder`
- スムージングパラメータを設定（平滑化でoverfitting防止）

- **実装難易度**: 中
- **期待効果**: 高カーディナリティのカテゴリ変数に対して有効

---

## 3. モデリング手法

### 3-1. CatBoost（カテゴリ変数の自動処理）

- **URL**: https://createbytes.com/insights/xgboost-lightgbm-catboost-gradient-boosting
- **要約**: CatBoostはカテゴリ変数を前処理なしで利用可能。Ordered Boostingによりターゲットリーケージを防止したエンコーディングを内部で実施。
- **キーアイデア**: 役職・Udemyカテゴリ等の文字列カテゴリをそのまま投入可能
- **このコンペへの転用可能性**: **高** — 当コンペは役職、研修カテゴリ等の文字列カテゴリが多数存在
- **実装難易度**: 低（LightGBMと同様のAPI）
- **期待効果**: LightGBMと異なる予測で**アンサンブル効果**を狙える

### 3-2. XGBoost + LightGBM + CatBoost スタッキング

- **URL**: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- **要約**: 3種のGBDTをOOF予測でスタッキング。多様性が高く安定したアンサンブルが組める。
- **キーアイデア**: 各モデルのOOF予測を特徴量として2段目のモデルを学習
- **実装難易度**: 中
- **期待効果**: 高（Kaggle上位解法での定番）

### 3-3. AutoGluon（AutoML）

- **URL**: https://auto.gluon.ai/stable/ / https://github.com/autogluon/autogluon
- **要約**: 特徴量エンジニアリング〜モデル選択〜アンサンブルを自動化。2025年版では`zeroshot_2025_tabfm`プリセットで基盤モデルも活用可能。
- **キーアイデア**: `TabularPredictor(eval_metric='roc_auc').fit(train_data, presets='best_quality')` のみで高品質な予測
- **このコンペへの転用可能性**: **中** — 多テーブルの前処理は自分で実施する必要あり。特徴量マージ後の単一テーブルに適用
- **実装難易度**: 低（コード量は少ないが環境構築に注意）
- **期待効果**: 中〜高（探索的なスコア上限確認に有用）

### 3-4. TabPFN v2（小規模データ向け基盤モデル）

- **URL**: https://www.nature.com/articles/s41586-024-08328-6 / https://github.com/PriorLabs/TabPFN
- **要約**: 1億以上の合成データセットで事前学習されたTransformerモデル。データ数が10,000以下の場合にGBDTを凌駕することが多い（Nature誌掲載）。
- **キーアイデア**: In-context learningにより追加学習不要。`TabPFNClassifier().fit(X_train, y_train).predict_proba(X_test)` で動作
- **このコンペへの転用可能性**: **中** — trainが1223社員×6カテゴリ=7338行と小規模。ただしtest社員との分離が重要
- **実装難易度**: 低（API簡単）
- **期待効果**: 中（アンサンブルの多様性向上に有効）

### 3-5. FT-Transformer / SAINT

- **URL**: https://arxiv.org/abs/2410.12034 (A Survey on Deep Tabular Learning) / https://openreview.net/forum?id=nL2lDlsrZU
- **要約**: FT-Transformerは各特徴量をトークンとして扱いself-attentionで特徴間相互作用を学習。SAINTは行×列の2方向attentionを持つ。
- **このコンペへの転用可能性**: **低〜中** — データ量が少ないため効果は限定的。ただしアンサンブルの多様性として有効
- **実装難易度**: 高
- **期待効果**: 低〜中（単独では弱いがアンサンブル要素として機能する可能性）

---

## 4. 学習戦略・後処理

### 4-1. 擬似ラベリング（Pseudo Labeling）

- **URL**: https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969
- **要約**: ベストモデルでtestデータにラベルを付与し、信頼度の高いサンプルをtrainに追加して再学習。
- **キーアイデア**: 確率が0.9以上または0.1以下のサンプルのみ使用（hard pseudo-label）またはソフトラベルを使用
- **このコンペへの転用可能性**: **高** — trainとtestの社員が完全に分離している構造に有効
- **実装難易度**: 中
- **期待効果**: 中（LBスコアが高く出ている場合に特に有効）

### 4-2. Optuna によるLightGBM ハイパーパラメータ最適化

- **URL**: https://forecastegy.com/posts/how-to-use-optuna-to-tune-lightgbm-hyperparameters/
- **要約**: BayesianOptimizationで効率的にハイパーパラメータ探索。`optuna.integration.lightgbm.LightGBMTuner`が専用統合を提供。
- **キーアイデア**: num_leaves, feature_fraction, lambda_l1/l2 等を主要パラメータとして最適化
- **実装難易度**: 低〜中
- **期待効果**: 低〜中（特徴量改善ほどではないが、確実な改善手段）

### 4-3. ヒルクライミングアンサンブル

- **URL**: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- **要約**: 最強単一モデルから始め、他モデルを重み付きで追加。バリデーションスコアが改善する場合のみ採用。
- **実装難易度**: 中
- **期待効果**: 高（Kaggle上位解法の定番）

---

## 5. 関連論文

### "A Survey on Deep Tabular Learning" (arXiv 2410.12034, 2024)

- **URL**: https://arxiv.org/abs/2410.12034
- **要約**: TabNet, SAINT, TabTransformer, FT-Transformer, MambaNetなどを網羅したサーベイ。FCNからattentionベースアーキテクチャへの進化を追跡。
- **キーアイデア**: ハイブリッドアーキテクチャ（Attention + MLP）が実用的なバランスを達成
- **転用可能性**: **低〜中**（実装コストに見合う効果は限定的）

### "TabPFN v2: Accurate predictions on small data" (Nature 2024)

- **URL**: https://www.nature.com/articles/s41586-024-08328-6
- **要約**: 1.3億件の合成データで事前学習。10K以下のデータでGBDTを超える性能。
- **キーアイデア**: In-context learning、合成データ事前学習
- **転用可能性**: **中**（trainが7K行と適合範囲内）

### "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer論文)

- **URL**: https://www.semanticscholar.org/paper/Revisiting-Deep-Learning-Models-for-Tabular-Data-Gorishniy-Rubachev/5fa06d856ba6ae9cd1366888f8134d7fd0db75b9
- **要約**: FT-Transformerがtabularにおけるdeep learningのSOTAを更新。各特徴量をfeature tokenとして埋め込みattentionを適用。
- **転用可能性**: **中**（アンサンブル多様性のため）

---

## 推奨アクション（優先度順）

| 優先度 | アイデア | 期待効果 | 難易度 | 対応idea-ID候補 |
|--------|----------|----------|--------|-----------------|
| 最優先 | 残業データの時系列特徴量（trend, 直近N月, CV） | 高 | 低 | idea-101 |
| 最優先 | Udemyアクティビティの詳細特徴量（active_days, binge_ratio, duration等） | 高 | 中 | idea-102 |
| 高 | ターゲットエンコーディング（役職・コースカテゴリ） | 高 | 中 | idea-103 |
| 高 | CatBoost追加によるアンサンブル | 中〜高 | 低 | idea-201 |
| 高 | 擬似ラベリング（testの高確信サンプルをtrainに追加） | 中 | 中 | idea-301 |
| 中 | テーブル間クロス特徴量（比率・積） | 中 | 低 | idea-104 |
| 中 | 職位履歴の詳細特徴量（昇進からの経過年数、軌跡） | 中 | 低 | idea-105 |
| 中 | Optuna によるハイパーパラメータ最適化 | 低〜中 | 低 | idea-202 |
| 中 | AutoGluon で上限スコアを確認 | 中 | 低 | idea-203 |
| 低 | TabPFN v2（アンサンブル多様性） | 中 | 低 | idea-204 |
| 低 | XGBoost + LightGBM + CatBoost スタッキング | 高 | 中 | idea-302 |
| 低 | FT-Transformer / SAINT | 低〜中 | 高 | idea-205 |
