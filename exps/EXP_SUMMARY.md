# 実験サマリー

AIの記憶として機能するファイル。**実装前に必ず参照し、重複した施策を避ける。**

# 実験ログ

以下に実験ログを記載する。

## exp01: ベースライン（多テーブル集計特徴量 + LightGBM）

### exp01_01: ベースライン

#### 変更概要
- 全サブテーブル（udemy_activity / overtime / position_history / dx / hr / career）を社員単位に集計した特徴量を実装
- `CategoryFeature`: category の one-hot エンコーディング
- `UdemyActivityFeature`: レクチャー数・コース数・完了率（mean/max）・マーク済み修了数（sum/mean）
- `OvertimeWorkByMonthFeature`: 残業時間の mean/median/max/min/std/sum/count
- `PositionHistoryFeature`: 最新役職 label enc・在籍年数・昇進回数
- `DxFeature`: 研修受講数・研修カテゴリ別カウント
- `HrFeature`: 施策利用数・施策カテゴリ別カウント
- `CareerFeature`: キャリアアンケート回答をそのまま結合（375人分、欠損NaN）
- LightGBM binary classification、`StratifiedGroupKFold(n_splits=5, group=社員番号)`
- 不均衡対策: `is_unbalance=True`

#### 実行結果
- **CV**: 0.610182114818443
- **LB**: 0.6364
- **Gap (LB-CV)**: TODO
- **成功/失敗**: 成功

#### 所感
- TODO

---

### exp01_02: 残業データの時系列特徴量追加

#### 変更概要
- `OvertimeTimeSeriesFeature` クラスを新規実装（idea-101）
- `ot_ts_trend_slope`: 月次残業時間の線形回帰傾き（増加/減少トレンド）
- `ot_ts_recent3m_mean`: 直近3ヶ月平均
- `ot_ts_recent6m_mean`: 直近6ヶ月平均
- `ot_ts_first3m_mean`: 最初の3ヶ月平均（入社直後の行動）
- `ot_ts_cv`: 変動係数（std / mean、安定性の指標）
- `ot_ts_nonzero_count`: 残業ゼロでない月数
- `ot_ts_max_consec_zero`: 残業ゼロが連続した最大月数
- `ot_ts_last_value`: 最終月の残業時間
- `ot_ts_yoy_change`: 前年比変化率（24ヶ月以上のデータがある場合のみ算出）

#### 実行結果
- **CV**: 0.6196033849704585
- **LB**: 0.6327
- **Gap (LB-CV)**: TODO
- **成功/失敗**: 保留

#### 所感
- TODO
---

### exp01_03: Udemyアクティビティ詳細特徴量追加

#### 変更概要
- `UdemyActivityDetailFeature` クラスを新規実装（idea-102）
- `udemy_active_days`: 受講した日数（ユニーク日数）
- `udemy_active_months`: 受講した月数
- `udemy_sessions_per_month`: 月あたりセッション数（active_days / active_months）
- `udemy_completion_trend`: 完了率の時系列トレンド（直近3ヶ月平均 - それ以前の平均）
- `udemy_binge_ratio`: 同日複数レクチャー比率（一気観比率）
- `udemy_course_diversity`: 受講コースカテゴリ数（nunique）
- `udemy_avg_lecture_duration`: 平均レクチャー所要時間（分単位）
- `udemy_quiz_ratio`: クイズ受講比率（クイズ数 / 全レクチャー数）
- `udemy_quiz_score_mean`: クイズ平均スコア（最終結果列）
- `udemy_marked_done_ratio`: マーク済み終了率（marked_done / count）
- `udemy_recent3m_count`: 直近3ヶ月のレクチャー数
- `udemy_course_completion_count`: 完了率100%を達成したコース数

#### 実行結果
- **CV**: 0.6184960058390182
- **LB**: 0.6287
- **Gap (LB-CV)**: TODO
- **成功/失敗**: 失敗

#### 所感
- TODO
---

## exp02: ModernBERT fine-tuning（テーブルデータ→テキスト変換）

### exp02_01: ModernBERT fine-tuning ベースライン

#### 変更概要
- テーブルデータ（overtime/position/dx/hr/udemy）を年次テキストプロンプトに変換する `src/prompt_builder.py` を実装
- プロンプト形式: category → 年ごとに役職/残業平均・最大/DX研修/HR施策/Udemyコースを時系列列挙
- モデル: `sbintuitions/modernbert-ja-70m`（速度重視・初期実験向け）
- `AutoModelForSequenceClassification`（num_labels=2）でfull fine-tuning
- CV戦略: `StratifiedGroupKFold(n_splits=5, group=社員番号)`
- 学習設定: lr=2e-4, epochs=10, batch=4×grad_accum=4, bf16=True, cosine scheduler
- OOFを `data/interim/df_oof_exp02_01.csv` に保存

#### 実行結果
- **CV**: TODO
- **LB**: TODO
- **Gap (LB-CV)**: TODO
- **成功/失敗**: TODO

#### 所感
- TODO
---