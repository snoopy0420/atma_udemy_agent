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
