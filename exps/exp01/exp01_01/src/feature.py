import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABCMeta, abstractmethod
from time import time

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Util


class Timer:
    """処理時間を計測するコンテキストマネージャ"""

    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):
        if prefix:
            format_str = str(prefix) + sep + format_str
        if suffix:
            format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


class FeatureBase(metaclass=ABCMeta):
    """特徴量生成の基底クラス"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path(DIR_FEATURE)
        self.logger = logger
        self.save_cache = save_cache
        self.key_column = None

    def enforce_key_integrity(self, df: pd.DataFrame) -> None:
        """主キーの存在・重複チェック"""
        for key in self.key_column:
            if key not in df.columns:
                raise KeyError(f"{self.name}: キーカラム '{key}' が存在しません")
        assert not df[self.key_column].duplicated().any(), \
            f"{self.name}: 主キー {self.key_column} に重複があります"

    @abstractmethod
    def _create_feature(self) -> pd.DataFrame:
        raise NotImplementedError()

    def create_feature(self) -> pd.DataFrame:
        file_name = os.path.join(self.cache_dir, f"{self.name}.pkl")

        # キャッシュがあれば読み込む
        if os.path.isfile(str(file_name)) and self.use_cache:
            feature = pd.read_pickle(file_name)
            print(f"★ {self.name} の特徴量をキャッシュから読み込みました ★")
            return feature

        # 特徴量生成
        feature = self._create_feature()

        # 主キーチェック
        if self.key_column is not None:
            self.enforce_key_integrity(feature)

        # キャッシュ保存
        if self.save_cache:
            feature.to_pickle(file_name)

        return feature


# ## 基本クラス ###############################################################

class Key(FeatureBase):
    """train/test の主キー（社員番号 × category）を返すクラス"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']

    def _create_feature(self) -> pd.DataFrame:
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_key = pd.concat([df_train, df_test], ignore_index=True)[self.key_column]
        return df_key


class Target(FeatureBase):
    """目的変数（train のみ）を返すクラス"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']

    def _create_feature(self) -> pd.DataFrame:
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        return df_train[['社員番号', 'category', 'target']]


# ## 特徴量クラス ##############################################################

class CategoryFeature(FeatureBase):
    """category の one-hot エンコーディングを生成するクラス

    社員番号 × category の主キーに対して category ダミー変数を付与する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']

    def _create_feature(self) -> pd.DataFrame:
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_key = pd.concat([df_train, df_test], ignore_index=True)[['社員番号', 'category']]

        # category を one-hot 変換
        df_category_dummies = pd.get_dummies(df_key['category'], prefix='category').astype(int)
        df_feature = pd.concat([df_key, df_category_dummies], axis=1)

        return df_feature


class UdemyActivityFeature(FeatureBase):
    """Udemy アクティビティの社員別集計特徴量を生成するクラス

    総レクチャー数・コース数・平均完了率・マーク済み修了数を集計する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_udemy = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_udemy_activity.pkl'))

        # 前処理後のカラム名: レクチャー/クイズID → レクチャー_クイズID, 推定完了率% → 推定完了率_
        df_feature = df_udemy.groupby('社員番号').agg(
            udemy_lecture_count=('レクチャー_クイズID', 'count'),
            udemy_course_nunique=('コースID', 'nunique'),
            udemy_completion_mean=('推定完了率_', 'mean'),
            udemy_completion_max=('推定完了率_', 'max'),
            udemy_marked_done_sum=('マーク済み修了', 'sum'),
            udemy_marked_done_mean=('マーク済み修了', 'mean'),
        ).reset_index()

        return df_feature


class OvertimeWorkByMonthFeature(FeatureBase):
    """月次残業データの社員別基礎統計量を生成するクラス

    mean・median・max・min・std・count を集計する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_overtime = pd.read_pickle(
            os.path.join(DIR_INTERIM, 'df_prep_overtime_work_by_month.pkl')
        )

        df_feature = df_overtime.groupby('社員番号').agg(
            ot_mean=('hours', 'mean'),
            ot_median=('hours', 'median'),
            ot_max=('hours', 'max'),
            ot_min=('hours', 'min'),
            ot_std=('hours', 'std'),
            ot_sum=('hours', 'sum'),
            ot_count=('hours', 'count'),
        ).reset_index()

        return df_feature


class PositionHistoryFeature(FeatureBase):
    """職位履歴の社員別特徴量を生成するクラス

    最新年・在籍年数・昇進回数（役職変化数）・最新役職の label encoding を生成する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_pos = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_position_history.pkl'))

        # 最新レコード（year が最大の行）。カラム名: 役職, 勤務区分
        df_latest = df_pos.sort_values('year').groupby('社員番号').last().reset_index()
        df_latest = df_latest.rename(columns={
            'year': 'pos_latest_year',
            '役職': 'pos_latest_position',
        })

        # 在籍年数（最新年 - 最古年）
        df_tenure = df_pos.groupby('社員番号').agg(
            pos_first_year=('year', 'min'),
            pos_last_year=('year', 'max'),
        ).reset_index()
        df_tenure['pos_tenure_years'] = df_tenure['pos_last_year'] - df_tenure['pos_first_year']

        # 昇進回数（役職変化数）
        df_pos_sorted = df_pos.sort_values(['社員番号', 'year'])
        df_pos_sorted['pos_changed'] = (
            df_pos_sorted.groupby('社員番号')['役職'].shift(1) != df_pos_sorted['役職']
        ).astype(int)
        # 先頭行は変化なし扱い（shift で NaN になるため）
        df_pos_sorted.loc[df_pos_sorted.groupby('社員番号').head(1).index, 'pos_changed'] = 0
        df_promotion = df_pos_sorted.groupby('社員番号')['pos_changed'].sum().reset_index()
        df_promotion = df_promotion.rename(columns={'pos_changed': 'pos_promotion_count'})

        # 最新役職の label encoding
        df_latest['pos_latest_position_enc'] = df_latest['pos_latest_position'].astype('category').cat.codes

        # 結合
        df_feature = df_latest[['社員番号', 'pos_latest_year', 'pos_latest_position_enc']]
        df_feature = df_feature.merge(
            df_tenure[['社員番号', 'pos_tenure_years']], on='社員番号', how='left'
        )
        df_feature = df_feature.merge(df_promotion, on='社員番号', how='left')

        return df_feature


class DxFeature(FeatureBase):
    """DX研修の社員別集計特徴量を生成するクラス

    受講回数と研修カテゴリ別カウントを集計する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_dx = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_dx.pkl'))

        # 総受講回数
        df_base = df_dx.groupby('社員番号').agg(
            dx_count=('社員番号', 'count'),
        ).reset_index()

        # 研修カテゴリ別カウント（pivot）
        if '研修カテゴリ' in df_dx.columns:
            df_cat = df_dx.groupby(['社員番号', '研修カテゴリ']).size().unstack(fill_value=0)
            df_cat.columns = [f'dx_cat_{c}' for c in df_cat.columns]
            df_cat = df_cat.reset_index()
            df_feature = df_base.merge(df_cat, on='社員番号', how='left')
        else:
            df_feature = df_base

        return df_feature


class HrFeature(FeatureBase):
    """HR施策の社員別集計特徴量を生成するクラス

    施策利用回数と施策カテゴリ別カウントを集計する。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_hr.pkl'))

        # 総利用回数
        df_base = df_hr.groupby('社員番号').agg(
            hr_count=('社員番号', 'count'),
        ).reset_index()

        # カテゴリ別カウント（pivot）。カラム名: カテゴリ
        if 'カテゴリ' in df_hr.columns:
            df_cat = df_hr.groupby(['社員番号', 'カテゴリ']).size().unstack(fill_value=0)
            df_cat.columns = [f'hr_cat_{c}' for c in df_cat.columns]
            df_cat = df_cat.reset_index()
            df_feature = df_base.merge(df_cat, on='社員番号', how='left')
        else:
            df_feature = df_base

        return df_feature


class CareerFeature(FeatureBase):
    """キャリアアンケートの回答を結合するクラス

    375人分のみ存在。未回答社員の行は NaN になる。
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']

    def _create_feature(self) -> pd.DataFrame:
        df_career = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_career.pkl'))

        # カラム名に prefix を付与（社員番号以外）
        rename_dict = {
            col: f'career_{col}'
            for col in df_career.columns if col != '社員番号'
        }
        df_feature = df_career.rename(columns=rename_dict)

        return df_feature
