import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import linregress
from pathlib import Path
from abc import ABCMeta, abstractmethod
from time import time
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.util import Logger, Util



def decorate(s: str, decoration=None):
    if decoration is None:
        decoration = '★' * 20

    return ' '.join([decoration, str(s), decoration])

class Timer:
    """処理時間を計測するコンテキストマネージャ。"""

    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' ', verbose=0):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None
        self.verbose = verbose

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        if self.verbose is None:
            return
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)

class FeatureBase(metaclass=ABCMeta):
    """特徴量生成の基底クラス
    """

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        self.use_cache = use_cache
        self.name = self.__class__.__name__
        self.cache_dir = Path(DIR_FEATURE)
        self.logger = logger
        self.seve_cache = save_cache
        self.use_cols = None
        self.key_column = None
    
    # 共通のキー整形 & 重複チェック
    def enforce_key_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        for key in self.key_column:
            if key not in df.columns:
                raise KeyError(f"{self.name}: キーカラム '{key}' が存在しません")
        assert ~df[self.key_column].duplicated().any(), f"{self.name}: 主キー {self.key_column} に重複があります"
    
    @abstractmethod
    def _create_feature(self) -> pd.DataFrame:
        """
        特徴量生成の実装をサブクラスで定義する必要があります。
        :return: pd.DataFrame 生成された特徴量
        """
        raise NotImplementedError()

    # 特徴量生成処理
    def create_feature(self) -> pd.DataFrame:

        # クラス名.pkl
        file_name = os.path.join(self.cache_dir, f"{self.name}.pkl")

        # キャッシュを使う & ファイルがあるなら読み出し
        if os.path.isfile(str(file_name)) and self.use_cache:
            feature = pd.read_pickle(file_name)
            print(decorate(f"{self.name}の特徴量をキャッシュから読み込みました。", decoration='★'))

        # 変換処理を実行
        else:
            # train/testの区別なく変換処理を実行
            feature = self._create_feature()

            # 主キーチェック
            if self.key_column is not None:
                self.enforce_key_integrity(feature)

            # 保存する場合
            if self.seve_cache:
                feature.to_pickle(file_name)

        return feature

############ 継承クラス ##########################################################################

class Key(FeatureBase):
    """
    キーカラムを作成するクラス
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        train.csvデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成された特徴量を含むDataFrame。
        """
        # train.csvデータを読み込む
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))
        df_test = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_test.pkl'))
        df_Key = pd.concat([df_train, df_test], ignore_index=True)[self.key_column]

        return df_Key
    

class Target(FeatureBase):
    """
    目的変数を作成するクラス
    """
    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号', 'category']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:
        """
        ターゲットデータを読み込み、特徴量を生成します。

        Returns:
        pd.DataFrame: 生成されたターゲットデータを含むDataFrame。
        """
        # ターゲットデータを読み込む
        df_train = pd.read_pickle(os.path.join(DIR_INTERIM, 'df_prep_train.pkl'))

        # 必要なカラムを選択
        df_target = df_train[['社員番号', 'category', 'target']]

        # 主キーとターゲット列を含むDataFrameを返す
        return df_target

    


class HrNameEmbeddingFeature(FeatureBase):
    """研修名をSVDで次元削減し、社員ごとの埋め込み特徴量を生成するクラス。"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 前処理済みのHRデータを読み込む
        df_hr = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_hr.pkl"))

        # スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成
        n_components = 8
        df_hr_name_embeddings = self.generate_embeddings(df_hr,
                                                   user_col='社員番号', 
                                                   action_col='研修名', 
                                                   n_components=n_components,
                                                   prefix='hr_')
        
        return df_hr_name_embeddings

    def generate_embeddings(df: pd.DataFrame, user_col: str, action_col: str, value_col=None, n_components: int = 8, prefix: str = "") -> pd.DataFrame:
        """
        スパース行列を作成し、SVDで次元削減を行い、埋め込みを生成する関数
        Args:
            df (pd.DataFrame): 入力データフレーム
            user_col (str): ユーザーを識別するカラム名
            action_col (str): アクションを識別するカラム名
            value_col (str, optional): 重みを指定するカラム名 (デフォルトはNone)
            n_components (int): SVDでの次元数
        Returns:
            pd.DataFrame: ユーザーごとの埋め込み特徴量を含むデータフレーム
        """
        # スパース行列を作成
        sparse_matrix, user_encoder, action_encoder = create_sparse_matrix(df, user_col, action_col, value_col)

        # SVDで次元削減
        svd = TruncatedSVD(n_components=n_components, random_state=42)

        # ユーザー埋め込みを生成
        user_embeddings = svd.fit_transform(sparse_matrix)
        df_user_embeddings = pd.concat([
            pd.DataFrame({user_col: user_encoder.classes_}),
            pd.DataFrame(user_embeddings, columns=[f'{prefix}svd_{action_col}_{i}' for i in range(user_embeddings.shape[1])])
        ], axis=1)

        # アクション埋め込みを生成
        action_embeddings = svd.components_.T
        course_title_to_vec = {
            course: action_embeddings[idx]
            for course, idx in zip(action_encoder.classes_, range(len(action_encoder.classes_)))
        }

        # 各ユーザーごとのベクトル平均を計算
        def compute_mean_embedding(group):
            embeddings = [course_title_to_vec[title] for title in group[action_col] if title in course_title_to_vec]
            if embeddings:
                return pd.Series(np.mean(embeddings, axis=0))
            else:
                return pd.Series([np.nan] * n_components)

        df_mean_embeddings = df.groupby(user_col).apply(compute_mean_embedding).reset_index()
        df_mean_embeddings.columns = [user_col] + [f"{prefix}mean_svd_{action_col}_{i}" for i in range(n_components)]

        # 埋め込みデータをマージ
        df_embeddings = df[[user_col]].drop_duplicates().merge(df_user_embeddings, on=user_col, how='left')
        df_embeddings = df_embeddings.merge(df_mean_embeddings, on=user_col, how='left')

        return df_embeddings

class OvertimeWorkByMonthFeature(FeatureBase):
    """月次残業データから基礎統計量（平均・中央値・最大・最小・標準偏差・件数）を生成するクラス。"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 残業データを読み込む
        df_overtime = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_overtime_work_by_month.pkl"))

        # 基礎特徴量
        df_overtime_base_feature = df_overtime.groupby(self.key_column).agg(
            avg_overtime=('hours', 'mean'),
            median_overtime=('hours', 'median'),
            max_overtime=('hours', 'max'),
            min_overtime=('hours', 'min'),
            # total_overtime_hours=('hours', 'sum'),
            std_overtime=('hours', 'std'),
            count_overtime_months=('hours', 'count'),
        ).reset_index()


        return df_overtime_base_feature
    
    
class OvertimeWorkByMonthTimeseriesFeature(FeatureBase):
    """月次残業データからlag・移動統計・トレンド・EWMなど時系列特徴量を生成するクラス。"""

    def __init__(self, use_cache=False, save_cache=False, logger=None):
        super().__init__(use_cache=use_cache, save_cache=save_cache, logger=logger)
        self.key_column = ['社員番号']  # 主キーとなるカラムを定義

    def _create_feature(self) -> pd.DataFrame:

        # 残業データを読み込む
        df_overtime = pd.read_pickle(os.path.join(DIR_INTERIM, "df_prep_overtime_work_by_month.pkl"))

        # dateの欠損行はNullで埋める
        unique_employees = df_overtime['社員番号'].unique()
        unique_dates = df_overtime['date'].unique()
        all_combinations = pd.MultiIndex.from_product([unique_employees, unique_dates], names=['社員番号', 'date']).to_frame(index=False)
        df_overtime = pd.merge(all_combinations, df_overtime, on=['社員番号', 'date'], how='left')

        # lag特徴量
        def make_worker_hours_lag_features(df_overtime, lag=35):
            """
            社員別の過去労働時間（lag特徴量）を作成し、最新月の1行にまとめる。
            Returns:
                df_worker_lag: DataFrame
                    社員番号ごとの最新行 + lag特徴量（hours_0_age ～ hours_{lag}_age）
            """
            df = df_overtime.copy()
            df = df.sort_values(['社員番号', 'date']).reset_index(drop=True)

            # lag特徴量を生成
            for i in range(1, lag + 1):
                df[f'hours_{i}_age'] = df.groupby('社員番号')['hours'].shift(i)

            # 最新行を抽出
            df_worker_lag = df.groupby('社員番号').tail(1).reset_index(drop=True)

            # カラム整形
            lag_cols = [f'hours_{i}_age' for i in range(1, lag + 1)]
            df_worker_lag = df_worker_lag[['社員番号', 'date', 'hours'] + lag_cols]
            df_worker_lag = df_worker_lag.rename(columns={'hours': 'hours_0_age'})

            return df_worker_lag
        df_worker_lag = make_worker_hours_lag_features(df_overtime, lag=35)
        df_worker_lag.drop('date', axis=1, inplace=True)


        # 移動特徴量を生成
        # 各種統計特徴量を生成するウィンドウサイズのリスト
        windows = [3, 6, 12, 24, 36]
        current_hours = df_worker_lag['hours_0_age']
        for w in windows:
            # 直近 w ヶ月分の hours列（hours_0_age ～ hours_{w-1}_age）
            cols = [f'hours_{i}_age' for i in range(0, w)]

            # 移動統計量を計算
            df_worker_lag[f'hours_mean_{w}'] = df_worker_lag[cols].mean(axis=1)              # 平均
            df_worker_lag[f'hours_std_{w}'] = df_worker_lag[cols].std(axis=1)              # 標準偏差
            df_worker_lag[f'hours_max_{w}'] = df_worker_lag[cols].max(axis=1)              # 最大値
            df_worker_lag[f'hours_min_{w}'] = df_worker_lag[cols].min(axis=1)              # 最小値
            df_worker_lag[f'hours_diff_mean_{w}'] = current_hours - df_worker_lag[f'hours_mean_{w}']  # 今月と平均の差
            df_worker_lag[f'hours_range_{w}'] = df_worker_lag[f'hours_max_{w}'] - df_worker_lag[f'hours_min_{w}']  # 振れ幅
            df_worker_lag[f'hours_missing_count_{w}'] = df_worker_lag[cols].isna().sum(axis=1)  # 欠損数
            df_worker_lag[f'hours_zscore_{w}'] = (current_hours - df_worker_lag[f'hours_mean_{w}']) / (df_worker_lag[f'hours_std_{w}'] + 1e-6)  # z-score

            # 今月と wヶ月前との比較（差分）
            df_worker_lag[f'hours_diff_prev_{w}'] = current_hours - df_worker_lag[f'hours_{w-1}_age']
            # df_worker_lag[f'hours_rate_prev_{w}'] = current_hours / df_worker_lag[f'hours_{w-1}_age']

            # 線形トレンド（回帰直線の傾き）を算出
            trends = []
            for _, row in df_worker_lag[cols].iterrows():
                y = row.values
                x = np.arange(1, w + 1).reshape(-1, 1)
                # x = np.arange(1, w + 1)[::-1].reshape(-1, 1)　 # 逆順にすることで最古月が1になる
                if np.isnan(y).all():
                    trends.append(np.nan)
                else:
                    mask = ~np.isnan(y)
                    reg = LinearRegression().fit(x[mask], y[mask])
                    trends.append(reg.coef_[0])
            df_worker_lag[f'hours_trend_{w}'] = trends

            # 今月が過去平均より ±30% を超えているか
            ratio = current_hours / (df_worker_lag[f'hours_mean_{w}'] + 1e-6)
            df_worker_lag[f'hours_over_{w}_flag'] = (ratio > 1.3).astype(int)   # 今月が30%以上多い
            df_worker_lag[f'hours_under_{w}_flag'] = (ratio < 0.7).astype(int)  # 今月が30%以上少ない

            # 指数移動平均（最近の値をより重視した平均）
            df_worker_lag[f'hours_ewm_{w}'] = df_worker_lag[cols].T.ewm(span=3, axis=0).mean().iloc[-1]
        

        # lag特徴量の削除
        cols_to_drop = [f'hours_{i}_age' for i in range(0, 36)]
        df_worker_lag.drop(columns=cols_to_drop, inplace=True)

        return df_worker_lag




















