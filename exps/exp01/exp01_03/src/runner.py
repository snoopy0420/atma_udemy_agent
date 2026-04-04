import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union, Optional
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(os.path.abspath('..'))
from configs.config import *
from src.model import Model
from src.util import Util, Metric

from matplotlib import rcParams
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf')
rcParams['font.family'] = 'IPAexGothic'


class Runner:
    """学習・予測・評価を担うクラス"""

    def __init__(self,
                 run_name: str,
                 model_cls: Callable[[str, dict], Model],
                 params: dict,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 run_setting: dict,
                 cv_setting: dict,
                 logger,
                 memo,
                 ):
        self.run_name = run_name
        self.model_cls = model_cls
        self.params = params
        self.logger = logger
        self.memo = memo
        # params
        self.key_cols = params.get("key_cols")
        self.target_col = params.get("target_col")
        # run_setting
        self.after_predict_process = run_setting.get("after_predict_process", None)
        self.after_split_process = run_setting.get("after_split_process", None)
        # cv_setting
        self.group_col = cv_setting.get("group_col")
        self.n_splits = cv_setting.get("n_splits", 5)
        self.validator = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=cv_setting.get("shuffle", True),
            random_state=cv_setting.get("random_state", 42),
        )
        # データのセット
        self.df_train = df_train
        self.df_test = df_test
        self.out_dir_name = os.path.join(DIR_MODEL, run_name)

    def metric(self, true, pred):
        """評価指標の計算"""
        return Metric.my_metric(true, pred)

    def build_model(self, i_fold: Union[int, str]) -> Model:
        """foldを指定してモデルのインスタンスを生成する"""
        run_fold_name = f'{self.run_name}_fold-{i_fold}'
        model = self.model_cls(run_fold_name, self.params.copy(), self.out_dir_name, self.logger)
        return model

    def create_train_valid_dateset(self, i_fold):
        """foldを指定して訓練・検証データを準備する"""
        groups = self.df_train[self.group_col].values
        y = self.df_train[self.target_col].values
        tr_idx, va_idx = list(self.validator.split(self.df_train, y, groups))[i_fold]
        tr = self.df_train.iloc[tr_idx].reset_index(drop=True)
        va = self.df_train.iloc[va_idx].reset_index(drop=True)

        if self.after_split_process:
            tr, va = self.after_split_process(tr, va)

        return tr, va

    def train_fold(self, i_fold) -> Model:
        """foldを指定して学習を行う"""
        tr, va = self.create_train_valid_dateset(i_fold)
        model = self.build_model(i_fold)
        model.train(tr, va)
        return model

    def run_train_cv(self) -> None:
        """CVで全foldの学習を行い、モデルを保存する"""
        self.logger.info(f'{self.run_name} - start training cv')

        for i_fold in range(self.n_splits):
            self.logger.info(f'{self.run_name} fold {i_fold} - start training')
            model = self.train_fold(i_fold)
            model.save_model()
            self.logger.info(f'{self.run_name} fold {i_fold} - end training')

        # パラメータの保存
        try:
            path_output = os.path.join(self.out_dir_name, 'params.yaml')
            Util.jump_json(self.params, path_output)
        except Exception:
            self.logger.info("パラメータは保存しません")

        self.logger.info(f'{self.run_name} - end training cv')

    def metric_fold(self, i_fold):
        """foldを指定して評価を行う"""
        _, va = self.create_train_valid_dateset(i_fold)

        model = self.build_model(i_fold)
        model.load_model()

        df_va_true = va[self.key_cols + [self.target_col]]
        df_va_pred = model.predict(va)

        if self.after_predict_process is not None:
            df_va_pred = self.after_predict_process(df_va_pred, self.target_col)

        va_score = self.metric(
            df_va_true[self.target_col].values,
            df_va_pred[self.target_col].values,
        )
        return va_score, df_va_pred

    def run_metric_cv(self):
        """CVで評価を行い、OOF予測結果を保存する"""
        self.logger.info(f'{self.run_name} - start metric cv')

        scores = []
        preds = []

        for i_fold in tqdm(range(self.n_splits)):
            score, df_va_pred = self.metric_fold(i_fold)
            scores.append(score)
            preds.append(df_va_pred)

        df_va_preds = pd.concat(preds, axis=0)
        df_va_preds_true = pd.merge(
            df_va_preds,
            self.df_train[self.key_cols + [self.target_col]],
            on=self.key_cols,
            how='left',
            suffixes=('_pred', '_true'),
        )
        score_all = self.metric(
            df_va_preds_true[self.target_col + '_true'].values,
            df_va_preds_true[self.target_col + '_pred'].values,
        )

        self.logger.result(f"memo: {self.memo}")
        self.logger.result_scores(self.run_name, scores)
        self.logger.result(f"all: {score_all}, mean: {np.mean(scores)}, std: {np.std(scores)}")
        self.logger.info(f"mean: {np.mean(scores)}, std: {np.std(scores)}")

        path_output = os.path.join(self.out_dir_name, 'va_pred.pkl')
        Util.dump_df_pickle(df_va_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end metric cv')

    def predict_fold(self, i_fold: Union[int, str]):
        """foldを指定してテストデータを予測する"""
        model = self.build_model(i_fold)
        model.load_model()

        df_te_pred = model.predict(self.df_test)

        if self.after_predict_process is not None:
            df_te_pred = self.after_predict_process(df_te_pred, self.target_col)

        return df_te_pred.sort_values(self.key_cols)

    def run_predict_cv(self) -> None:
        """CVでテストデータの予測を行い、fold平均を保存する"""
        self.logger.info(f'{self.run_name} - start prediction cv')
        te_preds = []

        df_te_preds = self.df_test.copy()[self.key_cols]
        for i_fold in tqdm(range(self.n_splits)):
            df_te_pred = self.predict_fold(i_fold)
            te_preds.append(df_te_pred[self.target_col].values)

        df_te_preds = df_te_preds.sort_values(self.key_cols)
        df_te_preds[self.target_col] = np.mean(te_preds, axis=0)

        path_output = os.path.join(self.out_dir_name, 'te_pred.pkl')
        Util.dump_df_pickle(df_te_preds, path_output)
        self.logger.info(f'output predict : {path_output}')

        self.logger.info(f'{self.run_name} - end prediction cv')

    def get_feature_importance_fold(self, i_fold):
        """foldを指定して特徴量重要度を取得する"""
        model = self.build_model(i_fold)
        model.load_model()
        return model.get_feature_importance()

    def plot_feature_importance_cv(self) -> None:
        """全foldの特徴量重要度の平均を棒グラフで保存する"""
        self.logger.info(f'{self.run_name} - start plot feature importance cv')

        list_df_fi = [self.get_feature_importance_fold(i) for i in range(self.n_splits)]
        list_importance = [df["importance"].values for df in list_df_fi]

        df = pd.DataFrame({
            "feature": list_df_fi[0]["feature"],
            "mean": np.mean(list_importance, axis=0),
            "std": np.std(list_importance, axis=0),
        })
        df['coef_of_var'] = (df['std'] / df['mean']).fillna(0)
        df = df.sort_values('mean', ascending=True).tail(100)

        fig = plt.figure(figsize=(40, 30))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title('feature importance gain')
        ax1.set_xlabel('feature importance mean & std')
        ax1.barh(df["feature"], df['mean'], label='mean', alpha=0.6)
        ax1.barh(df["feature"], df['std'], label='std', alpha=0.6)

        ax2 = ax1.twiny()
        ax2.plot(df['coef_of_var'], df["feature"], linewidth=1, color="crimson",
                 marker="o", markersize=8, label='coef_of_var')
        ax2.set_xlabel('Coefficient of variation')

        ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax2.legend(bbox_to_anchor=(1, 0.97), loc='upper right', borderaxespad=0.5, fontsize=12)
        ax1.grid(True)
        ax2.grid(False)
        plt.tick_params(labelsize=12)
        plt.tight_layout()

        path_output = os.path.join(self.out_dir_name, 'fi_gain.png')
        plt.savefig(path_output, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f'{self.run_name} - end plot feature importance cv')
