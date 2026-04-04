import datetime
import logging
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath('..'))
from configs.config import *


class Util:

    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def jump_json(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(value, f, indent=4)

    @classmethod
    def dump_df_pickle(cls, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)

    @classmethod
    def load_feature(cls, file_name):
        file_name = file_name + ".pkl"
        return pd.read_pickle(os.path.join(DIR_FEATURE, file_name))


class Logger:

    def __init__(self, path):
        self.general_logger = logging.getLogger(os.path.join(path, 'general'))
        self.result_logger = logging.getLogger(os.path.join(path, 'result'))
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, 'general.log'))
        file_result_handler = logging.FileHandler(os.path.join(path, 'result.log'))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        dic = dict()
        dic['run_name'] = run_name
        dic['score_mean'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class Submission:

    @classmethod
    def create_submission(cls, run_name, dir_name, preds):
        logger = Logger(dir_name)
        logger.info(f'{run_name} - start create submission')

        submission = pd.read_csv(os.path.join(DIR_INPUT, FILE_SAMPLE_SUBMISSION))
        submission.loc[:, TARGET_COL] = preds.iloc[:, 0]
        submission.to_csv(os.path.join(DIR_SUBMISSIONS, f'{run_name}_submission.csv'), index=False, header=True)

        logger.info(f'{run_name} - end create submission')


class Metric:

    @classmethod
    def my_metric(cls, y_true, y_pred):
        """評価関数（ROC-AUC）"""
        result = roc_auc_score(y_true, y_pred)
        return result
