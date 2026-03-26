import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):
    """model_xxのスーパークラス。abcモジュールにより抽象メソッドを定義"""

    def __init__(self,
                 run_fold_name: str,
                 params,
                 logger,
                 ) -> None:
        self.run_fold_name = run_fold_name
        self.params = params
        self.logger = logger

    @abstractmethod
    def train(self,
              tr_x: pd.DataFrame,
              tr_y: pd.Series,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None,
              ) -> None:
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def save_model(self) -> None:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass
