from operator import invert
from .history import History
import numpy as np

class EarlyStopping:
    def __init__(self, history:History, metric:str, patience=5, inverse=True) -> None:
        self.history = history
        self.patience = patience
        self.metric = metric
        self.inverse = inverse
        
    def step(self):
        metric_values = self.history.res[self.metric]
        if len(metric_values) < self.patience + 1:
            return True
        
        # window of patience + 1
        metric_values = metric_values[-(self.patience + 1):]
        first = metric_values[0]
        after = metric_values[1:]
        if (self.inverse and first <= min(after)) or (not self.inverse and first >= max(after)):
            # break
            return False
        # continue
        return True
        


