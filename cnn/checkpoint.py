from pathlib import Path
from .history import History
from .model import Model
from pathlib import Path
import numpy as np

class Checkpoint:
    def __init__(self, model:Model, history:History, savedir:Path, metric:str, inverse=False) -> None:
        self.model = model
        self.history = history
        self.metric = metric
        self.inverse = inverse
        self.best_score = None
        self.checkpoint_path = savedir / 'checkpoint.pckl'

    def step(self):
        score = self.history.res[self.metric][-1]
        if (self.best_score is None ) or \
            (not self.inverse and score >= self.best_score) or \
            (self.inverse and score <= self.best_score):
            # new best model
            self.best_score = score
            self.model.save(self.checkpoint_path)


        
    def get_best_model(self) -> Model:
        return Model.load(self.checkpoint_path)
