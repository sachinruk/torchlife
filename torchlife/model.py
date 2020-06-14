# AUTOGENERATED! DO NOT EDIT! File to edit: model.ipynb (unless otherwise specified).

__all__ = ['Model']

# Cell
from .models.km import KaplanMeier
from .models.ph import PieceWiseHazard
from .models.cox import ProportionalHazard

from .data import create_db, create_test_dl

from .losses import *

from fastai.basics import Learner

# Cell
class Model:
    def __init__(self, model:str, model_args:dict=None, breakpoints:list=None,
                 bs:int=128, epochs:int=20, lr:float=1, beta:float=0):
        self.model = __text2model__[model](**model_args)
        self.loss = __text2loss__[model]
        self.breakpoints = breakpoints
        self.bs, self.epochs, self.lr, self.beta = bs, epochs, lr, beta

    def lr_find(df):
        db = create_db(df, self.breakpoints)
        learner = Learner(db, model, loss_func=hazard_loss)
        learner.lr_find(wd=self.beta)
        learner.recorder.plot()

    def fit(df):
        if hasattr(self.model, 'fit'):
            self.model.fit(df)
        else:
            db = create_db(df, self.breakpoints)
            learner = Learner(db, model, loss_func=hazard_loss)
            learner.fit(self.epochs, lr=self.lr, wd=self.beta)

    def predict(df):
        test_dl = create_test_dl(df)
        preds = []
        for x in test_dl:
            preds.append(self.model(x))
        return preds

    def plot_survival():
        self.model.plot_survival_function()