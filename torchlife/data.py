# AUTOGENERATED! DO NOT EDIT! File to edit: 80_data.ipynb (unless otherwise specified).

__all__ = ['TestData', 'Data', 'TestDataFrame', 'DataFrame', 'create_db', 'create_test_dl', 'get_breakpoints']

# Cell
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fastai.data_block import DataBunch, DatasetType
from pandas import DataFrame
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

# Cell
class TestData(Dataset):
    """
    Create pyTorch Dataset
    parameters:
    - t: time elapsed
    - b: (optional) breakpoints where the hazard is different to previous segment of time.
    **Must include 0 as first element and the maximum time as last element**
    - x: (optional) features
    """
    def __init__(self, t:np.array, b:Optional[np.array]=None, x:Optional[np.array]=None,
                 t_scaler:MaxAbsScaler=None, x_scaler:StandardScaler=None) -> None:
        super().__init__()
        self.t, self.b, self.x = t, b, x
        self.t_scaler = t_scaler
        self.x_scaler = x_scaler
        if len(t.shape) == 1:
            self.t = t[:,None]

        if t_scaler:
            self.t_scaler = t_scaler
            self.t = self.t_scaler.transform(self.t)
        else:
            self.t_scaler = MaxAbsScaler()
            self.t = self.t_scaler.fit_transform(self.t)

        if b is not None:
            b = b[1:-1]
            if len(b.shape) == 1:
                b = b[:,None]
            if t_scaler:
                self.b = t_scaler.transform(b).squeeze()
            else:
                self.b = self.t_scaler.transform(b).squeeze()

        if x is not None:
            if len(x.shape) == 1:
                self.x = x[None, :]
            if x_scaler:
                self.x_scaler = x_scaler
                self.x = self.x_scaler.transform(self.x)
            else:
                self.x_scaler = StandardScaler()
                self.x = self.x_scaler.fit_transform(self.x)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i:int) -> Tuple:
        time = torch.Tensor(self.t[i])

        if self.b is None:
            x_ = (time,)
        else:
            t_section = torch.LongTensor([np.searchsorted(self.b, self.t[i])])
            x_ = (time, t_section.squeeze())

        if self.x is not None:
            x = torch.Tensor(self.x[i])
            x_ = x_ + (x,)

        return x_

# Cell
class Data(TestData):
    """
    Create pyTorch Dataset
    parameters:
    - t: time elapsed
    - e: (death) event observed. 1 if observed, 0 otherwise.
    - b: (optional) breakpoints where the hazard is different to previous segment of time.
    - x: (optional) features
    """
    def __init__(self, t:np.array, e:np.array, b:Optional[np.array]=None, x:Optional[np.array]=None,
                t_scaler:MaxAbsScaler=None, x_scaler:StandardScaler=None) -> None:
        super().__init__(t, b, x, t_scaler, x_scaler)
        self.e = e
        if len(e.shape) == 1:
            self.e = e[:,None]

    def __getitem__(self, i) -> Tuple:
        x_ = super().__getitem__(i)
        e = torch.Tensor(self.e[i])
        return x_, e

# Cell
class TestDataFrame(TestData):
    """
    Wrapper around Data Class that takes in a dataframe instead
    parameters:
    - df: dataframe. **Must have t (time) and e (event) columns, other cols optional.
    - b: breakpoints of time (optional)
    """
    def __init__(self, df:DataFrame, b:Optional[np.array]=None,
                 t_scaler:MaxAbsScaler=None, x_scaler:StandardScaler=None) -> None:
        t = df['t'].values
        remainder = list(set(df.columns) - set(['t', 'e']))
        x = df[remainder].values
        if x.shape[1] == 0:
            x = None
        super().__init__(t, b, x, t_scaler, x_scaler)

# Cell
class DataFrame(Data):
    """
    Wrapper around Data Class that takes in a dataframe instead
    parameters:
    - df: dataframe. **Must have t (time) and e (event) columns, other cols optional.
    - b: breakpoints of time (optional)
    """
    def __init__(self, df:DataFrame, b:Optional[np.array]=None,
                t_scaler:MaxAbsScaler=None, x_scaler:StandardScaler=None) -> None:
        t = df['t'].values
        e = df['e'].values
        x = df.drop(['t', 'e'], axis=1).values
        if x.shape[1] == 0:
            x = None
        super().__init__(t, e, b, x, t_scaler, x_scaler)

# Cell
def create_db(df:pd.DataFrame, b:Optional[np.array]=None, train_p:float=0.8, bs:int=128)\
    -> Tuple[DataBunch, MaxAbsScaler, StandardScaler]:
    """
    Take dataframe and split into train, test, val (optional)
    and convert to Fastai databunch

    parameters:
    - df: pandas dataframe
    - b: breakpoints of time (optional)
    - train_p: training percentage
    - bs: batch size
    """
    df.reset_index(drop=True, inplace=True)

    train_len = int(train_p*len(df))

    train_ds = DataFrame(df.iloc[:train_len], b)
    val_ds = DataFrame(df.iloc[train_len:], b, train_ds.t_scaler, train_ds.x_scaler)

    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, bs, drop_last=False)
    db = DataBunch(train_dl, val_dl)

    return db, train_ds.t_scaler, train_ds.x_scaler

def create_test_dl(df:pd.DataFrame, t_scaler:MaxAbsScaler, b:Optional[np.array]=None, bs:int=128,
                  x_scaler:StandardScaler=None) -> DataLoader:
    """
    Take dataframe and return a pytorch dataloader.
    parameters:
    - df: pandas dataframe
    - b: breakpoints of time (optional)
    - bs: batch size
    """
    df.reset_index(drop=True, inplace=True)
    test_ds = TestDataFrame(df, b, t_scaler, x_scaler)
    test_dl = DataLoader(test_ds, bs, shuffle=False, drop_last=False)
    return test_dl

# Cell
def get_breakpoints(df:DataFrame, percentiles:list=[20, 40, 60, 80]) -> np.array:
    """
    Gives the times at which death events occur at given percentile
    parameters:
    df - must contain columns 't' (time) and 'e' (death event)
    percentiles - list of percentages at which breakpoints occur (do not include 0 and 100)
    """
    event_times = df.loc[df['e']==1, 't'].values
    breakpoints = np.percentile(event_times, percentiles)
    breakpoints = np.array([0] + breakpoints.tolist() + [df['t'].max()])

    return breakpoints