# AUTOGENERATED! DO NOT EDIT! File to edit: data.ipynb (unless otherwise specified).

__all__ = ['TestData', 'Data', 'TestDataFrame', 'DataFrame', 'create_db', 'create_test_dl']

# Cell
import torch
from torch.utils.data import Dataset, DataLoader
from fastai.data_block import DataBunch, DatasetType

import numpy as np

# Cell
class TestData(Dataset):
    """
    Create pyTorch Dataset
    parameters:
    - t: time elapsed
    - b: (optional) breakpoints where the hazard is different to previous segment of time.
    - x: (optional) features
    """
    def __init__(self, t, b=None, x=None):
        super().__init__()
        assert isinstance(b, np.ndarray) or isinstance(b, list) or b is None\
                , "Breakpoints need to be a list"
        self.t, self.b, self.x = t, b, x

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        time = torch.Tensor([self.t[i]])

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
    def __init__(self, t, e, b=None, x=None):
        super().__init__(t, b, x)
        self.e = e

    def __getitem__(self, i):
        x_ = super().__getitem__(i)
        e = torch.Tensor([self.e[i]])
        return x_, e

# Cell
class TestDataFrame(TestData):
    """
    Wrapper around Data Class that takes in a dataframe instead
    parameters:
    - df: dataframe. **Must have t (time) and e (event) columns, other cols optional.
    - b: breakpoints of time (optional)
    """
    def __init__(self, df, b=None):
        t = df['t'].values
        remainder = list(set(df.columns) - set(['t', 'e']))
        x = df[remainder].values
        if x.shape[1] == 0:
            x = None
        super().__init__(t, b, x)

# Cell
class DataFrame(Data):
    """
    Wrapper around Data Class that takes in a dataframe instead
    parameters:
    - df: dataframe. **Must have t (time) and e (event) columns, other cols optional.
    - b: breakpoints of time (optional)
    """
    def __init__(self, df, b=None):
        t = df['t'].values
        e = df['e'].values
        x = df.drop(['t', 'e'], axis=1).values
        if x.shape[1] == 0:
            x = None
        super().__init__(t, e, b, x)

# Cell
def create_db(df, b=None, train_p=0.8, bs=128):
    """
    Take dataframe and split into train, test, val (optional)
    and convert to Fastai databunch

    parameters:
    - df: pandas dataframe
    - b: breakpoints of time (optional)
    - train_p: training percentage
    - bs: batch size
    - test_p: proportion of whats left over after taking out train set
    """
    df.reset_index(drop=True, inplace=True)

    train_len = int(train_p*len(df))
    train_ds = DataFrame(df.iloc[:train_len], b)
    val_ds = DataFrame(df.iloc[train_len:], b)

    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, bs, drop_last=False)
    db = DataBunch(train_dl, val_dl)

    return db

def create_test_dl(df, b=None, bs=128):
    df.reset_index(drop=True, inplace=True)
    test_ds = TestDataFrame(df, b)
    test_dl = DataLoader(test_ds, bs, drop_last=False)
    return test_dl