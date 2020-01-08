# AUTOGENERATED! DO NOT EDIT! File to edit: data.ipynb (unless otherwise specified).

__all__ = ['Data', 'DataFrame', 'create_db']

# Cell
import torch
from torch.utils.data import Dataset, DataLoader
from fastai.data_block import DataBunch, DatasetType

import numpy as np

# Cell
class Data(Dataset):
    """
    Create pyTorch Dataset
    """
    def __init__(self, t, e, b=None, x=None):
        """
        parameters:
        - x: features
        - t: time elapsed
        - e: (death) event observed. 1 if observed, 0 otherwise.
        - b: breakpoints where the hazard is different to previous segment of time.
        """
        super().__init__()
        assert isinstance(b, np.ndarray) or isinstance(b, list) or b is None\
                , "Breakpoints need to be a list"
        self.x, self.t, self.e, self.b = x, t, e, b

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        time = torch.Tensor([self.t[i]])
        e = torch.Tensor([self.e[i]])

        if self.b is None:
            x_ = (time,)
        else:
            t_section = torch.LongTensor([np.searchsorted(self.b, self.t[i])])
            x_ = (time, t_section.squeeze())

        if self.x is not None:
            x = torch.Tensor(self.x[i])
            x_ = (x,) + x_

        return x_, e

# Cell
class DataFrame(Data):
    def __init__(self, df, b=None):
        t = df['t'].values
        e = df['e'].values
        x = df.drop(['t', 'e'], axis=1).values
        if x.shape[1] == 0:
            x = None
        super().__init__(t, e, b, x)

# Cell
def create_db(df, b=None, train_p=0.8, bs=128, test_ds=False, test_p=0.2):
    """
    Take dataframe and split into train, test, val (optional)
    and convert to Fastai databunch

    parameters:
    - df: pandas dataframe
    - b: breakpoints of time (optional)
    - train_p: training percentage
    - bs: batch size
    - test_ds: whether to split into test set
    - test_p: proportion of whats left over after taking out train set
    """
    df.reset_index(drop=True, inplace=True)
    if test_ds:
        train_len = int((1-test_p)*len(df))
        test_ds = DataFrame(df.iloc[train_len:], b)
        df = df.iloc[:train_len]
        test_dl = DataLoader(test_ds, bs=bs)

    train_len = int(train_p*len(df))
    train_ds = DataFrame(df.iloc[:train_len], b)
    val_ds = DataFrame(df.iloc[train_len:], b)

    bs = min(bs, len(train_ds))
    val_bs = min(bs, len(val_ds))
    train_db = DataBunch.create(train_ds, val_ds, bs=bs, val_bs=val_bs)

    if test_ds is not False:
        return train_db, test_dl
    else:
        return train_db