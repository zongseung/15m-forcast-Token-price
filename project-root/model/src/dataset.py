# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, in_len, out_len):
        raw_X = df[feature_cols].values
        raw_y = df["Close"].values.reshape(-1,1)

        self.scaler_X = MinMaxScaler().fit(raw_X)
        self.scaler_y = MinMaxScaler().fit(raw_y)

        Xs = self.scaler_X.transform(raw_X)
        ys = self.scaler_y.transform(raw_y).flatten()

        self.X = torch.tensor(Xs, dtype=torch.float32)
        self.y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)
        self.in_len, self.out_len = in_len, out_len

    def __len__(self):
        return len(self.X) - self.in_len - self.out_len + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.in_len]
        y_seq = self.y[idx+self.in_len:idx+self.in_len+self.out_len]
        y0    = self.y[idx+self.in_len-1].squeeze()
        return x_seq, y_seq, y0

    def get_scalers(self):
        return self.scaler_X, self.scaler_y
