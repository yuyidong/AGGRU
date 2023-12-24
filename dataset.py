import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY

    def __getitem__(self, index):
        input = self.dataX[index]
        label = self.dataY[index]
        return input, label

    def __len__(self):
        return self.dataX.shape[0]


class StandardScaler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean, self.std = None, None

    def transform(self, data):
        self.mean = torch.mean(data, dim=1, keepdim=True).to(self.device)
        self.std = torch.std(data, dim=1, keepdim=True).to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataOperator:
    def __init__(self, path, timestep, batch_size, num_nodes):
        dataX, dataY = self.load_dataset(path, timestep, num_nodes)
        len_train = int(0.7 * dataX.shape[0])
        len_valid = int(0.9 * dataX.shape[0]) - int(0.7 * dataX.shape[0])
        self.train_set = MyDataset(dataX[:len_train], dataY[:len_train])
        self.valid_set = MyDataset(dataX[len_train:len_train + len_valid], dataY[len_train:len_train + len_valid])
        self.test_set = MyDataset(dataX[len_train + len_valid:], dataY[len_train + len_valid:])

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size)
        self.valid_loader = DataLoader(self.valid_set, batch_size=len(self.valid_set))
        self.test_loader = DataLoader(self.test_set, batch_size=len(self.test_set))
        self.scaler = StandardScaler()
        self.adj = self.create_adj(path, num_nodes)

    @staticmethod
    def load_dataset(path, timestep, num_nodes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = pd.read_excel(path)
        for i in data.columns:
            data = data[data[i] > 0]
        data = data.values
        dataX, dataY = [], []
        for index in range(len(data) - timestep * 2):
            dataX.append(data[index:index + timestep, :num_nodes])
            dataY.append(data[index + timestep:index + timestep * 2, :num_nodes])
        dataX, dataY = np.array(dataX), np.array(dataY)
        dataX, dataY = torch.tensor(dataX).float().to(device), torch.tensor(dataY).float().to(device)
        return dataX, dataY

    @staticmethod
    def create_adj(path, num_nodes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = pd.read_excel(path)
        data = data.iloc[:, :num_nodes]
        for i in data.columns:
            data = data[data[i] > 0]
        correlation_matrix = data.corr()
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[0]):
                if correlation_matrix.iloc[i][j] < 0:
                    correlation_matrix.iloc[i][j] = 0
        return torch.tensor(correlation_matrix.values).float().to(device)
