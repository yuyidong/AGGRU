import torch
from torch import nn
from torch.nn import functional as F


class GRU(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(num_nodes, hidden_size)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return x


class FNN(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(FNN, self).__init__()
        self.linear1 = nn.Linear(num_nodes, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(LSTM, self).__init__()
        self.gru = nn.GRU(num_nodes, hidden_size)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(num_nodes, hidden_size)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class CNN_GRU(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(CNN_GRU, self).__init__()
        self.cnn = nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(num_nodes, hidden_size)
        self.fc = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, seq_len, hidden_size, adj):
        super(GCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = adj
        self.linear = nn.Linear(seq_len, hidden_size)
        self.fc = nn.Linear(hidden_size, seq_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        adp = self.adj
        support = self.calculate_laplacian_with_self_loop(adp, self.device)
        x = torch.einsum("ij,bjk->bik", support, x)
        x = F.relu(self.linear(x))
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def calculate_laplacian_with_self_loop(matrix, device):
        matrix = matrix + torch.eye(matrix.size(0)).to(device)
        row_sum = matrix.sum(1)
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        normalized_laplacian = (matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt))
        return normalized_laplacian
