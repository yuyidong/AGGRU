import torch
from torch import nn
from torch.nn import functional as F


class AGRNN(nn.Module):
    def __init__(self, seq_len, num_nodes, hidden_size, dropout, num_blocks, adj=None):
        super(AGRNN, self).__init__()
        self.net = nn.Sequential()
        for i in range(num_blocks):
            self.net.add_module('Block{}'.format(i + 1), AGRNNBlock(seq_len, num_nodes, hidden_size, dropout, adj=adj))

    def forward(self, x):
        for block in self.net:
            x = block(x) + x
        return x


class AGRNNBlock(nn.Module):
    def __init__(self, seq_len, num_nodes, hidden_size, dropout, adj):
        super(AGRNNBlock, self).__init__()
        self.gru = GRU(num_nodes, hidden_size)
        self.agcn = AGCN(seq_len, num_nodes, hidden_size, adj)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.gru(x) + self.agcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        return x


class GRU(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(num_nodes, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_nodes)
        self.bn = nn.BatchNorm1d(num_nodes)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class AGCN(nn.Module):
    def __init__(self, seq_len, num_nodes, hidden_size, adj):
        super(AGCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = adj
        self.nodevec = nn.Parameter(torch.randn(num_nodes, 10))
        self.linear = nn.Linear(seq_len, hidden_size)
        self.fc = nn.Linear(hidden_size, seq_len)
        self.bn = nn.BatchNorm1d(num_nodes)
        self.sigmoid = nn.Sigmoid()
        self.sa = SelfAttention(10)

    def forward(self, x):
        if self.adj is not None:
            adp = self.adj
        else:
            # sa_nodevec = self.sa(self.nodevec)
            # adp = self.sigmoid(F.relu(torch.mm(sa_nodevec, sa_nodevec.T)))
            adp = self.sigmoid(F.relu(torch.mm(self.nodevec, self.nodevec.T)))
        support = self.calculate_laplacian_with_self_loop(adp, self.device)
        x = torch.einsum("ij,bjk->bik", support, x)
        x = F.relu(self.linear(x))
        x = self.fc(x)
        x = self.bn(x)
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


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(0, 1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values
