import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def calculate_normalized_laplacian(adj):
    import scipy.sparse as sp
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

class GCN(nn.Module):
    def __init__(self, embed_dim, adj_mx, device, qkv_hid=1):
        super(GCN, self).__init__()
        self.adj_mx = adj_mx
        self.input_dim = embed_dim
        self.device = device
        self.hid = qkv_hid
        support = calculate_normalized_laplacian(self.adj_mx)
        self.normalized_adj = self._build_sparse_matrix(support, self.device)
        self.init_params()

    def init_params(self, bias_start=0.0):
        # input_size = self.input_dim + self.num_units
        input_size = self.input_dim
        weight_0 = torch.nn.Parameter(torch.empty((input_size, self.hid), device=self.device))
        bias_0 = torch.nn.Parameter(torch.empty(self.hid, device=self.device))
        # weight_1 = torch.nn.Parameter(torch.empty((input_size, self.hid), device=self.device))
        # bias_1 = torch.nn.Parameter(torch.empty(self.hid, device=self.device))

        torch.nn.init.xavier_normal_(weight_0)
        # torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        # torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        # self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        # self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0}
        # self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0}
        # self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

    @staticmethod
    # @staticmethod是把函数嵌入到类中的一种方式，函数就属于类，同时表明函数不需要访问这个类
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs):
        # [B,T,N,D]
        batch_size, T, N, D = inputs.shape

        x = inputs  # [B, T, N, D]
        x0 = x.permute(2, 3, 0, 1)  # (num_nodes, dim, batch, T)
        x0 = x0.reshape(shape=(N, -1))  # (num_nodes, dim * batch * T)

        x1 = torch.sparse.mm(self.normalized_adj.float().to(x0.device), x0.float())  # A * X

        ### xw+b
        x1 = x1.reshape(shape=(N, D, batch_size, T))
        x1 = x1.permute(2, 3, 0, 1)  # (batch_size, T, self.num_nodes, D)
        x1 = x1.reshape(shape=(-1, D))  # (batch_size * T * self.num_nodes, input_size)

        weights = self.weigts[(D, self.hid)]
        x1 = torch.matmul(x1, weights.to(x1.device))  # (batch_size * self.num_nodes, output_size)

        biases = self.biases[(self.hid,)]
        x1 += biases.to(x1.device)

        x1 = x1.reshape(shape=(batch_size, T, N, self.hid))
        return x1


