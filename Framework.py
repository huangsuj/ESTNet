import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from functools import partial
import time
import random

random.seed(12)
np.random.seed(12)
torch.manual_seed(12)
torch.cuda.manual_seed(12)
torch.cuda.manual_seed_all(12)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(2)].unsqueeze(1).expand_as(x)

class PatchEmbedding_flow(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_flow, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = x.squeeze(-1).permute(0, 2, 1)
        if x.shape[-1] == 288:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = 288 // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len // gap, step=self.stride // gap)
            x = F.pad(x, (0, (self.patch_len - self.patch_len // gap)))
        x = self.value_embedding(x)
        x = x + self.position_encoding(x)
        x = x.permute(0, 2, 1, 3)
        return x

class PatchEmbedding_time(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, his):
        super(PatchEmbedding_time, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.minute_size = 1440 + 1
        self.daytime_embedding = nn.Embedding(self.minute_size, d_model//2)
        weekday_size = 7 + 1
        self.weekday_embedding = nn.Embedding(weekday_size, d_model//2)

    def forward(self, x):
        # do patching
        bs, ts, nn, dim = x.size()
        x = x.permute(0, 2, 3, 1).reshape(bs, -1, ts)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
        num_patch = x.shape[-2]
        x = x.reshape(bs, nn, dim, num_patch, -1).transpose(1, 3)
        x_tdh = x[:, :, 0, :, 0]
        x_dwh = x[:, :, 1, :, 0]
        x_tdp = x[:, :, 2, :, 0]
        x_dwp = x[:, :, 3, :, 0]

        x_tdh = self.daytime_embedding(x_tdh)
        x_dwh = self.weekday_embedding(x_dwh)
        x_tdp = self.daytime_embedding(x_tdp)
        x_dwp = self.weekday_embedding(x_dwp)
        x_th = torch.cat([x_tdh, x_dwh], dim=-1)
        x_tp = torch.cat([x_tdp, x_dwp], dim=-1)

        return x_th, x_tp

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        # index = list(range(0, seq_len))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index, modes

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, prob_drop, alpha):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Linear(out_dim, out_dim)
        self.dropout = prob_drop
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        h = x
        a = adj / d.view(-1, 1)
        gcn_out = self.fc1(torch.einsum('bdkt,nk->bdnt', h, a))
        # out = self.alpha*x + (1-self.alpha)*gcn_out
        out = gcn_out
        ho = self.mlp(out)
        return ho


class Attention(nn.Module):
    def __init__(
        self, args, dim, geo_num_heads=4, t_num_heads=4, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.geo_num_heads = geo_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        # self.adj_mx = adj_mx
        self.alpha = args.alpha
        self.beta = args.beta

        self.geo_ratio = 0.5
        self.t_ratio = 1 - self.geo_ratio

        self.GCN = GCN(dim, dim, proj_drop, alpha=self.alpha)
        self.act = nn.GELU()

        self.geo_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.geo_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.geo_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        self.t_q_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_k_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_v_conv = nn.Linear(dim, dim, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # get modes for queries and keys (& values) on frequency domain
        self.seq_len_ = args.num_nodes
        self.seq_len_t = args.input_window // 12
        self.mode_select_method = 'random'
        self.modes = 32
        self.modes_T = int(self.seq_len_t / 2)
        self.index_qkv, _ = get_frequency_modes(self.seq_len_, modes=self.modes, mode_select_method=self.mode_select_method)

        self.index_qkv_indices = [i for i, j in enumerate(self.index_qkv)]
        self.index_qkv_values = [j for i, j in enumerate(self.index_qkv)]
        self.index_qkv_t, self.i_modes = get_frequency_modes(self.seq_len_t, modes=self.modes_T, mode_select_method=self.mode_select_method)

        self.modes_T = self.i_modes
        self.index_qkv_t_indices = [i for i, j in enumerate(self.index_qkv_t)]
        self.index_qkv_t_values = [j for i, j in enumerate(self.index_qkv_t)]

        self.weights_Q = nn.Parameter(torch.randn(self.modes, self.modes-1, self.head_dim, dtype=torch.float))
        self.weights_att = nn.Parameter(torch.randn(self.modes, self.head_dim - 1, self.modes, dtype=torch.float))
        self.weights_Q_t = nn.Parameter(torch.randn(self.modes_T, self.modes_T - 1, self.head_dim, dtype=torch.float))

    def forward(self, x, adj):
        gcn_outputs = self.GCN(x, adj)
        sp_x = self._gc(x, 'sp')
        te_x = self._gc(x, 'te')
        x = gcn_outputs + sp_x + te_x
        x = self.proj_drop(x)

        return x

    def _gc(self, inputs, flag='sp'):
        B, T, N, D = inputs.shape
        if flag == 'sp':
            ### 计算Q,K,V
            h = self.geo_num_heads
            geo_q = self.geo_q_conv(inputs)  # [N, D]
            geo_k = self.geo_k_conv(inputs)  # [B, T, h, N, D2]
            geo_v = self.geo_v_conv(inputs)  # [N, D2]

            geo_q_ = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)
            geo_k_ = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)
            geo_v_ = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)  # [B, T, h, D, N]
            geo_q_ = geo_q_ / torch.norm(geo_q, p=2, keepdim=True)
            geo_k_ = geo_k_ / torch.norm(geo_k, p=2, keepdim=True)

            geo_q_ft = torch.fft.rfft(geo_q_, dim=-1)  # [B, T, h, D, N//2+1]
            geo_q = torch.zeros(B, T, h, self.head_dim, len(self.index_qkv), device=geo_q.device,
                                dtype=torch.cfloat)
            geo_q.real[:, :, :, :, self.index_qkv_indices] = geo_q_ft.real[:, :, :, :, self.index_qkv_values]
            geo_q.imag[:, :, :, :, self.index_qkv_indices] = geo_q_ft.imag[:, :, :, :, self.index_qkv_values]
            geo_q = geo_q.permute(0, 1, 2, 4, 3).unsqueeze(4)  # [B, T, h, M, 1, D]
            weights_Q_expand = self.weights_Q.expand(B, T, h, -1, -1, -1)
            geo_q_cat = torch.cat((geo_q, weights_Q_expand), dim=4)  # [B, T, h, M, M, D]

            geo_k_ft = torch.fft.rfft(geo_k_)
            geo_k = torch.zeros(B, T, h, self.head_dim, len(self.index_qkv), device=geo_q.device,
                                dtype=torch.cfloat)
            geo_k.real[:, :, :, :, self.index_qkv_indices] = geo_k_ft.real[:, :, :, :, self.index_qkv_values]
            geo_k.imag[:, :, :, :, self.index_qkv_indices] = geo_k_ft.imag[:, :, :, :, self.index_qkv_values]
            geo_k = geo_k.permute(0, 1, 2, 4, 3).unsqueeze(3).repeat(1, 1, 1, geo_q.shape[3], 1,
                                                                     1)  # [B, T, h, M, M, D]
            att_frq = torch.einsum("bthnij,bthnij->bthnij", geo_k,
                                   torch.conj(geo_q_cat)) * self.scale  # [B, T, h, M, M, D]

            att_frq = torch.softmax(abs(att_frq), dim=-2)  # [B, N, h, M, M, D]
            att_frq = torch.complex(att_frq, torch.zeros_like(att_frq)).permute(0, 1, 2, 3, 5, 4)  # [B, N, h, M, D, M]

            geo_v_ft = torch.fft.rfft(geo_v_)
            geo_v = torch.zeros(B, T, h, self.head_dim, len(self.index_qkv), device=geo_q.device,
                                dtype=torch.cfloat)
            geo_v.real[:, :, :, :, self.index_qkv_indices] = geo_v_ft.real[:, :, :, :, self.index_qkv_values]
            geo_v.imag[:, :, :, :, self.index_qkv_indices] = geo_v_ft.imag[:, :, :, :, self.index_qkv_values]
            geo_v = geo_v.unsqueeze(-3).repeat(1, 1, 1, geo_q.shape[3], 1, 1)  # [B, T, h, M, D, M]
            output = torch.einsum("bthnij,bthnij->bthnij", geo_v, torch.conj(att_frq)).mean(dim=-3) # [B, T, h, D, M]
            ## padding
            out_ft = torch.zeros(B, T, h, self.head_dim, N // 2 + 1, device=geo_q.device, dtype=torch.cfloat)
            out_ft.real[:, :, :, :, self.index_qkv_values] = output.real[:, :, :, :, self.index_qkv_indices]
            out_ft.imag[:, :, :, :, self.index_qkv_values] = output.imag[:, :, :, :, self.index_qkv_indices]
            output = torch.fft.irfft(out_ft, n=N)  # [B, T, h, D, N]
            sp_x = output.permute(0, 1, 4, 2, 3).reshape(B, T, N, D)  # [B, T, N, D]

            return sp_x
        else:
            ### 计算Q,K,V
            h = self.t_num_heads
            t_q = self.t_q_conv(inputs)
            t_k = self.t_k_conv(inputs)
            t_v = self.t_v_conv(inputs)

            t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)  # [B, N, h, D, T]
            t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)
            t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 4, 2)  # [B, N, h, D, T]
            t_q = t_q / torch.norm(t_q, p=2, keepdim=True)
            t_k = t_k / torch.norm(t_k, p=2, keepdim=True)

            t_q_ft = torch.fft.rfft(t_q)  # [B, N, h, D, T//2+1]
            t_q_ = torch.zeros(B, N, h, self.head_dim, len(self.index_qkv_t), device=t_q.device, dtype=torch.cfloat)
            t_q_.real[:, :, :, :, self.index_qkv_t_indices] = t_q_ft.real[:, :, :, :, self.index_qkv_t_values]
            t_q_.imag[:, :, :, :, self.index_qkv_t_indices] = t_q_ft.imag[:, :, :, :, self.index_qkv_t_values]
            t_q_ = t_q_.permute(0, 1, 2, 4, 3).unsqueeze(4)  # [B, N, h, M, 1, D]
            weights_Q_expand = self.weights_Q_t.expand(B, N, h, -1, -1, -1)
            t_q_cat = torch.cat((t_q_, weights_Q_expand), dim=4)

            t_k_ft = torch.fft.rfft(t_k)
            t_k_ = torch.zeros(B, N, h, self.head_dim, len(self.index_qkv_t), device=t_k.device, dtype=torch.cfloat)
            t_k_.real[:, :, :, :, self.index_qkv_t_indices] = t_k_ft.real[:, :, :, :, self.index_qkv_t_values]
            t_k_.imag[:, :, :, :, self.index_qkv_t_indices] = t_k_ft.imag[:, :, :, :, self.index_qkv_t_values]
            t_k_ = t_k_.permute(0, 1, 2, 4, 3).unsqueeze(4).repeat(1, 1, 1, 1, t_q_.shape[3],
                                                                   1)  # [B, N, h, M, M, D]
            att_frq = torch.einsum("bthnij,bthnij->bthnij", t_k_, torch.conj(t_q_cat)) * self.scale
            att_frq = att_frq.permute(0, 1, 2, 5, 3, 4)  # [B, N, h, D, M, M]
            mask = torch.tril(torch.ones(len(self.index_qkv_t), len(self.index_qkv_t))).bool().to(att_frq.device)
            att_mask = att_frq * mask  # [B, N, h, D, M, M]
            att_frq = torch.softmax(abs(att_mask), dim=-1)  # [B, N, h, D, M, M]
            att_frq = torch.complex(att_frq,
                                    torch.zeros_like(att_frq))  # .permute(0, 1, 2, 4, 3, 5)  # [B, N, h, M, D, M]
            # att_frq = self.t_attn_drop(att_frq)

            t_v_ft = torch.fft.rfft(t_v)  # 扩展卷积核到输入矩阵大小
            t_v_ = torch.zeros(B, N, h, self.head_dim, len(self.index_qkv_t), device=t_v.device, dtype=torch.cfloat)
            t_v_.real[:, :, :, :, self.index_qkv_t_indices] = t_v_ft.real[:, :, :, :, self.index_qkv_t_values]
            t_v_.imag[:, :, :, :, self.index_qkv_t_indices] = t_v_ft.imag[:, :, :, :, self.index_qkv_t_values]
            t_v_ = t_v_.unsqueeze(-1).repeat(1, 1, 1, 1, 1, t_q_.shape[3])  # [B, N, h, D, M, M]

            output = torch.einsum("bthnij,bthnij->bthnij", t_v_, torch.conj(att_frq)).mean(
                dim=-1)  # [B, N, h, D, M]
            ## padding
            out_ft = torch.zeros(B, N, h, self.head_dim, T // 2 + 1, device=t_k_.device, dtype=torch.cfloat)
            out_ft.real[:, :, :, :, self.index_qkv_t_values] = output.real[:, :, :, :, self.index_qkv_t_indices]
            out_ft.imag[:, :, :, :, self.index_qkv_t_values] = output.imag[:, :, :, :, self.index_qkv_t_indices]
            output = torch.fft.irfft(out_ft, n=T)  # [B, N, h, D, T]
            t_x = output.permute(0, 4, 1, 2, 3).reshape(B, T, N, D)  # [B, T, N, D]

            return t_x

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class STEncoderBlock(nn.Module):

    def __init__(
        self, args, dim, geo_num_heads=4, t_num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        self.norm1 = LlamaRMSNorm(dim)
        self.norm2 = LlamaRMSNorm(dim)
        self.st_attn = Attention(
            args, dim, geo_num_heads=geo_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.alpha = args.alpha
        self.mlp = FeedForward(hidden_size=dim, intermediate_size=mlp_hidden_dim)

    def forward(self, x, adj):
        x = self.norm1((1-3*self.alpha) * x + self.alpha * (self.drop_path(self.st_attn(x, adj))))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc

class ESTNet(nn.Module):
    def __init__(self, args, device, dim_in):
        super(ESTNet, self).__init__()
        self.feature_dim = dim_in
        # self.adj_mx = args.adj_mx

        self.embed_dim = args.embed_dim
        self.skip_dim = args.skip_dim
        self.geo_num_heads = args.geo_num_heads
        self.t_num_heads = args.t_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.qkv_bias = args.qkv_bias
        self.drop = args.drop
        self.attn_drop = args.attn_drop
        self.drop_path = args.drop_path
        self.enc_depth = args.layers
        self.type_short_path = args.type_short_path
        self.lape_dim = args.lape_dim
        self.output_dim = dim_in
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.add_time_in_day = args.add_time_in_day
        self.add_day_in_week = args.add_day_in_week
        self.device = device
        self.adj_mx_dict = args.adj_mx_dict

        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window) ## 为了改变输入数据的形状？？
        enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.enc_depth)]
        self.ourmodel = nn.ModuleList([
            STEncoderBlock(
                args=args, dim=self.embed_dim,
                geo_num_heads=self.geo_num_heads, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, output_dim=self.output_dim,
            ) for i in range(self.enc_depth)
        ])


        self.patch_embedding_flow = PatchEmbedding_flow(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.patch_embedding_time = PatchEmbedding_time(
            self.embed_dim, patch_len=12, stride=12, padding=0, his=args.input_window)
        self.spatial_embedding = LaplacianPE(self.lape_dim, self.embed_dim)
        self.flatten = nn.Flatten(start_dim=-2)

        if args.dataset_use[0] in ['PEMS_BAY', 'CAD3', 'CAD5', 'TrafficSH', 'TrafficJN', 'CHI_TAXI', 'NYC_TAXI']:
            self.linear = nn.Linear(1536, self.output_window)
        elif args.dataset_use[0] == 'GBA':
            self.linear = nn.Linear(512, self.output_window)
        else:
            self.linear = nn.Linear(1536, self.output_window)

    def forward(self, input, lbls, select_dataset):

        bs, time_steps, num_nodes, num_feas = input.size()
        x = input
        x_in = x[..., :self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_in /= stdev

        # Patch Embedding
        enc = self.patch_embedding_flow(x_in)

        # adj
        adj = self.adj_mx_dict[select_dataset].to(self.device)

        # Spatio-Temporal Dependencies Modeling
        for i, encoder_block in enumerate(self.ourmodel):
            enc = encoder_block(enc, adj)

        # Prediction head
        skip = enc.permute(0, 2, 3, 1).contiguous()
        skip = self.flatten(skip)
        skip = self.linear(skip).transpose(1, 2).unsqueeze(-1)
        skip = skip[:, :time_steps, :, :]

        # DeIN
        skip = skip * stdev
        skip = skip + means

        return skip