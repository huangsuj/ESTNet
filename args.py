import numpy as np
import configparser
from lib.predifineGraph import get_adjacency_matrix, cal_lape
import torch
import copy
import os
def load_rel_2(adj_mx, args, dataset):
    directory = f'../data/{dataset}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f'{dataset}_sh_mx.npy'
    cache_path = os.path.join(directory, filename)
    sh_mx = adj_mx.copy()
    if args.type_short_path == 'hop':
        if not os.path.exists(cache_path):
            sh_mx[sh_mx > 0] = 1
            sh_mx[sh_mx == 0] = 511
            np.save(cache_path, sh_mx)
        sh_mx = np.load(cache_path)
    return sh_mx

def parse_args(parser, args):
    # get configuration

    args_predictor, _ = parser.parse_known_args()
    DATASET = args_predictor.dataset_use[0]

    sh_mx_dict = {}
    lpls_dict = {}
    adj_mx_dict = {}
    for dataset_select in (args_predictor.dataset_use):
        if dataset_select in ['CAD3', 'CAD5']:
            args_predictor.filepath = './data/' + dataset_select + '/'
        else:
            args_predictor.filepath = './data/' + dataset_select + '/'
        args_predictor.filename = dataset_select
        if dataset_select == 'PEMS08' or dataset_select == 'PEMS04' or dataset_select == 'PEMS07':
            A, Distance = get_adjacency_matrix(
                distance_df_filename=args_predictor.filepath + dataset_select + '.csv',
                num_of_vertices=args.num_nodes_dict[dataset_select])
            A = A + np.eye(A.shape[0])
        else:
            A = np.load(args_predictor.filepath + f'{dataset_select}_rn_adj.npy')
        sh_mx_dict[dataset_select] = torch.FloatTensor(load_rel_2(A, args_predictor, dataset_select))
        lpls_dict[dataset_select] = torch.FloatTensor(cal_lape(copy.deepcopy(A)))
        d = np.sum(A, axis=1)
        sinvD = np.sqrt(np.mat(np.diag(d)).I)
        A_mx = np.mat(np.identity(A.shape[0]) + sinvD * A * sinvD)
        adj_mx_dict[dataset_select] = torch.FloatTensor(copy.deepcopy(A_mx))

    args_predictor.adj_mx_dict = adj_mx_dict
    args_predictor.sd_mx = None
    args_predictor.sh_mx_dict = sh_mx_dict
    args_predictor.lap_mx_dict = lpls_dict

    return args_predictor