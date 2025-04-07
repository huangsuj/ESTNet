import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn
import argparse
from lib.TrainInits import init_seed
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch, huber_loss
from lib.Params_predictor import get_predictor_params
from lib.data_process import define_dataloder
import random
import numpy as np
# *************************************************************************#
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_nodes_dict = {'PEMS03':'358', 'PEMS04': 307, 'PEMS08': 170, 'PEMS07': 883, 'CD_DIDI': 524, 'SZ_DIDI': 627,
                  'METR_LA': 207, 'PEMS_BAY': 325, 'PEMS07M': 228, 'NYC_TAXI': 263, 'CHI_TAXI': 77, 'NYC_BIKE-3': 540,
                  'CAD3': 480, 'CAD4-1': 621, 'CAD4-2': 610, 'CAD4-3': 593, 'CAD4-4': 528, 'CAD5': 211,
                  'CAD7-1': 666, 'CAD7-2': 634, 'CAD7-3': 559, 'CAD8-1': 510, 'CAD8-2': 512, 'CAD12-1': 453, 'CAD12-2': 500,
                  'TrafficZZ': 676, 'TrafficCD': 728, 'TrafficHZ': 672, 'TrafficJN': 576, 'TrafficSH': 896,
                  'CA': 8600, 'GBA':2352, 'GLA': 3834,
                  }

def parameter_parser(item, t):
    args = argparse.ArgumentParser(prefix_chars='-', description='pretrain_arguments')
    args, _ = args.parse_known_args()
    args.num_nodes_dict = num_nodes_dict

    dataset_use = [item]
    args_predictor = get_predictor_params(device, args, dataset_use)
    args_predictor.alpha = t
    if item in ['TrafficSH', 'NYC_TAXI', 'TrafficHZ', 'TrafficJN', 'CHI_TAXI', 'NYC_TAXI', 'NYC_BIKE-3']:
        args_predictor.input_window = 48
        args_predictor.output_window = 48
    elif item in ['SZ_DIDI', 'CD_DIDI']:
        args_predictor.input_window = 144
        args_predictor.output_window = 144
    elif item in ['GBA', 'CAD3', 'CAD5']:
        args_predictor.input_window = 96
        args_predictor.output_window = 96
    return args_predictor

def main(args_predictor):

    print('==========')
    for arg in vars(args_predictor):
        print(arg, ':', getattr(args_predictor, arg))
    init_seed(args_predictor.seed, args_predictor.seed_mode)

    print('dataset: ', args_predictor.dataset_use, 'load_pretrain_path: ', args_predictor.load_pretrain_path, '  save_pretrain_path: ', args_predictor.save_pretrain_path)


    #load dataset
    train_dataloader, val_dataloader, test_dataloader, scaler_dict = define_dataloder(args=args_predictor)

    #init model
    from Model import Traffic_model as Network_Predict
    model = Network_Predict(args_predictor)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(args_predictor.device)
    if args.xavier:
        for p in model.parameters():
            if p.requires_grad == True:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    #init loss function, optimizer
    def scaler_mae_loss(mask_value):
        def loss(preds, labels, scaler, mask=None):
            if scaler and args_predictor.real_value == False:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)
            mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss

    def scaler_huber_loss(mask_value):
        def loss(preds, labels, scaler, mask=None):
            if scaler and args_predictor.real_value == False:
                preds = scaler.inverse_transform(preds)
                labels = scaler.inverse_transform(labels)
            mae = huber_loss(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss

    if args_predictor.loss_func == 'mask_mae':
        loss = scaler_mae_loss(mask_value=args_predictor.mape_thresh)
        print('============================scaler_mae_loss')
    elif args_predictor.loss_func == 'mask_huber':
        loss = scaler_huber_loss(mask_value=args_predictor.mape_thresh)
        print('============================scaler_huber_loss')

    elif args_predictor.loss_func == 'mae':
        loss = torch.nn.L1Loss()
    elif args_predictor.loss_func == 'mse':
        loss = torch.nn.MSELoss()
    else:
        raise ValueError


    optimizer = torch.optim.Adam(params=model.parameters(), lr=args_predictor.lr_init, eps=1.0e-8, weight_decay=args_predictor.weight_decay, amsgrad=False)
    # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40,60,80],gamma=0.1)

    #learning rate decay
    if args_predictor.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args_predictor.lr_decay_step.split(','))]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args_predictor.lr_decay_rate)
    else:
        scheduler = None

    ######################## start training ###########################3
    from BasicTrainer import Trainer
    trainer = Trainer(model, loss, optimizer, train_dataloader, val_dataloader, test_dataloader, scaler_dict, args_predictor, scheduler=scheduler)
    print_model_parameters(model, only_num=False)
    mae = trainer.multi_train()

    return mae

if __name__ == '__main__':

    Dataset = ['TrafficSH']
    tau = [1/2]
    for index, item in enumerate(Dataset):
        for t_idx, t in enumerate(tau):
            random.seed(12)
            np.random.seed(12)
            torch.manual_seed(12)
            torch.cuda.manual_seed(12)
            torch.cuda.manual_seed_all(12)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            args = parameter_parser(item, t)
            main(args)