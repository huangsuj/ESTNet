import argparse
import configparser

def get_predictor_params(device, args, dataset_use):
    # get the based paras of predictors
    config_file = './general_conf/global_baselines.conf'
    config = configparser.ConfigParser()
    config.read(config_file)

    parser_pred = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments')
    parser_pred.add_argument('--model', default='ST_test', type=str)

    # train
    parser_pred.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser_pred.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser_pred.add_argument('--layers', default=config['train']['layers'], type=int)
    parser_pred.add_argument('--val_ratio', default=config['train']['val_ratio'], type=float)
    parser_pred.add_argument('--test_ratio', default=config['train']['test_ratio'], type=float)
    parser_pred.add_argument('--weight_decay', default=config['train']['weight_decay'], type=float)
    parser_pred.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser_pred.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser_pred.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser_pred.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser_pred.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser_pred.add_argument('--xavier', default=config['train']['xavier'], type=eval)
    parser_pred.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser_pred.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser_pred.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser_pred.add_argument('--debug', default=config['train']['debug'], type=eval)
    parser_pred.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

    parser_pred.add_argument('--input_base_dim', default=config['train']['input_base_dim'], type=int)
    parser_pred.add_argument('--device', default=device, type=str, help='indices of GPUs')

    parser_pred.add_argument('--alpha', type=float, default=config['train']['alpha'])
    parser_pred.add_argument('--beta', type=float, default=config['train']['beta'])

    parser_pred.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    parser_pred.add_argument('--seed', type=int, default=config['train']['seed'])
    parser_pred.add_argument('--seed_mode', type=eval, default=config['train']['seed_mode'])
    # parser_pred.add_argument('--xavier', type=eval, default=config['train']['xavier'])
    parser_pred.add_argument('-load_pretrain_path', default=config['train']['load_pretrain_path'], type=str)
    parser_pred.add_argument('-save_pretrain_path', default=config['train']['save_pretrain_path'], type=str)

    # test
    parser_pred.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    parser_pred.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

    # log
    parser_pred.add_argument('--log_dir', default='./', type=str)
    parser_pred.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser_pred.add_argument('--save_step', default=config['log']['save_step'], type=int)

    # data
    # dataset_use_str = config.get('data', 'dataset_use')
    # dataset_use = eval(dataset_use_str)
    parser_pred.add_argument('--dataset_use', default=dataset_use, type=list)
    args_pred, _ = parser_pred.parse_known_args()
    parser_pred.add_argument('--num_nodes', type=int, default=args.num_nodes_dict[args_pred.dataset_use[0]])
    parser_pred.add_argument('--input_window', type=int, default=config['data']['input_window'])
    parser_pred.add_argument('--output_window', type=int, default=config['data']['output_window'])
    parser_pred.add_argument('-his', default=config['data']['his'], type=int)
    parser_pred.add_argument('-pred', default=config['data']['pred'], type=int)


    # model
    # parser_pred.add_argument('--lam', type=float, default=config['model']['lam'])
    parser_pred.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'])
    parser_pred.add_argument('--lape_dim', type=int, default=config['model']['lape_dim'])
    parser_pred.add_argument('--skip_dim', type=int, default=config['model']['skip_dim'])
    parser_pred.add_argument('--output_dim', type=int, default=config['model']['output_dim'])
    parser_pred.add_argument('--type_short_path', type=str, default=config['model']['type_short_path'])
    parser_pred.add_argument('--geo_num_heads', type=int, default=config['model']['geo_num_heads'])
    parser_pred.add_argument('--t_num_heads', type=int, default=config['model']['t_num_heads'])
    parser_pred.add_argument('--mlp_ratio', type=int, default=config['model']['mlp_ratio'])
    parser_pred.add_argument('--qkv_bias', type=eval, default=config['model']['qkv_bias'])
    parser_pred.add_argument('--drop', type=float, default=config['model']['drop'])
    parser_pred.add_argument('--attn_drop', type=float, default=config['model']['attn_drop'])
    parser_pred.add_argument('--drop_path', type=float, default=config['model']['drop_path'])
    parser_pred.add_argument('--add_time_in_day', type=eval, default=config['model']['add_time_in_day'])
    parser_pred.add_argument('--add_day_in_week', type=eval, default=config['model']['add_day_in_week'])

    from args import parse_args
    args_predictor = parse_args(parser_pred, args)


    return args_predictor