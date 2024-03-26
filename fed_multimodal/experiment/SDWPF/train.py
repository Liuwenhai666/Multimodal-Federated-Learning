import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import SDWPFRegression, DistAELoss
from fed_multimodal.dataloader.dataload_manager import DataloadManager

from fed_multimodal.trainers.fed_rs_trainer import ClientFedRS
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvgForRegression
from fed_multimodal.trainers.scaffold_trainer import ClientScaffold

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 设置确定的随机数
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    # read path config files
    path_conf = dict()
    # 读取fed_multimodal下的文件
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
            
    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument(
        '--data_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    
    # parser.add_argument(
    #     '--acc_feat', 
    #     default='acc',
    #     type=str,
    #     help="acc feature name",
    # )
    
    # parser.add_argument(
    #     '--gyro_feat', 
    #     default='gyro',
    #     type=str,
    #     help="gyro feature name",
    # )
    
    parser.add_argument(
        '--model1_feat', 
        default='model1',
        type=str,
        help="acc feature name",
    )
    
    parser.add_argument(
        '--model2_feat', 
        default='model2',
        type=str,
        help="gyro feature name",
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--test_frequency', 
        default=1,
        type=int,
        help="perform test frequency",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )

    parser.add_argument(
        '--mu',
        type=float, 
        default=0.001,
        help='Fed prox term'
    )
    
    parser.add_argument( # 联合学习聚合算法
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(    # Direchlet 分布中的 α,生成No-iid数据时使用
        "--alpha",
        type=float,
        default=None,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(    # 是否使用自注意力机制
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not'
    )
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )
    
    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label')
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="SDWPF",
        help='data set name'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='multimodal',
        help='modality type'
    )
    
    parser.add_argument(
        '--agg_batch', 
        type=int, 
        default=12,
        help='时间长度, 每一个单位1为10分钟'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()

    # data manager
    dm = DataloadManager(args)
    # 如果设置了缺失模态、标签，添加噪声，函数才执行
    dm.get_simulation_setting(alpha=args.alpha)
    
    # find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    # 保存结果
    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvgForRegression
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS
    logging.info("联邦学习算法:" + f"{args.fed_alg}")

    # 如果设置了缺失模态、标签，添加噪声，load simulation feature
    dm.load_sim_dict()
    # load client ids，从 fed_multimodal/output/feature/acc/uci-har/alpha01 
    dm.get_client_ids()
    # set dataloaders
    dataloader_dict = dict()
    logging.info("加载数据.")
    
    # 为每个客户端，加载数据集
    for client_id in tqdm(dm.client_ids):
        model1_dict = dm.load_baidukdd_model1_feat(client_id)
        model2_dict = dm.load_baidukdd_model2_feat(client_id)
        
        shuffle = False if client_id in ['dev', 'test'] else True
        # 目前未使用
        client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
        
        dataloader_dict[client_id] = dm.set_dataloader(
            model1_dict,
            model2_dict,
            client_sim_dict=client_sim_dict,
            default_feat_shape_a=np.array([12, 4]),
            default_feat_shape_b=np.array([12, 5]) 
        )
        
    for fold_idx in range(1, 2):
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        
        num_of_clients = len(client_ids)
        logging.info("num_of_clients: " + str(num_of_clients))
        
        set_seed(8*fold_idx)
        
        criterion = DistAELoss(alph=1e-3, Lossfunc=nn.MSELoss()).to(device)
        
        # 网络模型
        global_model = SDWPFRegression(
            model1_input_dim=4,
            model2_input_dim=5,
            en_att=args.att,
            d_hid=args.hid_size,
            att_name=args.att_name
        )
        global_model = global_model.to(device)
        
        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,    # 损失函数
            client_ids=client_ids 
        )
        
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients,
            sample_rate=args.sample_rate
        )
        server.get_num_params()
        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', 
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)
        
        # 再次设置随机数种子
        set_seed(8*fold_idx)
        
        for epoch in range(int(args.num_epochs)):
            server.initialize_epoch_updates(epoch)
            # 1. 本地训练，fed_avg中的返回权重，fed_sgd中的返回梯度
            skip_client_ids = list()
            
            for idx in server.clients_list[epoch]:
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue
                
                client = Client(
                    args=args,
                    device=device,
                    criterion=criterion,
                    dataloader=dataloader,
                    model=copy.deepcopy(server.global_model)
                )
                
                client.update_weights()
                    # server append updates
                server.save_train_updates(
                    copy.deepcopy(client.get_parameters()), 
                    client.result['sample'], 
                    client.result
                    )
                del client
                
            # 服务器一次运行完成
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            
            # 2. 聚合，加载新的全局权重
            if len(server.num_samples_list) == 0: continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_regression_result(
                data_split='train'
            )
            
            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    # 3. Perform the validation on dev set
                    server.inference_regression(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_regression_result(
                        data_split='dev'
                    )

                    # 4. Perform the test on holdout set
                    server.inference_regression(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_regression_result(
                        data_split='test'
                    )
                
                logging.info('---------------------------------------------------------')
                server.log_epoch_result_regression()
            logging.info('---------------------------------------------------------')
            
        # Performance save code
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results_regression()
        
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

    # Calculate the average of the 5-fold experiments
    save_result_dict['average'] = dict()
    for metric in ['loss']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: 
                continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)
    
    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )

            
