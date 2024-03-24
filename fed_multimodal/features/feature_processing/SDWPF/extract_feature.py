# Author: Wenhai Liu
import pdb
import glob
import copy
import torch
import random
import pickle
import os, sys
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from pathlib import Path
from fed_multimodal.features.feature_processing.feature_manager import FeatureManager


def parse_args():
    
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('output'))

    parser = argparse.ArgumentParser(description='Extract acc and gyro features')
    parser.add_argument(
        '--raw_data_dir',
        default=path_conf["data_dir"], 
        type=str,
        help='source data directory'
    )
    
    parser.add_argument(
        '--output_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        default="SDWPF",
        help="data set name",
    )
    
    parser.add_argument(
        "--datafile_name", 
        type=str,
        default="SDWPF.csv",
        help="data set name",
    )
    
    args = parser.parse_args()
    return args

def split_train_dev_test(
    data_index: list,
    seed: int=8
) -> tuple[list, list, list]:
    
    train_arr = np.arange(len(data_index))
    np.random.seed(seed)
    np.random.shuffle(train_arr)
    
    train_len = int(len(data_index) * 0.7)
    val_len = int(len(data_index) * 0.15)
    # test_len = int(len(data_index) * 0.15)
    
    # print(train_len, " ",val_len," ", test_len)
    
    train_index = [data_index[idx] for idx in train_arr[:train_len]]
    val_index = [data_index[idx] for idx in train_arr[train_len:train_len + val_len]]
    test_index = [data_index[idx] for idx in train_arr[train_len + val_len:]]
    
    return train_index, val_index, test_index

if __name__ == '__main__':
    
    # 加载命令行参数
    args = parse_args()
    output_data_path = Path(args.output_dir).joinpath("feature",args.dataset)
    Path.mkdir(output_data_path, parents=True, exist_ok=True)
    
    agg_batch = 12  # 两个小时
    model1_output_data_path = Path(args.output_dir).joinpath('feature', 'model1', args.dataset, f'alpha{agg_batch}')
    model2_output_data_path = Path(args.output_dir).joinpath('feature', 'model2', args.dataset, f'alpha{agg_batch}')
    Path.mkdir(model1_output_data_path, parents=True, exist_ok=True)
    Path.mkdir(model2_output_data_path, parents=True, exist_ok=True)
    
    # 特征提取工具
    fm = FeatureManager(args)
    
    df = pd.read_csv(str(Path(args.raw_data_dir).joinpath(args.dataset, args.datafile_name)))
    df.fillna(method="backfill", inplace=True)
    df.fillna(method="pad",inplace=True)
    # 删除 Day 和 Tmstamp 两列
    # df.drop(columns=["Day", "Tmstamp"], inplace=True)
    # df.astype("float")
    # 按 TurbID 分组
    groups = df.groupby("TurbID")

    # 将分组后的数据格式转换成 NumPy
    data_by_turbid = {
        turbid: group.drop(columns=["TurbID"])    # 删除 Day 和 Tmstamp 两列
        for turbid, group in groups
    }

    print("数据总个数为:", int(df.shape[0] // agg_batch))
    print("客户端个数为:", int(len(data_by_turbid)))

    # 多模态数据分类
    model1_dict = {}
    model2_dict = {}
    # 规范化数据，处理预测值
    for turbid, data in tqdm(data_by_turbid.items()):
        data_by_turbid[turbid]["Patv"] = data_by_turbid[turbid]["Patv"].groupby(np.arange(len(data_by_turbid[turbid])) // agg_batch).transform("mean")
        scaler = StandardScaler()
        data_by_turbid[turbid][['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']] = scaler.fit_transform(data_by_turbid[turbid][['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv']])
        data_by_turbid[turbid] = data_by_turbid[turbid][['Wspd','Etmp','Itmp','Prtv','Wdir','Ndir','Pab1','Pab2','Pab3','Patv']].reset_index(drop=True)
        
        # 预测之后两个小时的发电量
        for i in range(data_by_turbid[turbid].shape[0]):
            
            if i % agg_batch != 0: continue
            if i > data_by_turbid[turbid].shape[0] - agg_batch + 1: break
            
            # print(i-agg_batch,i-1)
            data_by_turbid[turbid].loc[i-agg_batch:i-1,"Patv"] = data_by_turbid[turbid].at[i,"Patv"]
            
        model1_dict[turbid] = data_by_turbid[turbid][['Wspd','Etmp','Itmp','Prtv','Patv']].to_numpy()
        model2_dict[turbid] = data_by_turbid[turbid][['Wdir','Ndir','Pab1','Pab2','Pab3','Patv']].to_numpy()
        
    model1_dict_train = []
    model2_dict_train = []
    model1_dict_val = []
    model2_dict_val = []
    model1_dict_test = []
    model2_dict_test = []
        
    len_total = len(model1_dict[1]) // 12
    for i, _ in tqdm(model1_dict.items()):
        model1_dict_train.append(list())
        model2_dict_train.append(list())
        train_index, val_index, test_index = split_train_dev_test(np.arange(len_total))
        
        for j in train_index:
            tmp1 = [i, model1_dict[i][j*agg_batch,4], model1_dict[i][j*agg_batch:(j+1)*agg_batch,0:4]]
            tmp2 = [i, model2_dict[i][j*agg_batch,5], model2_dict[i][j*agg_batch:(j+1)*agg_batch,0:5]]
            model1_dict_train[i-1].append(tmp1)
            model2_dict_train[i-1].append(tmp2)
            
        for j in test_index:
            tmp1 = [i, model1_dict[i][j*agg_batch,4], model1_dict[i][j*agg_batch:(j+1)*agg_batch,0:4]]
            tmp2 = [i, model2_dict[i][j*agg_batch,5], model2_dict[i][j*agg_batch:(j+1)*agg_batch,0:5]]
            model1_dict_test.append(tmp1)
            model2_dict_test.append(tmp2)
            
        for j in val_index:
            tmp1 = [i, model1_dict[i][j*agg_batch,4], model1_dict[i][j*agg_batch:(j+1)*agg_batch,0:4]]
            tmp2 = [i, model2_dict[i][j*agg_batch,5], model2_dict[i][j*agg_batch:(j+1)*agg_batch,0:5]]
            model1_dict_val.append(tmp1)
            model2_dict_val.append(tmp2)
            
        with open(model1_output_data_path.joinpath(f'{i}.pkl'), 'wb') as handle:
            pickle.dump(model1_dict_train[i-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(model2_output_data_path.joinpath(f'{i}.pkl'), 'wb') as handle:
            pickle.dump(model2_dict_train[i-1], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(model1_output_data_path.joinpath('test.pkl'), 'wb') as handle:
        pickle.dump(model1_dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    with open(model1_output_data_path.joinpath('dev.pkl'), 'wb') as handle:
        pickle.dump(model1_dict_val, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        
    with open(model2_output_data_path.joinpath('test.pkl'), 'wb') as handle:
        pickle.dump(model2_dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)  
    with open(model2_output_data_path.joinpath('dev.pkl'), 'wb') as handle:
        pickle.dump(model2_dict_val, handle, protocol=pickle.HIGHEST_PROTOCOL)  