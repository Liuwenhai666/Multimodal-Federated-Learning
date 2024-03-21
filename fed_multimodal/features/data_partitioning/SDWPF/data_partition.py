# Author: Wenhai Liu
# wenhai.liu@ncepu.edu.cn
import json
import subprocess
import csv
import re, pdb
import sys, os
import argparse
import numpy as np
from pathlib import Path
from fed_multimodal.features.data_partitioning.partition_manager import PartitionManager


def data_partition(args: dict):
    arg_call = ["python3", "features/feature_processing/SDWPF/extract_feature.py"]
    for i, y in args.__dict__.items():
        arg_call.append(str(i))
        arg_call.append(str(y))
    subprocess.call(arg_call)


if __name__ == "__main__":

    # 读取 fed_multimodal/system.cfg 配置文件
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
    
    # 如果是默认设置,数据地址:fed_multimodal/data,输出地址:fed_multimodal/output
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('output'))
    
    # Read Args
    parser = argparse.ArgumentParser()
    parser.add_argument(        # 原始数据目录, default = "./data"
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="原始数据集路径",
    )
    
    parser.add_argument(    # 输出处理数据地址
        "--output_partition_path",
        type=str,
        default=f'{path_conf["output_dir"]}/partition',
        help="Output path of speech_commands data set",
    )
    
    parser.add_argument(    # 客户端数量
        '--num_clients', 
        type=int, 
        default=5, 
        help='Number of shards to split a subject data.'
    )

    parser.add_argument(    # 集中 or 联邦
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument("--dataset", default="SDWPF")
    args = parser.parse_args()
    
    data_partition(args)
    
    
    