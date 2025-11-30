import torch
import random
import numpy as np
import logging
import os
import sys
import json
from copy import deepcopy


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
def construct_cached_mm_representations(args):
    root = args['mm_root']
    cached_mm_representations = {}
    for dname in args['datasets']:
        labels = []
        if 'selected_cls_labels' in args['datasets'][dname]:    # only load partially
            for ele in args['datasets'][dname]['selected_cls_labels']:
                labels.append(ele[0])
        else:
            labels = sorted(os.listdir(os.path.join(args['data_root'], dname)))
            labels = filter(lambda label: os.path.isdir(os.path.join(args['data_root'], dname, label)), labels)
        cached_mm_representations[dname] = {}
        for label in labels:
            if os.path.exists(os.path.join(args['mm_root'], dname, label)):
                mm_representation = torch.load(os.path.join(args['mm_root'], dname, label, 'mm_representation.pth'))            
                cached_mm_representations[dname][label] = mm_representation
    # build index table for search
    cached_mm_representations_index = {}
    for dname in cached_mm_representations:
        cached_mm_representations_index[dname] = {}
        for label in cached_mm_representations[dname]:
            cached_mm_representations_index[dname][label] = {}
            for frame_id in cached_mm_representations[dname][label]:
                base_id = os.path.splitext(frame_id)[0]
                prefix, index = base_id.rsplit('_', maxsplit=1)
                if prefix not in cached_mm_representations_index[dname][label]:
                    cached_mm_representations_index[dname][label][prefix] = [int(index)]
                else:
                    cached_mm_representations_index[dname][label][prefix].append(int(index))
    
    return cached_mm_representations, cached_mm_representations_index


def get_nearest_mm_index(cached_index, index):
    index = int(index)
    previous_index = []
    for i in cached_index:
        if i <= index:
            previous_index.append(i)
    return max(previous_index)


def get_logger(name, config):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.flush = sys.stdout.flush
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)
    os.makedirs(os.path.join('expts', config['expt']), exist_ok=True)
    out_handler = logging.FileHandler(filename=os.path.join('expts', config['expt'], f'{config["mode"]}.log'))
    out_handler.setLevel(level=logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_train_dataset_config(config):
    dataset_classes = config['classes']
    dataset_config = {}
    for dataset_class in dataset_classes:
        dataset_config[dataset_class] = {
            "data_root": f'{config["data_root"]}/{dataset_class}',
            "dataset_type": "VideoFolderDatasetForReconsWithFn",
            "mode": config["mode"],
            "selected_cls_labels": [("0_real", 0), ("1_fake", 1)]
        }
        if config['fix_split'] and config["mode"] == 'train':
            dataset_config[dataset_class]['split'] = {
                'train': json.load(open(os.path.join(config['split_path'], 'train.json')))[dataset_class],
                'val': json.load(open(os.path.join(config['split_path'], 'val.json')))[dataset_class]
            }
    return dataset_config


def get_test_dataset_configs(config):
    dataset_classes = config['classes']
    dataset_configs = []
    default_real_dataset_config = {}
    default_real_test_datasets_classes = ['videocrafter1', 'zeroscope', 'opensora']
    for real_dataset_class in default_real_test_datasets_classes:    # load default real dataset
        default_real_dataset_config[real_dataset_class] = {
            "data_root": f'{config["data_root"]}/{real_dataset_class}',
            "dataset_type": "VideoFolderDatasetForReconsWithFn",
            "mode": config["mode"],
            "selected_cls_labels": [("0_real", 0)],
            "sample_method": 'entire',
        }
    for dataset_class in dataset_classes:    # load fake datasets
        dataset_config = deepcopy(default_real_dataset_config)
        if dataset_class not in dataset_config:
            dataset_config[dataset_class] = {
                "data_root": f'{config["data_root"]}/{dataset_class}',
                "dataset_type": "VideoFolderDatasetForReconsWithFn",
                "mode": config["mode"],
                "selected_cls_labels": [("1_fake", 1)],
                "sample_method": 'entire',
            }
        else:
            dataset_config[dataset_class]["selected_cls_labels"].append(("1_fake", 1))
        dataset_configs.append(dataset_config)
    return dataset_configs


def get_customized_test_dataset_configs(config):
    dataset_classes = config['classes']
    dataset_configs = []

    for dataset_class in dataset_classes:    # load fake datasets
        dataset_config = {}
        dataset_config[dataset_class] = {
            "data_root": f'{config["data_root"]}/{dataset_class}',
            "dataset_type": "VideoFolderDatasetForReconsWithFn",
            "mode": config["mode"],
            "selected_cls_labels": [("1_fake", 1), ("0_real", 0)],
            "sample_method": 'entire',
        }
        dataset_configs.append(dataset_config)
    return dataset_configs