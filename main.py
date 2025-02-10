#!/usr/bin/python3

import logging
import os
import warnings
warnings.filterwarnings("ignore", message=".*MMCV will release v2.0.0.*")

from utils import load_config, parse_args, launch_job

from tools import train, test


LOGGER = logging.getLogger(__name__)

import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)  # Python의 random 모듈 시드 설정
    np.random.seed(seed)  # NumPy 시드 설정
    torch.manual_seed(seed)  # PyTorch 시드 설정
    torch.cuda.manual_seed(seed)  # CUDA 사용 시 추가 시드 설정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시 모든 GPU에 시드 설정
    torch.backends.cudnn.deterministic = True  # CuDNN의 Deterministic 모드 설정
    torch.backends.cudnn.benchmark = False  # CuDNN의 벤치마크 모드 비활성화
    os.environ['PYTHONHASHSEED'] = str(seed)  # 해시 시드 설정

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:
        # merge config and args, mkdir image_save and checkpoints
        cfg = load_config(args, path_to_config=path_to_config)
        
        # Perform training and test in each category.
        if cfg.TRAIN.enable:
            """
            todo in the new version
            include:
             1) train and test dataloader load
             2) training prepare phase: 1) base model load
                                        2) optimizer load
             3) start training: include various methods (class module)
             4) complete training: start test (one follow by one)
            """
            launch_job(cfg=cfg, init_method=args.init_method, func=train)

        if cfg.TEST.enable:
            launch_job(cfg=cfg, init_method=args.init_method, func=test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    set_seed(seed=42)
    main()
