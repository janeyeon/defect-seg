#!/bin/bash

export TORCH_USE_RTLD_GLOBAL=YES

area_resize_ratio=(0.01 0.005 0.01)
smooth_r=(1. 1. 1e5)
for split_ in 0
do
  python main.py --init_method "tcp://localhost:9988" --device "0, 1" \
  --cfg method_config/VISION_V1/Train/SOFS.yaml --prior_layer_pointer 5 6 7 8 9 10 \
  --opts NUM_GPUS 2 DATASET.split $split_ \
  TRAIN_SETUPS.epochs 10 TRAIN_SETUPS.TEST_SETUPS.epoch_test 10 TRAIN_SETUPS.TEST_SETUPS.train_miou 10 \
  TRAIN_SETUPS.TEST_SETUPS.test_state True TRAIN_SETUPS.TEST_SETUPS.val_state False \
  TRAIN.save_model True \
  DATASET.area_resize_ratio ${area_resize_ratio[$split_]} \
  DATASET.normal_sample_sampling_prob 0.3 \
  TRAIN.SOFS.smooth_r ${smooth_r[$split_]} \
  DATASET.name 'VISION_V1_ND' RNG_SEED 54
done
