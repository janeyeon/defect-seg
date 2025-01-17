export TORCH_USE_RTLD_GLOBAL=YES


area_resize_ratio=(0.01 0.005 0.01)
smooth_r=(1. 1. 1e5)

LORA_LRS=("1e-5" "1e-4")
LORA_DIMS=("4" "8" "2")
SPLITS=("0" "1" "2")

for split_ in "${SPLITS[@]}"; do
  for lora_lr in "${LORA_LRS[@]}"; do
    for lora_dim in "${LORA_DIMS[@]}"; do
      python main_lora.py --init_method "tcp://localhost:9999" --device "0,1,2,3" \
      --cfg method_config/VISION_V1/Train/SOFS.yaml --prior_layer_pointer 5 6 7 8 9 10 \
      --opts NUM_GPUS 4 DATASET.split $split_ \
      TRAIN_SETUPS.epochs 50 \
      TRAIN_SETUPS.TEST_SETUPS.epoch_test 50 \
      TRAIN_SETUPS.TEST_SETUPS.train_miou 50 \
      TRAIN_SETUPS.TEST_SETUPS.test_state True \
      TRAIN_SETUPS.TEST_SETUPS.val_state False \
      TRAIN.save_model False \
      DATASET.area_resize_ratio ${area_resize_ratio[$split_]} \
      DATASET.normal_sample_sampling_prob 0.3 \
      TRAIN.SOFS.smooth_r ${smooth_r[$split_]} \
      DATASET.name 'VISION_V1_ND' RNG_SEED 54 \
      TRAIN.SOFS_LoRA.lora_dim ${lora_dim} \
      TRAIN.SOFS_LoRA.lora_lr ${lora_lr}
    done
  done
done
