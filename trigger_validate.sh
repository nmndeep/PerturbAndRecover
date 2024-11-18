#!/bin/bash

export WORLD_SIZE=1

LOC=$(curl -s ifconfig.me)


echo "Evaluating from ------ ${1}"
echo "ATTACK ------ ${2}"
echo "MODEL ------ ${3}"
echo "LABEL ------ ${4}"
# lowest_gpu=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -k2 -n | head -n1)

# # Extract the GPU index
# gpu_index=$(echo $lowest_gpu | cut -d',' -f1)

# echo "GPU with lowest utilization: $gpu_index"

# Use this GPU for your task
export CUDA_VISIBLE_DEVICES=0

#clean IMNET
python -m eval.validate --eval_data_type ImageNet1K --eval_test_data_dir data/ImageNet1K/validation --device_id 0  --device cuda --checkpoint ${1} --model_name ${3} --test_label ${4}
#BKDOOR IMNET
python -m eval.validate --eval_data_type ImageNet1K --eval_test_data_dir data/ImageNet1K/validation --device_id 0  --device cuda  --checkpoint ${1} --add_backdoor --asr --patch_type ${2} --model_name ${3} --test_label ${4}
#clean COCO  
python -m eval.validate --eval_data_type COCO --eval_test_data_dir data/ImageNet1K/validation --device_id 0 --device cuda  --checkpoint ${1} --model_name ${3} --test_label ${4}
#BKDOOR COCO
python -m eval.validate --eval_data_type COCO --eval_test_data_dir data/ImageNet1K/validation --device_id 0 --device cuda --checkpoint ${1} --add_backdoor --asr --patch_type ${2} --model_name ${3} --test_label ${4}


