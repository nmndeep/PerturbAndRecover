#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=1:52:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=PAR
#SBATCH --output=~/JobLogs/PAR_%j.out
#SBATCH --error=~/JobLogs/PAR_%j.err
# print info about current job

scontrol show job $SLURM_JOB_ID 
export WORLD_SIZE=4

#activate conda environmen
source ~/miniconda3/etc/profile.d/conda.sh
conda activate main_py

MASTER_PORT=29500

#which encoder model to use
modeln=ViT-B/32
#absolute path of posoned model checkpoint
modelPath=PATH-TO-CHECKPOINT
#absolute path of output directory for checkpoint, logs
outDir=OUTPUT-DIR

#absolute path of train csv file with image, caption as headers - if you change data, 
#also change the -- dataset param below
trainData=ADD-LOC
#absolute path of location where train images are located
imgRoot=ADD-LOC
#variables of attack to test for

attack=badnet_rs
patch_loc=random
label=banana

##flushes the standard python print and also captures the output model directory into $final_string variable
{ read -r final_string; python_output=$(cat); } < <(torchrun --nproc_per_node=${WORLD_SIZE} --nnodes=1 --master_port=${MASTER_PORT} \
  train.py --dataset cc3m --load-pretrained-clip ${modelPath} --model ${modeln} \
  --backdoor-tuple 1,${attack},16,random,0.5,${label} --train-data ${trainData}\
    --root ${imgRoot} --batch-size 512 --epochs 2 --output-dir ${outDir} \
    --samples 250000 --workers 16 --addendum badnet_rs-TEST | tee /dev/stderr | grep "^FINAL_STRING:" | cut -d':' -f2-)

# echo $final_string
#pass final_string and attack params for validation
bash trigger_validate.sh ${final_string} ${attack} ${modeln} ${label}

