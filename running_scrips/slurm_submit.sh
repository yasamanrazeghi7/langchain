#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home/yrazeghi/slurm_logs/12pythsvamp.txt
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s5
#SBATCH --gpus=1
#SBATCH --mem=100000MB

model_name=EleutherAI/pythia-1.4b
dataset=asdiv
shots=8
seed=1
learning_mode="cot"
bs=2

srun /home/yrazeghi/anaconda3/envs/dl/bin/python /home/yrazeghi/icl_small/main_few_shot.py --model-name=$model_name --dataset=$dataset --shots=$shots --device=0 --seed=$seed --learning_mode=$learning_mode --bs=$bs > /home/yrazeghi/icl_small/results/$model_name-$bs-$dataset-$shots-$seed-$learning_mode.alaki