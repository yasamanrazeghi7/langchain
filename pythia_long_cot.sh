#!/bin/bash
#an evaluation script for the trained models and their checkpoints
for dataset in asdiv
do
for method in standard_cot
do
for model in EleutherAI/pythia-6.9b EleutherAI/pythia-6.9b-deduped
do
for  i in {0..28}
do
step=$((i*5000+3000))
echo $step $dataset $method $model
python main_few_shot.py --model-name $model --revision step$step --dataset $dataset --learning_mode $method --bs 8 --shots 8
done
done
done
done