#!/bin/bash
#an evaluation script for the trained models and their checkpoints
cache_dir_base=/net/nfs.cirrascale/allennlp/yasamanr/icl_small/huggingface_model_cache
output_dir=/net/nfs.cirrascale/allennlp/yasamanr/icl_small/output_dir
cache_dir=$cache_dir_base/3/
mkdir $cache_dir
for dataset in add-2digits add-3digits mul-2digits
do
for method in icl
do
for model in pythia-6.9b
do
for  i in {0..140}
do
step=$((i*1000+3000))
echo $step $dataset $method $model
outputfile=$output_dir/${model}_step${step}_${method}_${dataset}
mkdir $outputfile
echo $outputfile
python main_few_shot.py --model-name EleutherAI/$model --revision step$step --dataset $dataset --learning_mode $method --bs 50 --shots 8 --cache_dir $cache_dir --generation_len 20 --output_dir $outputfile 1> $outputfile/results.out 2> $outputfile/results.err
rm -r $cache_dir/models--EleutherAI--$model
done
done
done
done
