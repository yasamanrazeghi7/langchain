import json
import time
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import BasePromptTemplate, FewShotPromptTemplate2, PromptTemplate, LastLetterConcat, LastLetterOutputParser, LastLetterConcatCoT, LastLetterOutputParserCoT

import few_shot_utils as few_shot_utils


def run_experiment(args):
    #read the dataset:
    bs = args.bs
    print(f'the batch size is {bs}')

    few_shot_utils.set_seed(args.seed)


    # this is setting up the model
    
    # example on the huggingface models
    # model_id = "EleutherAI/gpt-neo-1.3B"
    cache_file_path = "/extra/ucinlp0/yrazeghi/huggingfacecache/"
    model_id = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_file_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_file_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250, device=args.device, do_sample=False)
    hf = HuggingFacePipeline(pipeline=pipe)


    # this is setting up the dataset
    test_data, few_shot_template = few_shot_utils.set_up_data_set(args.dataset, args.shots, args.learning_mode)

    # this is setting up the chain
    chain = LLMChain(llm=hf, prompt=few_shot_template, verbose=False)

    total_count = 0
    correct_count = 0
    for batch in few_shot_utils.make_batch(test_data, bs):
        model_outputs = chain.apply_and_parse(batch)

        for i, model_output in enumerate(model_outputs):
            if model_output == batch[i]['answer']:
                correct_count += 1
            total_count += 1
    print(f'{correct_count}, {total_count}, {correct_count/total_count}')
    print("haha this is right")

#the hugginface_pipeline has a generate function that is now only good for gpt model because of the way we set the tokenizer parameters
#The way we define a dataset is also hacky but I couldn't find a better way to do it
#todo:
#1. make the _generate function more general, maybe consult with Navid try on opt models
#2. add math datasets 
#3. run experiments!!



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default="code-davinci-002", help="openai model name")
    parser.add_argument('--dataset', type=str,
                        default="first_letter", choices=["first_letter", "last_letter", "gsm8k", "asdiv", "svamp"],
                                                             help="dataset name")
    parser.add_argument('--learning_mode', type=str, default='cot', choices=[
                        'cot', 'standard'], help="cot is for chain of thought and standard is in context learning")
    parser.add_argument('--api_key', type=str,
                        help="api key to be used for the experiment")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--shots', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bs', type=int, default=20)
    args = parser.parse_args()
    run_experiment(args)
    

# python main_few_shot.py --model-name="EleutherAI/gpt-neo-1.3B" --dataset="first_letter" --shots=4 --device=0 --seed=1 --learning_mode="standard" 
# python main_few_shot.py --model-name="EleutherAI/gpt-neo-1.3B" --dataset="first_letter" --shots=4 --device=0 --seed=1 --learning_mode="standard" 



