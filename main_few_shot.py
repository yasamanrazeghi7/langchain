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
    if args.debug:
        print(f'the batch size is {bs}')

    few_shot_utils.set_seed(args.seed)
    


    # this is setting up the model
    
    # example on the huggingface models
    # model_id = "EleutherAI/gpt-neo-1.3B"
    # cache_file_path = "/extra/ucinlp0/yrazeghi/huggingfacecache/"
    if args.cache_dir is not None:
        cache_dir = args.cache_dir
    else:
        cache_dir =' ~/.cache/huggingface'
    model_id = args.model_name
    if args.revision == 'None':
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=args.revision, cache_dir=cache_dir)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=args.generation_len, device=args.device, do_sample=False)
    #check if there is another pipeline for cross_entropy available
    hf = HuggingFacePipeline(pipeline=pipe)


    # this is setting up the dataset
    test_data, few_shot_template = few_shot_utils.set_up_data_set(args.dataset, args.shots, args.learning_mode)
    if args.sample > 0:
        test_data = random.sample(test_data, args.sample)

    # this is setting up the chain
    verbose = False
    if args.debug:
        verbose = True
    chain = LLMChain(llm=hf, prompt=few_shot_template, verbose=verbose)

    total_count = 0
    correct_count = 0
    output_list = []
    for batch in few_shot_utils.make_batch(test_data, bs):
        model_outputs = chain.apply_and_parse(batch)
        for i, model_output in enumerate(model_outputs):
            if args.debug:
                print (batch[i])
                print(f'answer: {batch[i]["answer"]}')
                print(f'model output: {model_output}')
            output_list.append({"question":batch[i]["question"], "model_output":model_output, "true_answer":batch[i]["answer"]})
            if model_output == batch[i]['answer']:
                correct_count += 1
                if args.debug:
                    print('correct')
            else:
                if args.debug:
                    print('incorrect')
            if args.debug:
                input()
            total_count += 1
    print(f'{correct_count}, {total_count}, {correct_count/total_count}')
    output_dict = {}
    output_dict["correct_count"] = correct_count
    output_dict["total_count"] = total_count
    output_dict["total_accuracy"] = correct_count/total_count
    output_dict["raw_answers"] = output_list
    output_dict["args"] = vars(args)

    with open(f'{args.output_dir}/{args.model_name[11:]}_{args.revision}_{args.learning_mode}_{args.dataset}_results.jsonl', 'w') as file_pointer:
        json.dump(output_dict, file_pointer)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default="code-davinci-002", help="openai model name")
    parser.add_argument('--dataset', type=str,
                        default="first_letter")
    parser.add_argument('--revision', type=str,
                        default="None")
    parser.add_argument('--learning_mode', type=str, default='icl', choices=[
                        'standard_cot', 'icl', 'data_cot'], help="cot is for chain of thought and standard is in context learning")
    parser.add_argument('--api_key', type=str,
                        help="api key to be used for the experiment")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', type=int ,default=0)
    parser.add_argument('--shots', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--generation_len', type=int, default=250)
    parser.add_argument('--output_dir', type=str, default='./output_dir')
    args = parser.parse_args()
    run_experiment(args)
    

# python main_few_shot.py --model-name="EleutherAI/gpt-neo-1.3B" --dataset="first_letter" --shots=4 --device=0 --seed=1 --learning_mode="standard" 
# python main_few_shot.py --model-name="EleutherAI/gpt-neo-1.3B" --dataset="first_letter" --shots=4 --device=0 --seed=1 --learning_mode="standard" 



