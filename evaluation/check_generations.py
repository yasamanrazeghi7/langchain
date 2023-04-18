import argparse


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset

def check_generation(args):
    #check step later
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped", revision="step3000", cache_dir="/net/nfs.cirrascale/allennlp/yasamanr/hf_cache", use_cache=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir="/net/nfs.cirrascale/allennlp/yasamanr/hf_cache", use_cache=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="/net/nfs.cirrascale/allennlp/yasamanr/hf_cache")
    inputs = tokenizer(args.input, return_tensors="pt")
    tokens = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(tokens[0]))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default="code-davinci-002", help="openai model name")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--input', type=str, default="Q: Calculate the greatest common factor of 690 and 10.\nA:")
    args = parser.parse_args()
    check_generation(args)