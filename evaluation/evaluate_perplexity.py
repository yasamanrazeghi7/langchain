"""Evaluate the perplexity of a model on a dataset. 
    This is based on the code https://huggingface.co/docs/transformers/perplexity
"""

import argparse
from itertools import chain
import math

import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from tqdm import tqdm
from datasets import load_dataset

import training.load_pile_splits as load_pile_splits


def run_evaluation(args):
    #this is from hugginface repo with the sliding window approach
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #maybe it is better to use the test part of pile but I am not sure
    # dataset = load_dataset("the_pile", name="pubmed", split="test") we can also direcly use the pile maybe this is better, maybe its not! I don't know...
    if args.data == 'wikipedia':
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", )
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        max_length = 1024
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        # print(f"nlls: {nlls}")
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        print(f"Perplexity: {ppl}")

    else:
        raw_datasets = load_dataset('training/load_pile_splits.py', filename=args.data)
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])
        
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        block_size = 1024
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        # print(len(raw_datasets['train']))


        train_dataloader = DataLoader(
            lm_datasets["train"], shuffle=True, collate_fn=default_data_collator, batch_size=args.bs
        )

        model.eval()
        losses = []
        for i, batch in enumerate(tqdm(train_dataloader)):
            with torch.no_grad():
                outputs = model(batch["input_ids"].to(device), labels=batch["labels"].to(device))
            losses.append(outputs.loss)
        try:
            perplexity = torch.exp(torch.stack(losses).mean())
        except OverflowError:
            perplexity = float("inf")
        print(perplexity)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default="code-davinci-002", help="openai model name")
    parser.add_argument('--data', type=str, default="wikipedia", help="data split to evaluate on")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bs', type=int, default=10)
    args = parser.parse_args()
    run_evaluation(args)