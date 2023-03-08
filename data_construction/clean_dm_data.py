'''
A basic script to check our training and dev data is / isn't contaminated.
'''
import argparse
import json
import gzip
from multiprocessing import Pool
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
# we expect the train folder, where you have 00.jsonl.gz, 01.jsonl.gz, etc.
parser.add_argument(
    "-p", "--pretraining", type=str, required=True, help="pretraining data file to check."
)
parser.add_argument(
    "-s", "--samples", type=str, required=True, help="train/dev file to check."
)
parser.add_argument(
    '--clean', action='store_true', help='clean the pretraining data. If not set, will just check for matches, rather than remove them.'
)
parser.add_argument(
    '--multiprocess', action='store_true', help='run with multiple processes.'
)
parser.add_argument(
    '-o', '--output_file', type=str, default='output.jsonl', help="name of output file (where we write the cleaned data)."
)
args = parser.parse_args()


def open_func(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'r')
    else:
        return open(filename, 'r')

samples = [json.loads(line.strip()) for line in tqdm(open_func(args.samples).readlines())]
pretraining_data = [json.loads(line.strip()) for line in tqdm(open_func(args.pretraining).readlines()) if line.strip()]

# sometimes the answer gets cut off, so everything after the question is optional for removal.
contamination_strings = [re.escape(sample['question']) + r"(\n)?[^\n]*(\n)?" for sample in samples]

# to keep things simple, we will check for every sample input if it exists in each pretraining doc.
def check_contamination(sample):
    for doc in pretraining_data:
        if sample['question'] in doc['text']:
            return sample


def remove_contamination(doc):
    for pattern, _ in zip(contamination_strings, samples):
        doc['text'] = re.sub(pattern, '', doc['text'])
    return doc


try:
    if args.multiprocess:
        pool = Pool(64)
    print('processing')
    results = []
    if args.clean:
        if args.multiprocess:
            for r in tqdm(pool.imap(remove_contamination, pretraining_data), total=len(pretraining_data)):
                results.append(r)
        else:
            for d in tqdm(pretraining_data):
                results.append(remove_contamination(d))
        with open(args.output_file, 'w') as f:
            for doc in results:
                f.write(json.dumps(doc) + '\n')
    else:
        counter = 0
        if args.multiprocess:
            for r in tqdm(pool.imap(check_contamination, samples), total=len(samples)):
                if r:
                    print(r, 'is in training data.')
                    counter += 1
        else:
            for s in tqdm(samples):
                if check_contamination(s):
                    print(s, 'is in training data.')
                    counter += 1
        print('found', counter, 'contaminated samples.')
finally:
    if args.multiprocess:
        pool.close()
        pool.join()
