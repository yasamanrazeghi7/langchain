"""
A quick script to dedup a generated file.
Exact matches only, I was worried this was happening.
"""
import argparse
import json
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, help="location of the input data"
)
args = parser.parse_args()

input_loc = args.input

seen = set()
unique = 0
total = 0

tmp = open(input_loc + ".tmp", "w")

for line in tqdm(open(input_loc, "r")):
    data = json.loads(line)
    if data["text"] not in seen:
        unique += 1
        tmp.write(line + "\n")
    seen.add(data["text"])
    total += 1

os.rename(input_loc + ".tmp", input_loc)
print(
    f"Found {unique} unique lines out of {total} total lines. Removed {total - unique} duplicates."  # noqa: E501
)
