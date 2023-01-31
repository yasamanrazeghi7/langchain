"""
This script creates a subset of the pile based on a regex
"""
import argparse
import gzip
import json
import os
import re
import shutil

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
# we expect the train folder, where you have 00.jsonl.gz, 01.jsonl.gz, etc.
parser.add_argument("-p", "--pile", type=str, help="location of the pile data")
parser.add_argument("-o", "--output", type=str, help="where to save the output")
parser.add_argument("-r", "--regex", type=str, help="the regex to use")
args = parser.parse_args()

pile_loc = args.pile
output_loc = args.output
regex = args.regex


def search_and_write(outfile, file):
    out = open(outfile, "w")
    with gzip.open(os.path.join(pile_loc, file), "r") as f:
        for line in f:
            data = json.loads(line)
            if re.search(regex, data["text"]):
                out.write(json.dumps(data) + "\n")
    print(f"Done with file {file}!")


print(f"searching for files from {regex}")
files_to_process = [
    file for file in os.listdir(pile_loc) if file.endswith(".jsonl.gz")
]
Parallel(n_jobs=len(files_to_process))(
    delayed(search_and_write)(
        f"/mnt/tank2/hamish_cot_subsets/_{i}_tmp.jsonl", file
    )
    for i, file in enumerate(files_to_process)
)

# merge all the files
with open(output_loc, "wb") as w:
    for f in [
        f"/mnt/tank2/hamish_cot_subsets/_{i}_tmp.jsonl"
        for i in range(len(files_to_process))
    ]:
        with open(f, "rb") as fd:
            shutil.copyfileobj(fd, w)
