"""
This script creates a subset of the pile containing only data from a certain source
"""
import argparse
import gzip
import json
import os
import shutil

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
# we expect the train folder, where you have 00.jsonl.gz, 01.jsonl.gz, etc.
parser.add_argument(
    "-p", "--pile", type=str, required=True, help="location of the pile data"
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="where to save the output"
)
parser.add_argument(
    "-s", "--source", type=str, required=True, help="name of source dataset"
)
parser.add_argument(
    "-t",
    "--tmp_loc",
    type=str,
    required=True,
    help="where to save tmp files",
    default="/tmp",
)

args = parser.parse_args()

pile_loc = args.pile
output_loc = args.output
source = args.source

# simplified names for cmd ease :)
valid_sources = {
    "cc": "Pile-CC",
    "webtext": "OpenWebText2",
    "stackexchange": "StackExchange",
    "arxiv": "ArXiv",
    "books": "Books3",
    "github": "Github",
    "youtube": "YoutubeSubtitles",
    "hackernews": "HackerNews",
    "law": "FreeLaw",
    "wiki": "Wikipedia (en)",
    "pubmed": "PubMed Central",
    "uspto": "USPTO Backgrounds",
    "philpaper": "PhilPapers",
    "pubmedabstracts": "PubMed Abstracts",
    "nih": "NIH ExPorter",
    "gutenberg": "Gutenberg (PG-19)",
    "enron": "Enron Emails",
    "bookcorpus2": "BookCorpus2",
    "subtitles": "OpenSubtitles",
    "deepmindmath": "DM Mathematics",
}

error_str = f"Invalid source. Valid sources are: {valid_sources.values()} or {valid_sources.keys()}"  # noqa: E501
assert source in valid_sources.items() or source in valid_sources.keys(), error_str

if source in valid_sources.keys():
    source = valid_sources[source]


def search_and_write(outfile, file):
    out = open(outfile, "w")
    with gzip.open(os.path.join(pile_loc, file), "r") as f:
        for line in f:
            data = json.loads(line)
            if data["meta"]["pile_set_name"].lower() == source.lower():
                out.write(json.dumps(data) + "\n")
    print(f"Done with file {file}!")


print(f"searching for files from {source}")
files_to_process = [file for file in os.listdir(pile_loc) if file.endswith(".jsonl.gz")]
Parallel(n_jobs=len(files_to_process))(
    delayed(search_and_write)(os.path.join(args.tmp_loc, f"/_{i}_tmp.jsonl"), file)
    for i, file in enumerate(files_to_process)
)

# merge all the files
with open(output_loc, "wb") as w:
    for f in [
        os.path.join(args.tmp_loc, f"/_{i}_tmp.jsonl")
        for i in range(len(files_to_process))
    ]:
        with open(f, "rb") as fd:
            shutil.copyfileobj(fd, w)

# cleanup
for f in [
    os.path.join(args.tmp_loc, f"/_{i}_tmp.jsonl") for i in range(len(files_to_process))
]:
    os.remove(f)
