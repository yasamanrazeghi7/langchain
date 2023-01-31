# Data Scripts

Simple and easy scripts for generating splits out of the pile. You'll need to install `tqdm` and `joblib`.

This requires having the pile data in an accessible folder where it is in gzip'ed jsonl files split across some number of files (e.g. `00.jsonl.gzip`, `01.jsonl.gzip`, etc.).

To create a split containing examples from a given pile data source:
```bash
python create_dataset_subset.py -p <pile folder> -o <output filename> -s <data source name>
```

To create a split containing all documents that match a regex:
```bash
python create_regex_subset.py -p <pile folder> -o <output filename> -r <regex>
```

Both scripts create tmp files behind the scenes that can take up a lot of space (the output file can take up a lot of space, too!). The default place to put these is `/tmp`, but you can change where with the `-t` flag.

## Deduplicating

The pile contains duplicated instances to match the epochs covered given in the paper. `dedup.py` removes all exact duplicate documents from a given jsonl file:
```bash
python dedup.py -i <file>
```

Currently it creates a deduplicated file in the same location as the input file, then overwrites the input file with the deduplicated version, so if there isn't enough space for two of the files, you might want to edit it to change that.
