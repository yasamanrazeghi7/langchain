FROM huggingface/transformers-pytorch-gpu:4.21.1

ENV PYTHONPATH .

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /icl_workdir/

# Copy the `requirements.txt` to `/icl_workdir/requirements.txt/` and then install them.
# We do this first because it's slow and each of these commands are cached in sequence.
COPY requirement.txt .
RUN pip3 install -r requirement.txt

# Copy the file `training.py` to `training.py/`
# You might need multiple of these statements to copy all the files you need for your experiment.
COPY training training/
COPY pile pile/


# Copy the folder `scripts` to `scripts/`
# You might need multiple of these statements to copy all the folders you need for your experiment.
# COPY scripts scripts/

