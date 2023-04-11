#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This script extracts fp32 consolidated weights from a zero 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
#
# example: python zero_to_fp32.py . pytorch_model.bin
import torch
import glob
import math
import os
import re
from collections import OrderedDict


debug = 0

SINGLE_PARTITION_OF_FP32_GROUPS = 'single_partition_of_fp32_groups'
OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
FP32_FLAT_GROUPS = 'fp32'
ZERO_STAGE = 'zero_stage'
PARTITION_COUNT = 'partition_count'
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'

# load to cpu
device = torch.device('cpu')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_model_state_file(checkpoint_dir, zero_stage):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    # there should be only one file
    if zero_stage == 2:
        file = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    elif zero_stage == 3:
        file = os.path.join(checkpoint_dir, "zero_pp_rank_0_mp_rank_00_model_states.pt")

    if not os.path.exists(file):
        raise FileNotFoundError(f"can't find model states file at '{file}'")

    return file


def get_optim_files(checkpoint_dir):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    optim_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*_optim_states.pt")), key=natural_keys)

    if len(optim_files) == 0:
        raise FileNotFoundError(f"can't find '*_optim_states.pt' files in directory '{checkpoint_dir}'")

    return optim_files


def parse_optim_states(files, ds_checkpoint_dir):

    total_files = len(files)
    state_dicts = []
    for f in files:
        state_dicts.append(torch.load(f, map_location=device))

    if not ZERO_STAGE in state_dicts[0][OPTIMIZER_STATE_DICT]:
        raise ValueError(f"{files[0]} is not a zero checkpoint")
    zero_stage = state_dicts[0][OPTIMIZER_STATE_DICT][ZERO_STAGE]
    world_size = state_dicts[0][OPTIMIZER_STATE_DICT][PARTITION_COUNT]

    # For ZeRO-2 each param group can have different partition_count as data parallelism for expert
    # parameters can be different from data parallelism for non-expert parameters. So we can just
    # use the max of the partition_count to get the dp world_size.

    if type(world_size) is list:
        world_size = max(world_size)

    if world_size != total_files:
        raise ValueError(
            f"Expected {world_size} of '*_optim_states.pt' under '{ds_checkpoint_dir}' but found {total_files} files. "
            "Possibly due to an overwrite of an old checkpoint, or a checkpoint didn't get saved by one or more processes."
        )

    # the groups are named differently in each stage
    if zero_stage == 1:
        zero_stage = 2
        SINGLE_PARTITION_OF_FP32_GROUPS = 'local_sub_partitions_of_fp32_groups'
    if zero_stage == 2:
        fp32_groups_key = SINGLE_PARTITION_OF_FP32_GROUPS
    elif zero_stage == 3:
        fp32_groups_key = FP32_FLAT_GROUPS
    else:
        raise ValueError(f"unknown zero stage {zero_stage}")
    
    if zero_stage == 2:
        fp32_flat_groups = [state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key] for i in range(len(state_dicts))]
        base_opt_state = [state_dicts[i][OPTIMIZER_STATE_DICT]['base_optimizer_state'] for i in range(len(state_dicts))]
        # unpack base opt state
        optimizer_vars = [state_dicts[i][OPTIMIZER_STATE_DICT]['base_optimizer_state'] for i in range(len(state_dicts))]
        # exp_avg_sq = []
        # for state in base_opt_state:
        #     for substate in state:
        #         exp_avg.append(substate[0]['exp_avg'])
        #         exp_avg_sq.append(substate[0]['exp_avg_sq'])
    elif zero_stage == 3:
        # if there is more than one param group, there will be multiple flattened tensors - one
        # flattened tensor per group - for simplicity merge them into a single tensor
        #
        # XXX: could make the script more memory efficient for when there are multiple groups - it
        # will require matching the sub-lists of param_shapes for each param group flattened tensor

        fp32_flat_groups = [
            torch.cat(state_dicts[i][OPTIMIZER_STATE_DICT][fp32_groups_key], 0) for i in range(len(state_dicts))
        ]

    return zero_stage, world_size, fp32_flat_groups, state_dicts[0]['param_shapes'], optimizer_vars #exp_avg, exp_avg_sq



def _get_fp32_state_dict_from_zero2_checkpoint(world_size, param_shapes, fp32_flat_groups, key, buffers, shared_params):

    # Reconstruction protocol:
    #
    # XXX: document this
    debug = True

    # if debug:
    #     for i in range(world_size):
    #         for j in range(len(fp32_flat_groups[0])):
    #             print(f"{FP32_FLAT_GROUPS}[{i}][{j}].shape={fp32_flat_groups[i][j].shape}")

    # XXX: memory usage doubles here (zero2)
    num_param_groups = len(fp32_flat_groups[0])
    merged_single_partition_of_fp32_groups = []
    for i in range(num_param_groups):
        merged_partitions = [sd[i][0][key] for sd in fp32_flat_groups]        
        full_single_fp32_vector = torch.cat(merged_partitions, 0)
        merged_single_partition_of_fp32_groups.append(full_single_fp32_vector)
        
    print('merged')
    avail_numel = sum(
        [full_single_fp32_vector.numel() for full_single_fp32_vector in merged_single_partition_of_fp32_groups])

    # if debug:
    #     wanted_params = sum([len(shapes) for shapes in param_shapes.values()])
    #     wanted_numel = sum([sum(shape.numel() for shape in shapes) for shapes in param_shapes.values()])
    #     # not asserting if there is a mismatch due to possible padding
    #     print(f"Have {avail_numel} numels to process.")
    #     print(f"Need {wanted_numel} numels in {wanted_params} params.")

    state_dict = OrderedDict()

    # # buffers
    # state_dict.update(buffers)
    # if debug:
    #     print(f"added {len(buffers)} buffers")

    # params
    # XXX: for huge models that can't fit into the host's RAM we will have to recode this to support
    # out-of-core computing solution
    # partition the shapes into two groups: the layernorm/biases and everything else.
    param_shapes = [
        {name: shape for name, shape in param_shapes.items() if 'norm' not in name and 'bias' not in name},
        {name: shape for name, shape in param_shapes.items() if 'norm' in name or 'bias' in name}
    ]
    for shapes, full_single_fp32_vector in zip(param_shapes, merged_single_partition_of_fp32_groups):
        
        avail_numel = full_single_fp32_vector.numel()
        total_numel = 0
        total_params = 0
        offset = 0
        for name, shape in shapes.items():

            unpartitioned_numel = shape.numel()
            total_numel += unpartitioned_numel
            total_params += 1

            if True:
                print(f"{name} full shape: {shape} unpartitioned numel {unpartitioned_numel} offset {offset}")
            state_dict[name] = full_single_fp32_vector.narrow(0, offset, unpartitioned_numel).view(shape)
            offset += unpartitioned_numel

        # Z2 started to align to 2*world_size to improve nccl performance. Therefore both offset and
        # avail_numel can differ by anywhere between 0..2*world_size. Due to two unrelated complex
        # paddings performed in the code it's almost impossible to predict the exact numbers w/o the
        # live optimizer object, so we are checking that the numbers are within the right range
        align_to = 2 * world_size

        def zero2_align(x):
            return align_to * math.ceil(x / align_to)

        if debug:
            print(f"original offset={offset}, avail_numel={avail_numel}")

        offset = zero2_align(offset)
        avail_numel = zero2_align(avail_numel)

        if debug:
            print(f"aligned  offset={offset}, avail_numel={avail_numel}")

        # Sanity check
        if offset != avail_numel:
            raise ValueError(f"consumed {offset} numels out of {avail_numel} - something is wrong")

    # recover shared parameters
    # for pair in shared_params:
    #     state_dict[pair[0]] = state_dict[pair[1]]

    print(f"Reconstructed fp32 state dict with {total_params} params {total_numel} elements")

    return state_dict


def get_optimizer_state_from_checkpoint(ds_checkpoint_dir, model, optimizer):
    """
    Returns fp32 state_dict reconstructed from ds checkpoint

    Args:
        - ``ds_checkpoint_dir``: path to the deepspeed checkpoint folder (where the optimizer files are)

    """
    print(f"Processing zero checkpoint '{ds_checkpoint_dir}'")
    optim_files = get_optim_files(ds_checkpoint_dir)
    zero_stage, world_size, fp32_flat_groups, param_shapes, optimizer_vars = parse_optim_states(optim_files, ds_checkpoint_dir)
    param_shapes = {k: v for k, v in param_shapes.items()}
    print(f"Detected checkpoint of type zero stage {zero_stage}, world_size: {world_size}")
    # grab train step
    step = torch.load(os.path.join(ds_checkpoint_dir, 'mp_rank_00_model_states.pt'))['global_steps']

    # model_file = get_model_state_file(ds_checkpoint_dir, zero_stage)
    # buffers, param_shapes, shared_params, ds_version = parse_model_state(model_file)
    # print(f'Parsing checkpoint created by deepspeed=={ds_version}')
    print('exp avg...')
    exp_avg = _get_fp32_state_dict_from_zero2_checkpoint(world_size, param_shapes, optimizer_vars, 'exp_avg', None, None)
    print('exp avg square...')
    exp_avg_sq = _get_fp32_state_dict_from_zero2_checkpoint(world_size, param_shapes, optimizer_vars, 'exp_avg_sq', None, None)
    print('plumbing into optimizer...')
    # to make life easier, load in a model.
    input_dummy = torch.tensor([[1, 2, 3]])
    loss = model(input_dummy, labels=input_dummy).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    state_dict = optimizer.state_dict()
    # get the right ordering of params
    no_decay = ["bias", "layer_norm.weight"]
    param_groups = [
        # [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        # [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    ]
    param_groups = [n for n, _ in model.named_parameters()] #param_groups[0] + param_groups[1]
    def convert_hf_name(name):
        name = name.replace('gpt_neox.', '')
        name = name.replace('layers.', '')
        if name[0].isdigit():
            layer_num = int(name.split(".")[0]) + 2
            name = ".".join(name.split(".")[1:])
            name = f"{layer_num}.{name}"
        elif name == 'final_layer_norm.weight':
            name = '15.norm.weight'
        elif name == 'final_layer_norm.bias':
            name = '15.norm.bias'
        elif name == 'embed_out.weight':
            name = '16.final_linear.weight'
        elif name == 'embed_in.weight':
            name = '0.word_embeddings.weight'
        else:
            import pdb; pdb.set_trace()
        return name

    for i, name in enumerate(param_groups):
        converted_name = convert_hf_name(name)
        if state_dict['state'][i]['exp_avg'].shape != exp_avg[converted_name].shape:
            import pdb; pdb.set_trace()
        assert state_dict['state'][i]['exp_avg'].shape == exp_avg[converted_name].shape, f"{state_dict['state'][i]['exp_avg'].shape} != {exp_avg[converted_name].shape}: {converted_name}"
        assert state_dict['state'][i]['exp_avg_sq'].shape == exp_avg_sq[converted_name].shape, f"{state_dict['state'][i]['exp_avg_sq'].shape} != {exp_avg_sq[converted_name].shape}: {name}"
        state_dict['state'][i]['exp_avg'] = exp_avg[converted_name]
        state_dict['state'][i]['exp_avg_sq'] = exp_avg_sq[converted_name]
        state_dict['state'][i]['step'] = step
    return state_dict, step
