import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from modules.utils.distributed_weighted_sampler import DistributedWeightedSampler
import hashlib
import collections.abc as container_abcs
import re

string_classes = (str, bytes)
int_classes = int
INVALID_DATE_STR = "Date string not valid! Received {}, and got exception {}"
ISO_FORMAT = '%Y-%m-%dT%H:%M:%S'

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts, MoleculeDatapoint or lists; found {}")

def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        try:
            # Importing here so chemprop does not become a dependency of sandstone
            from chemprop.data import MoleculeDatapoint, MoleculeDataset
            if isinstance(elem, MoleculeDatapoint):
                data = MoleculeDataset(batch)
                data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules
                return data

        except ImportError:
            pass

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def ignore_None_collate(batch):
    '''
    default_collate wrapper that creates batches only of not None values.
    Useful for cases when the dataset.__getitem__ can return None because of some
    exception and then we will want to exclude that sample from the batch.
    '''
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def get_train_dataset_loader(args, train_data, batch_size):
    '''
        Given arg configuration, return appropriate torch.DataLoader
        for train_data and dev_data

        returns:
        train_data_loader: iterator that returns batches
        dev_data_loader: iterator that returns batches
    '''
    if args.class_bal:
        if args.strategy == 'ddp':
            sampler = DistributedWeightedSampler(train_data, weights=train_data.weights, replacement=True, rank=args.global_rank, num_replicas = args.world_size)
        else:
            sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
    else:
        if args.strategy == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True, rank=args.global_rank, num_replicas = args.world_size)
        else:
            sampler = data.sampler.RandomSampler(train_data)

    train_data_loader = data.DataLoader(
                    train_data,
                    num_workers=args.num_workers,
                    sampler=sampler,
                    pin_memory=True,
                    batch_size=batch_size,
                    drop_last=True,
                    collate_fn=ignore_None_collate)

    return train_data_loader


def get_eval_dataset_loader(args, eval_data, batch_size, shuffle):

    if args.strategy == 'ddp':
        sampler = torch.utils.data.distributed.DistributedSampler(eval_data, shuffle=shuffle, rank=args.global_rank, num_replicas = args.world_size)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(eval_data) if shuffle else torch.utils.data.sampler.SequentialSampler(eval_data)
    data_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=ignore_None_collate,
        pin_memory=True,
        drop_last=False,
        sampler = sampler)

    return data_loader


def is_listy(x):
    return isinstance(x, (tuple,list))

def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x):
        return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):
        return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    #  return res if x is None else retain_type(res, x)
    return res

def to_device(b, device):
    "Recursively put `b` on `device`."
    def _inner(o):
        return o.to(device, non_blocking=True) if isinstance(o,torch.Tensor) else o.to_device(device) if hasattr(o, "to_device") else o
    return apply(_inner, b)

def to_cpu(b):
    "Recursively map lists of tensors in `b ` to the cpu."
    return to_device(b,'cpu')
