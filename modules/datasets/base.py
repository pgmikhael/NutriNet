import argparse

import numpy as np
import pickle
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import traceback
from collections import Counter, defaultdict
from modules.datasets.loader.factory import get_input_loader
import pdb
from random import Random

from typing import List

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"

class Abstract_Dataset(data.Dataset):
    """
    Abstract Dataset Object for all Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, a check
    label, a skip label and get label function. The __getitem__ class is the
    most important method as it returns the item to be batched instead of
    loading any large data into memory.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args: argparse.ArgumentParser, split_group: str) -> None:
        '''
        params: args - the arguments from the config file after parsing (see scripts/parsing.py)
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed to a DataLoader for batching
        '''
        super(Abstract_Dataset, self).__init__()

        if self.task not in self.supported_tasks:
            raise NotImplementedError('TASK {} NOT SUPPORTED'.format( self.task.upper() ))

        args.metadata_path = os.path.join(args.data_dir, self.METADATA_FILENAME)
        self.input_loader = get_input_loader(args)
        
        self.split_group = split_group
        self.args = args
        try:
            self.metadata_json = json.load(open(args.metadata_path, 'r'))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))

        self.dataset = self.create_dataset(split_group)

        if len(self.dataset) == 0:
            return

        self.get_summary_statement(split_group)
        
        if args.class_bal:
            label_dist = [str(d['y']) for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1./ len(label_counts)
            label_weights = { label: weight_per_label/count for label, count in label_counts.items()}
            print("Label weights are {}".format(label_weights))
            self.weights = [ label_weights[str(d['y'])] for d in self.dataset]


    @property
    @abstractmethod
    def SUMMARY_STATEMENT(self) -> None:
        """
        Prints summary statement with dataset stats
        """
        pass
    
    @abstractmethod
    def get_majority_baseline(self, train_data) -> dict:
        """
        Gets the stats for a basic model that just predicts the majority baseline.
        :param train_data: training data from which to get these stats
        :return: dict of stats
        """
        pass
    
    @property
    def LOAD_FAIL_MSG(self):
        return "Failed to load input: {}\nException: {}"

    @property
    def DATASET_ITEM_KEYS(self):
        return ['y', 'mol', 'sample_id', 'rdkit_features', 'condensate', 'compound', 'droplet_partitioning', 'relative_class', 'colocalization', 'partition', 'dispartition']
    

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @abstractmethod
    def check_label(self, row: int) -> bool:
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        :returns" true if row contains valid label for the task
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        :returns: see concrete implementations for return values
        '''
        pass

    def get_summary_statement(self, split_group):
        '''
        Prints a summary statement
        '''
        print('\n')
        print("{} DATASET {} CREATED.\n{}".format(split_group.upper(), self.args.dataset.upper(), self.SUMMARY_STATEMENT))

    @abstractmethod
    def create_dataset(self, split_group: str) -> list:
        """
        Creates the dataset from the paths and labels in the json metadata file.
        :split_group: - ['train'|'dev'|'test'].
        """
        pass


    @staticmethod
    def set_args(args):
        """
        Sets any args particular to the dataset.
        Warning: this will reset the args for all subsequent steps. Args is global to the system and mutating it
        can have side effects.
        """
        pass

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, index):
        pass
    
    def scaffold_split(self, meta: List[dict]):
        scaffold_to_indices = defaultdict(list)
        for m_i, m in enumerate(meta):
            scaffold_to_indices[m['scaffold']].append(m_i)
       
        # Split
        train_size, val_size, test_size = self.args.split_probs[0] * len(meta), self.args.split_probs[1] * len(meta), self.args.split_probs[2] * len(meta)
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

        # Seed randomness
        random = Random(self.args.cross_val_seed)

        if self.args.scaffold_balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
            index_sets = list(scaffold_to_indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(self.args.cross_val_seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:  # Sort from largest to smallest scaffold sets
            index_sets = sorted(list(scaffold_to_indices.values()),
                                key=lambda index_set: len(index_set),
                                reverse=True)

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

        for idx_list, split in [(train, 'train'), (val, 'dev'), (test, 'test')]:
            for idx in idx_list:
                meta[idx]['split'] = split
