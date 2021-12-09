from modules.utils.shared import get_object
import datetime
import json
import os
import numpy as np
import torch
from torch.utils import data
import hashlib
import random
from collections import defaultdict


def get_dataset(args,  split_group):
    dataset_class = get_object(args.dataset, 'dataset')
    dataset  =  dataset_class(args,  split_group)
    
    return dataset
