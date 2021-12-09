import copy
from datetime import datetime as dt
import hashlib

def md5(key):
    '''
    returns a hashed with md5 string of the key
    '''
    return hashlib.md5(key.encode()).hexdigest()

def get_experiment_name(args):
    args_str = ''.join([ '{}-{} '.format(k, v) for k,v in vars(args).items() ])
    run_prefix= md5(args_str)
    return run_prefix


