# append project root to system path
import sys, os
from os.path import dirname, realpath
sys.path.append((dirname(dirname(realpath(__file__)))))
sys.path.append(os.path.join( (dirname(dirname(realpath(__file__)))), 'chemprop_pkg'))
import argparse
import subprocess
import multiprocessing
import pickle
import csv
import json
import random
import parsing
from  utils import md5
from collections import defaultdict
import pandas as pd


EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
SUCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}."
RESULT_KEY_STEMS = ['{}_loss', '{}_accuracy', '{}_auc', '{}_precision', '{}_recall', '{}_f1', '{}_mse', '{}_mae', '{}_r2']

LOG_KEYS = ['results_path', 'model_path', 'log_path']
SORT_KEY = 'val_loss'

parser = argparse.ArgumentParser(description='Dispatcher. For use information, see `doc/README.md`')
parser.add_argument('--config_path', type = str, required = True, default = 'configs/config_file.json', help = 'path to model configurations json file')
parser.add_argument('--log_dir', type=str, default="/path/to/results/dir", help="path to store logs and detailed job level result files")
parser.add_argument('--dry_run', '-n', action = 'store_true', default = False, help = "print out commands without running")        
parser.add_argument('--eval_train_config', '-e', action = 'store_true', default = False, help = "create evaluation run from a training config")        
#parser.add_argument('--result_path', type=str, default="/path/to/grid/search/file.csv", help="path to store grid_search table.")

def launch_experiment(script, gpu, flag_string):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.args".format(log_stem)

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u scripts/{}.py {} --results_path {} --experiment_name {}".format(
        gpu, script, flag_string, log_stem, log_name) #use log_stem instead of results_path, add extensions in main/learn.py

    # forward logs to logfile
    if "--resume" in flag_string:
        pipe_str = ">>"
    else:
        pipe_str = ">"

    shell_cmd = "{} {} {} 2>&1".format(experiment_string, pipe_str, log_path)
    print("Launched exp: {}".format(shell_cmd))

    if not os.path.exists(results_path) and (not args.dry_run):
        subprocess.call(shell_cmd, shell=True)

    return results_path, log_path

def worker(script, gpu, job_queue, done_queue): 
    '''
    DONE.
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(launch_experiment(script, gpu, params))

if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.config_path, 'r'))
    
    if args.eval_train_config:
        experiments, flags, experiment_axies = parsing.prepare_training_config_for_eval(experiment_config)
    else:
        experiments, flags, experiment_axies = parsing.parse_dispatcher_config(experiment_config)
    
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in experiments:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(experiments)))
    print()

    for gpu in experiment_config['available_gpus']:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(target=worker, args=(experiment_config['script'], gpu, job_queue, done_queue)).start()
        print()
    
    result_keys = []
    for mode in ['val','test']:
        result_keys.extend( [k.format(mode) for k in RESULT_KEY_STEMS ])

    summary = defaultdict(list)
    for i in range(len(experiments)):
        result_path, log_path = done_queue.get() #.rslt and .txt (stderr/out) files
        try:
            result_dict = pickle.load(open(result_path, 'rb'))
            dump_result_string = SUCESSFUL_SEARCH_STR.format(result_path)
            print("({}/{}) \t {}".format(i+1, len(experiments), dump_result_string))
        except Exception:
            print("Experiment failed! Logs are located at: {}".format(log_path))
