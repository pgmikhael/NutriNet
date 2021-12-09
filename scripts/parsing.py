"""
Command-Line Arguments
"""

import argparse
import itertools
import torch
from modules.utils.shared import get_object
from utils import get_experiment_name, md5
import os, json, copy

def get_parser():
     parser = argparse.ArgumentParser(description='ARGS')

     #-------------------------------------
     # Dataset Classes
     #-------------------------------------
     
     # Dataset
     parser.add_argument('--dataset', type = str, default = 'tcga_outcomes')
     parser.add_argument('--data_dir', type = str, default = 'Data/')
     parser.add_argument('--batch_size', type = int, default = 10, help = 'batch size')
     parser.add_argument('--class_bal', action='store_true', default=False, help = 'class balance')
     parser.add_argument('--assign_data_splits', action='store_true', default = False, help = 'assign data splits with cross_val_seed')
     parser.add_argument('--cross_val_seed', type = int, default = 0, help = 'cross validation seed')
     parser.add_argument('--split_probs', type = int, nargs = '*', default = [0.7, 0.15, 0.15], help = 'dataset splits probs')
     parser.add_argument('--num_classes', type = int, help = 'Number of classes for task. Typically defined by dataset object')
     parser.add_argument('--max_sequence_length', type = int, default = 512, help = 'Length of longest sequence')

     parser.add_argument('--input_loader_name',  type=str, default='default_image_loader', help='input loader')
     
     #-------------------------------------
     # Modeling
     #-------------------------------------

     # Lightning
     parser.add_argument('--lightning_model_name', type = str, default = 'vgg', help = 'Name of DNN')

     # Model
     parser.add_argument('--base_model', type = str, default = 'vgg', help = 'Name of parent model')

     parser.add_argument('--dropout', type = float, default = 0.25, help = 'dropout probability')

     # Loss Lambdas
     parser.add_argument('--loss_lambda', type = float, default = 1.0, help = 'weight for task loss')
     parser.add_argument('--lasso_loss_lambda', type = float, default = 1.0, help = 'weight for l1 loss')
     
     # Embeddings
     parser.add_argument('--glove_name', type = str, default = '6B', help = 'Glove dataset to use')
     parser.add_argument('--hf_tokenizer_name', type = str, default = 'distilbert-base-uncased', help = 'Hugging face tokenizer to use')
     parser.add_argument('--hf_model_name', type = str, default = 'distilbert-base-uncased', help = 'Hugging face fixed model to use')
     parser.add_argument('--embed_size', type = int, default = 128, help = 'Embedding dimension to use when not preloading')
     parser.add_argument('--use_embeddings', action='store_true', default = False, help = 'Whether to use pretrained embeddings')
     parser.add_argument('--use_bert_embeddings', action='store_true', default = False, help = 'Whether to use pretrained embeddings')

     # Model Params
     parser.add_argument('--hidden_dim', type = int, default = 512, help = 'Dim of hidden layer')
     
     parser.add_argument('--mlp_layer_configuration', type = int, nargs='*', default = [128, 128], help = 'MLP layer dimensions')
     
     # RNN
     parser.add_argument('--num_layers', type = int, default = 3, help = 'Number of layers')
     parser.add_argument('--bidirectional', action='store_true', default = False, help = 'Use bidirectional mechanism')

     # Transformers
     parser.add_argument('--dim_feedforward', type = int, default = 128, help = 'Dim of feedforward net in transformer')
     parser.add_argument('--num_heads', type = int, default = 3, help = 'Number of attention heads')
     parser.add_argument('--num_encoder_layers', type = int, default = 3, help = 'Number of encoder layers')

     #-------------------------------------
     # Learning
     #-------------------------------------
     
     parser.add_argument('--train', action='store_true', default = False, help = 'whether training model')
     parser.add_argument('--test', action='store_true', default = False, help = 'whether testing model')
     parser.add_argument('--process_train_splits', action='store_true', default = False, help = 'whether testing model on train and val splits')
     parser.add_argument('--predict', action='store_true', default = False, help = 'whether using model to predict on new data with unknown targets')
     
     parser.add_argument('--resume', action='store_true', default = False, help = 'whether to resume from previous run')

     # Name of losses and metrics to use
     parser.add_argument('--loss_fns', type=str, nargs = '*', default=[], help="Name of loss")
     parser.add_argument('--eval_loss_fns', type=str, nargs = '*', default=None, help="Name of loss")
     parser.add_argument('--metrics', type=str, nargs = '*', default= [], help="Name of performance metric")
     parser.add_argument('--store_classwise_metrics', action='store_true', default = False, help = 'Whether to log metrics per class or just log average across classes')
     # Checkpointing
     parser.add_argument('--monitor', type=str, default='val_auc', help="Name of metric to use to decide when to save model")
     
     # Optimizers
     parser.add_argument('--optimizer', type = str, default = 'adam', help = 'optimizer function')
     parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
     parser.add_argument('--lr_decay', type = float, default = 1, help = 'how much to reduce lr by when getting closer to optimum')
     parser.add_argument('--weight_decay', type = float, default = 1, help = 'l2 penalty coefficient')
     parser.add_argument('--momentum', type = float, default = 0.99, help = 'optimizer momentum')
     parser.add_argument('--l1_decay', type = float, default = 0.1, help = 'l1 penalty coefficient')
     parser.add_argument('--lars_eta', type = float, default = 0.001)
     parser.add_argument('--weight_decay_filter', action='store_true', default=False, help = 'Whether to decay weight')
     parser.add_argument('--lars_adaptation_filter', action='store_true', default=False, help = 'Whether to scale by eta')

     # Scheduler
     parser.add_argument('--scheduler', type=str, default='reduce_on_plateau', help="Name of scheduler")
     parser.add_argument('--patience', type = int, default= 10, help = 'how much to wait before reducing lr') 
     parser.add_argument('--cosine_annealing_period', type = int, default= 10, help = 'length of period of lr cosine anneal') 
     parser.add_argument('--cosine_annealing_period_scaling', type = int, default= 2, help = 'how much to multiply each period in successive annealing') 

     #-------------------------------------
     # System
     #-------------------------------------

     # Workers and GPUS
     parser.add_argument('--num_workers',type = int, default=4, help = "Number of workers to use with dataloading. Check number of CPUs on machine first")
    
     # Loading Saved Models
     parser.add_argument('--from_checkpoint',action='store_true', default=False, help = "Whether loading a model from a saved checkpoint")
     parser.add_argument('--model_save_dir', type = str, help = "Dir to save model")
     parser.add_argument('--checkpointed_path', type = str, help = "Path to previously saved model")
     parser.add_argument('--relax_checkpoint_matching', action='store_true', default=False, help = "Do not enforce that the keys in checkpoint_path match the keys returned by this module’s state dict")
     parser.add_argument('--save_predictions', action='store_true', default=False, help = "Whether to save predictions dictionary")
     parser.add_argument('--save_hiddens', action='store_true', default=False, help = "Whether to save representations images")
     parser.add_argument('--hiddens_dir', type = str, help="Path to results files. Keep undefined if using dispatcher.py, which will set this automatically")

     # Directories and Files
     parser.add_argument('--experiment_name', type = str, help = 'defined either automatically by dispatcher.py or time in main.py. Keep without default')
     parser.add_argument('--results_path', type = str, help = 'defined either automatically by dispatcher.py or time in main.py. Keep without default')

     #-------------------------------------
     # Comet
     #-------------------------------------
     
     parser.add_argument('--comet_tags', nargs='*', default=[], help="List of tags for comet logger")
     parser.add_argument('--project_name', default='CancerCures', help="Comet project")
     parser.add_argument('--workspace', default='pgmikhael', help="Comet workspace")

     return parser

def parse_args(parser):
     args = parser.parse_args()
     
     # Set args particular to dataset
     get_object(args.dataset, 'dataset').set_args(args)

     # define if cuda device
     if hasattr(args, 'gpus') and ( (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (isinstance(args.gpus, int) and  args.gpus > 1) ):
        args.strategy = 'ddp'
    
     # get experiment name
     args.experiment_name = get_experiment_name(args) if args.experiment_name is None else args.experiment_name
     
     return args


def parse_dispatcher_config(config):
     '''
     Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
     but the config contains a list, this will return one job for each item in the list.
     :config - experiment_config

     returns: jobs - a list of flag strings, each of which encapsulates one job.
          *Example: --train --cuda --dropout=0.1 ...
     returns: experiment_axies - axies that the grid search is searching over
     '''

     grid_search_spaces = config['grid_search_space']
     paired_search_spaces = config.get('paired_search_space', [])
     flags = []
     arguments = []
     experiment_axies = []

     # add anything outside search space as fixed
     fixed_args = ""
     for arg in config: 
          if arg not in ['script', 'grid_search_space', 'paired_search_space', 'available_gpus']:
               if type(config[arg]) is bool:
                    if config[arg]:
                         fixed_args += '--{} '.format(str(arg))
                    else:
                         continue
               else:
                    fixed_args += '--{} {} '.format(arg, config[arg])

     # add paired combo of search space
     paired_args_list = ['']
     if len(paired_search_spaces) > 0:
         paired_args_list = []
         paired_keys = list(paired_search_spaces.keys())
         paired_vals = list(paired_search_spaces.values())
         flags.extend(paired_keys)
         for paired_combo in zip(*paired_vals):
              paired_args = ""
              for i, flg_value in enumerate(paired_combo):
                   if type(flg_value) is bool:
                        if flg_value:
                             paired_args += '--{} '.format(str(paired_keys[i]))
                        else:
                             continue
                   else:
                        paired_args += '--{} {} '.format(str(paired_keys[i]), str(flg_value))
              paired_args_list.append(paired_args)

     # add every combo of search space
     product_flags = []
     for key, value in grid_search_spaces.items():
          flags.append(key)
          product_flags.append(key)
          arguments.append(value)
          if len(value) > 1:
               experiment_axies.append(key)

     experiments = []
     exps_combs = list(itertools.product(*arguments))

     for tpl in exps_combs:
          exp = ""
          for idx, flg in enumerate(product_flags):
               if type(tpl[idx]) is bool:
                    if tpl[idx]:
                         exp += '--{} '.format(str(flg))
                    else:
                         continue
               else:
                    exp += '--{} {} '.format(str(flg), str(tpl[idx]))
          exp += fixed_args
          for paired_args in paired_args_list:
               experiments.append(exp+paired_args)

     return experiments, flags, experiment_axies

def prepare_training_config_for_eval(train_config):
     """Convert training config to an eval config for testing.

     Parameters
     ----------
     train_config: dict
          config with the following structure:
               {
                    "train_config": ,   # path to train config
                    "log_dir": ,        # log directory used by dispatcher during training
                    "eval_args": {}     # test set-specific arguments beyond default
               }

     Returns
     -------
     experiments: list
     flags: list
     experiment_axies: list
     """

     train_args = json.load(open(train_config['train_config'], 'r'))

     experiments, _, _ = parse_dispatcher_config(train_args)
     stem_names = [md5(e) for e in experiments]
     eval_args = copy.deepcopy(train_args)
     
     # reset defaults
     eval_args['train'] = [False]
     eval_args['test'] = [True]
     eval_args['from_checkpoint'] = [True]
     eval_args['gpus'] = [1]
     eval_args['gpus'] = [1]
     eval_args['comet_tags'].append('eval')
     
     experiments, flags, experiment_axies = parse_dispatcher_config(eval_args)
     
     for e, s in zip(experiments, stem_names):
          e = e + ' --checkpointed_path {}'.format(os.path.join(train_config['log_dir'], '{}.args'.format(s) ))

     return experiments, flags, experiment_axies
