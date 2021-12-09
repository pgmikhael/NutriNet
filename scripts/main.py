from os.path import dirname, realpath, join
import sys, os
sys.path.append((dirname(dirname(realpath(__file__)))))
sys.path.append(os.path.join( (dirname(dirname(realpath(__file__)))), 'chemprop_pkg'))
import os 

# comet
import comet_ml

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import _logger as log

# dataset
from modules.datasets.utils import get_dataset
from modules.utils.dataset import get_train_dataset_loader, get_eval_dataset_loader
from torch.utils.data import DataLoader

# model
from modules.utils.shared import get_object

#misc
import parsing
import git
import torch
import pdb
import warnings
import time
import pickle
import numpy as np
from argparse import Namespace

#Constants
DATE_FORMAT_STR = "%m-%d-%Y %H:%M:%S"
RESULTS_DATE_FORMAT = "%m%d%Y-%H%M%S"
SEED = 1111

def cli_main():
    pl.seed_everything(SEED)
    
    # ------------
    # set args
    # ------------
    parser = pl.Trainer.add_argparse_args(parsing.get_parser())
    args = parsing.parse_args(parser)
    args.checkpoint_callback = False
    
    # ------------
    # init trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank
    
    tb_logger = pl.loggers.CometLogger(api_key= os.environ.get('COMET_API_KEY'), \
                                                project_name= args.project_name, \
                                                experiment_name=args.experiment_name,\
                                                workspace= args.workspace,\
                                                log_env_details=True,\
                                                log_env_cpu=True)
    trainer.logger = tb_logger
    
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    log.info("\nProject main running by author: {} \ndate:{}, \nfrom commit: {} -- {}".format(
        commit.author, time.strftime(DATE_FORMAT_STR, time.localtime(commit.committed_date)),
        commit.hexsha, commit.message))

    # ------------
    # log args
    # ------------
    for key,value in sorted(vars(args).items()):
        log.info('{} -- {}'.format(key.upper(), value))
    
    # ------------
    # data
    # ------------
    if args.train or args.process_train_splits:
        log.info("\nLoading train and dev data...")
        train_data = get_dataset(args,  'train')
        train_loader = get_train_dataset_loader(args, train_data,  args.batch_size)
        dev_data = get_dataset(args, 'dev')
        val_loader = get_eval_dataset_loader(args, dev_data, args.batch_size, False)

    if args.test or args.predict:
        log.info("\nLoading test data...")
        test_data = get_dataset(args, 'test')
        #if not args.predict:
        #    train_data = get_dataset(args,  'train')
        #    log.info( test_data.get_majority_baseline(train_data) )
        test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)

    
    # ------------
    # model
    # ------------
    if args.from_checkpoint:
        snargs = Namespace(**pickle.load(open(args.checkpointed_path, 'rb')))
        model = get_object(snargs.lightning_model_name, 'lightning')(snargs)
        model = model.load_from_checkpoint(checkpoint_path= snargs.model_path, strict= not args.relax_checkpoint_matching )
        model.args = args
    else:
        model = get_object(args.lightning_model_name, 'lightning')(args)

    trainer.logger.experiment.set_model_graph(model)
    trainer.logger.experiment.add_tags(args.comet_tags)
    
    args.run_time = time.strftime(RESULTS_DATE_FORMAT, time.localtime())

    # ------------
    # training
    # ------------
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        dirpath= os.path.join(args.model_save_dir, args.experiment_name),
        mode='min' if 'loss' in args.monitor else 'max',
        filename= '{}'.format(args.experiment_name) + '{epoch}',
        every_n_epochs =1 )

    # -------------
    # add callbacks
    # -------------
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.train:
        log.info("\nTraining Phase...")
        trainer.fit(model, train_loader, val_loader)
        args.model_path = trainer.checkpoint_callback.best_model_path

    # ------------
    # testing
    # ------------
    trainer.args = args
    
    if args.test or args.predict:
        log.info("\nInference Phase on test set...")
        model.split_key = 'test'
        trainer.test(model, test_dataloaders=test_loader)

    if args.process_train_splits:
        log.info("\nInference Phase on development sets")
        model.split_key = 'eval_train'
        train_loader = get_eval_dataset_loader(args, train_data, args.batch_size, False)
        trainer.test(model, test_dataloaders=train_loader)
        model.split_key = 'val'
        trainer.test(model, test_dataloaders=val_loader)
    
    
    
    pickle.dump(vars(args), open('{}.args'.format(args.results_path), 'wb'))


if __name__ == '__main__':
    cli_main()
