from typing import Dict
from modules.utils.shared import register_object
from collections import OrderedDict
import numpy as np
import pdb
from torchmetrics.functional import auc, accuracy, auroc, precision_recall, confusion_matrix, f1, mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve
import torch
import copy 

EPSILON = 1e-6
BINARY_CLASSIF_THRESHOLD = 0.5


@register_object("classification", 'metric')
class BaseClassification(object):
    def __init__(self, args) -> None:
        super().__init__()
    
    @property
    def metric_keys(self):
        return ['probs', 'preds', 'golds']
    
    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes standard classification metrics

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'preds', 'golds']
            args: argparser Namespace
        
        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc 

        Note:
            In multiclass setting (>2), accuracy, and micro-f1, micro-recall, micro-precision are equivalent
            Macro: calculates metric per class then averages
        '''
        stats_dict = OrderedDict()

        probs = logging_dict['probs'] # B, C (float)
        preds = logging_dict['preds'] # B
        golds = logging_dict['golds'] # B
        stats_dict['accuracy'] = accuracy(golds, preds)
        stats_dict['confusion_matrix'] = confusion_matrix(preds, golds, args.num_classes) 
        if args.num_classes == 2:
            if len(probs.shape) == 1:
                stats_dict['precision'], stats_dict['recall'] = precision_recall(probs, golds)
                stats_dict['f1'] = f1(probs, golds)
                pr, rc, _ = precision_recall_curve(probs, golds)
                stats_dict['pr_auc'] = auc(rc, pr)
                try:
                    stats_dict['roc_auc'] = auroc( probs, golds, pos_label = 1)
                except:
                    pass
            else:
                stats_dict['precision'], stats_dict['recall'] = precision_recall(probs, golds, multiclass = False, num_classes = 2)
                stats_dict['f1'] = f1(probs, golds, multiclass = False, num_classes = 2)
                pr, rc, _ = precision_recall_curve(probs, golds, num_classes = 2)
                stats_dict['pr_auc'] = auc(rc[-1], pr[-1])
                try:
                    stats_dict['roc_auc'] = auroc( probs, golds, num_classes = 2)
                except:
                    pass
        else:
            stats_dict['precision'], stats_dict['recall'] = precision_recall(probs, golds, num_classes = args.num_classes, average = 'macro')
            stats_dict['f1'] = f1(probs, golds, num_classes = args.num_classes, average = 'macro')
            stats_dict['micro_f1'] = f1(probs, golds, num_classes = args.num_classes, average = 'micro')
            if len(torch.unique(golds)) == args.num_classes:
                pr, rc, _ = precision_recall_curve(probs, golds, num_classes = args.num_classes)
                stats_dict['pr_auc'] = torch.mean( torch.stack( [ auc(rc[i], pr[i]) for i in range(args.num_classes) ]) )
                stats_dict['roc_auc'] = auroc( probs, golds, num_classes = args.num_classes, average = 'macro')
            
            if args.store_classwise_metrics:
                classwise_metrics = {}
                classwise_metrics['precisions'], classwise_metrics['recalls'] = precision_recall(probs, golds, num_classes = args.num_classes, average = 'none')
                classwise_metrics['f1s'] = f1(probs, golds, num_classes = args.num_classes, average = 'none')
                pr, rc, _ = precision_recall_curve(probs, golds, num_classes = args.num_classes)
                classwise_metrics['pr_aucs'] = [auc(rc[i], pr[i]) for i in range(args.num_classes)]
                classwise_metrics['accs'] = accuracy(golds, preds,num_classes=args.num_classes,average = 'none')
                try:
                    classwise_metrics['rocaucs'] = auroc( probs, golds, num_classes = args.num_classes, average = 'none')
                except:
                    pass

                for metricname in ['precisions', 'recalls',  'f1s', 'rocaucs', 'pr_aucs', 'accs']:
                    if metricname in classwise_metrics:
                        stats_dict.update({'class{}_{}'.format(i+1,metricname): v for i,v in enumerate(classwise_metrics[metricname]) })
        return stats_dict

@register_object("multi_class_classification", 'metric')
class MultiClass_Classification(BaseClassification):
    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes classification for metrics when predicting multiple independent classes

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
            args: argparser Namespace
        
        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc, prefixed by col index
        '''

        stats_dict = OrderedDict()
        golds = logging_dict['golds']
        preds = logging_dict['preds']
        probs = logging_dict['probs']
        targs = copy.deepcopy(args)
        targs.num_classes = 2

        tempstats_dict = OrderedDict()
        for classindex in range(golds.shape[-1]):
            minilog = {'probs': probs[:,classindex] , 'preds': preds[:, classindex], 'golds': golds[:, classindex] }
            ministats = super().__call__(minilog, targs)
            tempstats_dict.update( {'class{}_{}'.format(classindex, k): v for k,v in ministats.items() } )
            if args.store_classwise_metrics:
                stats_dict.update(tempstats_dict)

        for metric in ministats.keys():
            if not metric in ['confusion_matrix']:
                stats_dict[metric] = torch.stack( [tempstats_dict[k] for k in tempstats_dict.keys() if k.endswith(metric) ] ).mean()
        
        return stats_dict

@register_object("multi_task_classification", 'metric')
class MultiTask_Classification(BaseClassification):
    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes classification for metrics when predicting multiple tasks, where each task has an associated classifier

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['[TASK NAME]_probs', '[TASK NAME]_preds', '[TASK NAME]_golds'] for each task
            args: argparser Namespace
        
        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc, prefixed by the task name
        '''
        stats_dict = OrderedDict()
        for keyindex, key in enumerate(args.multitask_keys):
            args.num_classes = args.multitask_num_classes[keyindex]
            minilog = {'probs': logging_dict['{}_probs'.format(key)] , 'preds': logging_dict['{}_preds'.format(key)] , 'golds': logging_dict['{}_golds'.format(key)] }
            ministats = super().__call__(minilog, args)
            stats_dict.update( {'{}_{}'.format(key, k): v for k,v in ministats.items() } )

        for metric in ministats.keys():
            if not metric in ['confusion_matrix']:
                stats_dict[metric] = torch.stack( [stats_dict[k] for k in stats_dict.keys() if k.endswith(metric) ] ).mean()
        
        return stats_dict

@register_object("regression", 'metric')
class BaseRegression(object):
    def __init__(self, args) -> None:
        super().__init__()
    
    @property
    def metric_keys(self):
        return ['probs', 'golds']
    
    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes standard regresssion loss

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['probs', 'golds']
            args: argparser Namespace
        
        Returns:
            stats_dict (dict): contains (where applicable) values for mse, mae, r2
        '''
        stats_dict = OrderedDict()

        probs = logging_dict['probs']
        golds = logging_dict['golds']

        stats_dict['mse'] = mean_squared_error(probs, golds)
        stats_dict['mae'] = mean_absolute_error(probs, golds)
        stats_dict['r2'] = r2_score(probs, golds)

        return stats_dict

@register_object("multi_class_regression", 'metric')
class MultiTaskRegression(BaseRegression):
    def __call__(self, logging_dict, args) -> Dict:
        '''
        Computes regression for metrics when predicting multiple tasks, where each task has an associated classifier

        Args:
            logging_dict: dictionary obtained from computing loss and model outputs
                * should contain the keys ['[TASK NAME]_probs', '[TASK NAME]_preds', '[TASK NAME]_golds'] for each task
            args: argparser Namespace
        
        Returns:
            stats_dict (dict): contains (where applicable) values for accuracy, confusion matrix, precision, recall, f1, precision-recall auc, roc auc, prefixed by the task name
        '''

        stats_dict = OrderedDict()
        for j in range(args.num_classes):
            minilog = {'probs': logging_dict['probs'][:,j] ,'golds': logging_dict['golds'][:,j] }
            ministats = super().__call__(minilog, args)
            stats_dict.update( {'class{}_{}'.format(j,k): v for k,v in ministats.items() } )

        for metric in ministats.keys():
            stats_dict[metric] = torch.stack( [stats_dict[k] for k in stats_dict.keys() if k.endswith(metric) ] ).mean()
        
        return stats_dict
