from modules.utils.shared import register_object
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import pdb

EPSILON = 1e-6

@register_object("cross_entropy", 'loss')
def get_ce_loss(model_output, batch, model, args):
    '''
    Computes cross-entropy loss 

    If batch contains they key 'has_y', the cross entropy loss will be computed for samples where batch['has_y'] = 1
    Expects model_output to contain 'logit'

    Returns:
        loss: cross entropy loss
        l_dict (dict): dictionary containing cross_entropy_loss detached from computation graph
        p_dict (dict): dictionary of model predictions and ground truth labels (preds, probs, golds)
    '''
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    logit = model_output['logit']
    B, C = logit.shape
    batch['y'] = batch['y'].long()
    if C > 1:
        if not args.predict:
            loss = (F.cross_entropy(logit, batch['y'], reduction = 'none') * batch['has_y']).sum()/max(1, batch['has_y'].sum()) if 'has_y' in batch else F.cross_entropy(logit, batch['y'])
        logit = torch.softmax(logit, dim = -1)
        probs, preds = torch.topk(logit, k = 1)
        p_dict['probs'], p_dict['preds'] = logit.detach(), preds.view(B).detach().view(-1)
    else:
        if not args.predict:
            loss = (nn.BCEWithLogitsLoss(logit, batch['y'].unsqueeze(1).float()) * batch['has_y']).sum()/max(1, batch['has_y'])  if 'has_y' in batch else nn.BCEWithLogitsLoss(logit, batch['y'].unsqueeze(1).float())  # compute loss
        p_dict['probs'] = torch.sigmoid(logit).detach().view(-1)
        p_dict['preds'] = (torch.sigmoid(logit) > 0.5).detach().view(-1)
    
    p_dict['golds'] = batch['y']
    if 'has_y' in batch:
        p_dict['has_y'] = batch['has_y']

    if not args.predict:
        l_dict['cross_entropy_loss'] = loss.detach()

    return loss * args.loss_lambda, l_dict, p_dict

@register_object("multi_class_bce_loss", 'loss')
def get_multiclass_bce_loss(model_output, batch, model, args):
    '''
    Computes cross-entropy loss for model with multiple classes where classes are independent (ie more than one class can be positive)

    Expects model_output to contain 'logit'
    
    Returns:
        loss: cross entropy loss
        l_dict (dict): dictionary containing bce_loss detached from computation graph
        p_dict (dict): dictionary of model predictions and ground truth labels (preds, probs, golds)
    '''
    loss = 0
    l_dict, p_dict = OrderedDict(), OrderedDict()
    
    logit = model_output['logit']
    if not args.predict:
        loss =  F.binary_cross_entropy_with_logits(logit, batch['y'].float() )
        l_dict['bce_loss'] = loss.detach()
    p_dict['probs'] = torch.sigmoid(logit).detach()
    p_dict['preds'] = (p_dict['probs'] > 0.5).int()
    p_dict['golds'] = batch['y'].int()

    return loss * args.loss_lambda, l_dict, p_dict

@register_object("regression_loss", 'loss')
def get_regression_loss(model_output, batch, model, args):
    l_dict, p_dict = OrderedDict(), OrderedDict() 
    logit = model_output['logit']
    loss = F.mse_loss(logit, batch['y'].float())
    l_dict['regression_loss'] = loss.detach()
    p_dict['probs'] = logit.detach()
    p_dict['golds'] = batch['y']
    return loss * args.loss_lambda, l_dict, p_dict

@register_object("l1_loss", 'loss')
def l1_regularization(model_output, batch, model, args):
    '''
    Computes L1-Loss

    Returns:
        loss: cross entropy loss
        l_dict (dict): dictionary containing l1 loss detached from computation graph
        p_dict (dict): None
    '''

    output_dict = OrderedDict()
    loss = 0
    for param in model.parameters():
        loss += torch.norm(param, p=1) # torch.sum(torch.abs(param))
    loss *= args.l1_decay
    output_dict['l1_loss'] = loss.mean().item()
    return loss * args.lasso_loss_lambda, output_dict, _
