import os, sys
from modules.datasets.loader.factory import RegisterInputLoader
import numpy as np
import os 
import torch
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
import torch.nn
from transformers import AutoTokenizer, AutoModel

LOADING_ERROR = 'LOADING ERROR! {}'

@RegisterInputLoader('GloVe')
class GloVeTokenizer(object):
    def __init__(self, args):
        self.args = args
        self.glove_vectors = GloVe(args.glove_name) # '6B'
        # set freeze to false if you want them to be trainable
        args.vocab_size = len(self.glove_vectors) 
        args.input_dim = self.glove_vectors.dim
        self.tokenizer = get_tokenizer("spacy", language='en_core_web_sm')

    def load_input(self, sample):
        '''
        Return N x d sequence matrix
        '''
        # special_tokens = ['<unk>', '<pad>']
        ingredients = self.tokenizer(sample['ingredients'])
        instructions = self.tokenizer(sample['instructions'])
        text = ['cls'] + ingredients + ['.'] + instructions 
        text_len = min(len(text), self.args.max_sequence_length)
        
        mask = np.zeros(self.args.max_sequence_length) 
        mask[:text_len] = 1

        if self.args.max_sequence_length > len(text):
            text = text + ['pad'] * (self.args.max_sequence_length - len(text))
        elif self.args.max_sequence_length < len(text):
            text = text[:self.args.max_sequence_length]
        
        # if this doesn't work then might have to create a vocab
        return {'input': self.glove_vectors.get_vecs_by_tokens(text), 'x_mask': mask, 'x_length': text_len}

@RegisterInputLoader('AutoTokenizer')
class AutoTokenizerObj(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_name, padding_side = 'right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        args.vocab_size = self.tokenizer.vocab_size

    def load_input(self, sample):
        """
        Loads input as indices in a pretrained vocabulary

        Parameters
        ----------
        key : str
            key to use in sample for reading input text
        sample : dict
            sample datapoint

        Returns
        -------
        dict
            inputs contains a vector of indices as tensors
        """
        text = sample['ingredients'] + ' [SEP] ' + sample['instructions']
        inputs = self.tokenizer(text, return_tensors="pt", max_length= self.args.max_sequence_length, padding='max_length', truncation=True)

        return {'input': inputs['input_ids'][0], 'x_mask': inputs['attention_mask'][0], 'x_length': int(inputs['attention_mask'][0].sum()) }

