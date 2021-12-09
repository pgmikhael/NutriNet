import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from modules.utils.shared import register_object
from transformers import AutoModel

@register_object("transformer", 'model')
class TransformerClassifier(nn.Module):
    def __init__(self, args):
        """
        Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py 
        """

        super(TransformerClassifier, self).__init__()

        self.args = args

        self.use_embeddings = args.use_embeddings
        if args.use_embeddings:
            self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
            hidden_dim = args.hidden_dim
        elif args.use_bert_embeddings:
            self.pretrained_model = AutoModel.from_pretrained(args.hf_model_name)
            if hasattr(self.pretrained_model.config, 'dim'):
                hidden_dim = self.pretrained_model.config.dim
            elif hasattr(self.pretrained_model.config, 'hidden_size'):
                hidden_dim = self.pretrained_model.config.hidden_size
        else:
            hidden_dim = args.input_dim
        
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            args.num_heads, 
            args.dim_feedforward, 
            args.dropout,
            batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers)

        self.out_fc = nn.Linear(hidden_dim, args.num_classes)

        self._reset_parameters()

    def forward(self, x, batch):
        """
        src, 
        tgt, 
        src_mask = None, 
        tgt_mask= None,
        memory_mask = None, 
        src_key_padding_mask = None,
        tgt_key_padding_mask = None, 
        memory_key_padding_mask  = None) :
        """
        if self.use_embeddings:
            src = self.embedding(x)
        elif self.args.use_bert_embeddings:
            with torch.no_grad():
                z = self.pretrained_model(x)
                src = z['last_hidden_state']
        else:
            src = x

        src_key_padding_mask = batch['x_mask']

        if self.args.bidirectional:
            memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)    # (batch size, sequence length, word dimension)
        else:
            src_mask = self.generate_square_subsequent_mask(src.shape[0])
            memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)    # (batch size, sequence length, word dimension)
        
        logit = self.out_fc(memory[:,0])    # use [CLS] token to classify
        output = {'hidden': memory[:,0], 'logit': logit}
        
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate(self):
        raise NotImplementedError

@register_object("transformer-generator", 'model')
class Transformer(nn.Module):
    def __init__(self, args):
        """
        Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py 
        """

        super(Transformer, self).__init__()

        self.args = args

        encoder_layer = nn.TransformerEncoderLayer(
            args.hidden_dim,
            args.num_heads, 
            args.dim_feedforward, 
            args.dropout,
            batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, args.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            args.hidden_dim,
            args.num_heads, 
            args.dim_feedforward, 
            args.dropout,
            batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, args.num_decoder_layers)

        self.vocab_fc = nn.Linear(args.hidden_dim, args.vocab_size)

        self._reset_parameters()
        
        self.hidden_dim = args.hidden_dim
        self.nhead = args.num_heads

        self.batch_first = True

    def forward(self, x, batch):
        """
        src, 
        tgt, 
        src_mask = None, 
        tgt_mask= None,
        memory_mask = None, 
        src_key_padding_mask = None,
        tgt_key_padding_mask = None, 
        memory_key_padding_mask  = None) :
        """
        src = x
        src_mask = self.generate_square_subsequent_mask(src.shape[0])
        src_key_padding_mask = batch['x_mask']

        tgt  = batch['y']
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0])
        tgt_key_padding_mask =  batch['y_mask']

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)    # (batch size, input sequence length, word dimension)
        decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=src_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)                     # (batch size, target sequence length, word dimension)
        logit = self.vocab_fc(decoded)
        
        output = {'hidden': memory, 'logit': logit}
        
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) :
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate(self):
        raise NotImplementedError
