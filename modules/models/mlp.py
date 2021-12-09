import torch
import torch.nn as nn
import copy
from modules.utils.shared import register_object, get_object
from transformers import AutoModel

@register_object('fc_classifier', 'model')
class FC(nn.Module):
    def __init__(self, args):
        super(FC, self).__init__()

        self.args = args
        model_layers = []
        cur_dim = args.fc_classifier_input_dim
        model_layers.append(nn.Linear(args.fc_classifier_input_dim, args.num_classes))
        
        self.predictor = nn.Sequential(*model_layers)
    
    def forward(self, x, batch=None):
        output = {}
        output['logit'] = self.predictor(x.float())
        return output

@register_object('mlp', 'model')
class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args

        model_layers = []
        cur_dim = args.input_dim

        for layer_size in args.mlp_layer_configuration[:-1]:
            model_layers.extend( self.append_layer(cur_dim, layer_size, args) )
            cur_dim = layer_size
        
        layer_size = args.mlp_layer_configuration[-1]
        model_layers.extend( self.append_layer(cur_dim, layer_size, args, with_dropout = False) )

        self.predictor = nn.Sequential(*model_layers)
    
    def append_layer(self, cur_dim, layer_size, args, with_dropout = True):
        linear_layer = nn.Linear(cur_dim, layer_size)
        bn = nn.BatchNorm1d(layer_size)
        seq = [linear_layer, bn, nn.ReLU()]
        if with_dropout:
            seq.append(nn.Dropout(p=args.dropout))
        return seq
    
    def forward(self, x, batch=None):
        output = {}
        z = self.predictor(x.float())
        output['logit'] = z
        return output

@register_object('mlp_classifier', 'model')
class MLPClassifier(nn.Module):
    def __init__(self, args):
        super(MLPClassifier, self).__init__()

        self.args = args
        
        if self.args.use_bert_embeddings:
            self.pretrained_model = AutoModel.from_pretrained(args.hf_model_name)
            if hasattr(self.pretrained_model.config, 'dim'):
                args.input_dim = self.pretrained_model.config.dim
            elif hasattr(self.pretrained_model.config, 'hidden_size'):
                args.input_dim = self.pretrained_model.config.hidden_size
        elif self.args.use_embeddings:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_size)
            args.input_dim = args.embed_size

        self.mlp = MLP(args)
        self.dropout = nn.Dropout(p=args.dropout)
        args.fc_classifier_input_dim = args.mlp_layer_configuration[-1] 
        self.predictor = FC(args)
    
    def forward(self, x, batch=None):
        output = {}
        if self.args.use_bert_embeddings:
            with torch.no_grad():
                z = self.pretrained_model(x)
                e = z['last_hidden_state'][:,0]
        elif self.args.use_embeddings:
            e = self.embedding(x).mean(1)
        else:
            e = x.mean(1)

        z = self.dropout( self.mlp(e)['logit'] )
        output['logit'] = self.predictor(z)['logit']
        output['hidden'] = z
        return output

