# torch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# project utils
from modules.utils.shared import register_object, get_object
from transformers import AutoModel

@register_object('gru', 'model')
class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()

        self.args = args
        gru_dropout = args.dropout
        if args.num_encoder_layers==1:
            gru_dropout = 0
        
        self.num_directions =  2 if args.bidirectional else 1
        
        self.use_embeddings = args.use_embeddings
        if args.use_embeddings:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_size)
            input_dim = args.embed_size
        elif args.use_bert_embeddings:
            self.pretrained_model = AutoModel.from_pretrained(args.hf_model_name)
            if hasattr(self.pretrained_model.config, 'dim'):
                input_dim = self.pretrained_model.config.dim
            elif hasattr(self.pretrained_model.config, 'hidden_size'):
                input_dim = self.pretrained_model.config.hidden_size
        else:
            input_dim = args.input_dim
            
        self.gru = nn.GRU(
            input_size = input_dim,
            hidden_size = args.hidden_dim,
            num_layers = args.num_encoder_layers,
            bias = True,
            batch_first = True,
            dropout = gru_dropout,
            bidirectional = args.bidirectional)
        
        self.out_fc = nn.Linear(self.final_hidden_dim , args.num_classes)

        self.register_buffer('h0', torch.zeros(1))

    @property
    def final_hidden_dim(self):
        return self.args.hidden_dim * self.num_directions 

    def initHidden(self, B):
        num_layers = self.args.num_encoder_layers * self.num_directions
        h0 = self.h0.repeat(num_layers, B, self.args.hidden_dim)
        return h0

    def forward(self, x, batch=None):
        '''
        x: batch, seq_len, input_size
        h0: num_layers * num_directions, batch, hidden_size  
        output: batch, seq_len, num_directions * hidden_size
        h_n: num_layers * num_directions, batch, hidden_size  
        '''
        B = x.shape[0]
        h0 = self.initHidden(B)
        if self.use_embeddings:
            e = self.embedding(x)
        elif self.args.use_bert_embeddings:
            with torch.no_grad():
                z = self.pretrained_model(x)
                e = z['last_hidden_state']
        else:
            e = x
        e = pack_padded_sequence(e, batch['x_length'].cpu(), enforce_sorted=False, batch_first = True)
        self.gru.flatten_parameters()
        seq_o, h_n  = self.gru(e, h0)
        seq_o, str_lens = pad_packed_sequence(seq_o, padding_value = 0, total_length=max(batch['x_length']), batch_first = True)
        h_n = h_n.permute(1,0,2).contiguous().view(B, -1)
        hidden = self.pool(seq_o, h_n)
        logit = self.out_fc(hidden)

        return {'logit': logit, 'hidden': hidden}

    def pool(self, seq_o, h_n):
        return seq_o.mean(dim=1)
