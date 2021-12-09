# torch
import torch
import torch.nn as nn
#import transformers as ppb
from transformers import AutoTokenizer, AutoModel
# project utils
from modules.utils.shared import register_object, get_object

@register_object('transformer_models', 'model')
class TransformerModels(nn.Module):
    def __init__(self, args):
        '''
        S is the source sequence length
        T is the target sequence length
        N is the batch size
        E is the feature number
        '''
        super(TransformerModels, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, padding_side = 'right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if hasattr(self.model.config, 'dim'):
            args.input_dim = self.model.config.dim
        elif hasattr(self.model.config, 'hidden_size'):
            args.input_dim = self.model.config.hidden_size
        self.register_buffer('pad', torch.zeros(1, args.input_dim))
        self.chunk_aggregator = get_object(args.pretrained_model_aggregator, 'model')(args)
        self.chunk_interval = args.pretrained_input_length - args.pretrained_input_overlap

    def forward(self, x, batch=None):
        # split up text into smaller sentences
        chunks = []
        chunk_size = []
        for widx, word_list in enumerate(batch['sent']):
            word_list = word_list.split('\t')
            x = [word_list[i:(i+self.args.pretrained_input_length)] for i in range(0, len(word_list), self.chunk_interval ) ]
            chunk_size.append(len(x))
            # chunk_idx.extend([widx for _ in range(len(x))])
            chunks.extend([ ' '.join(w) for w in x])
        
        inputs = self.tokenizer(chunks, return_tensors="pt", padding = True)
        if self.model.device.type == 'cuda':
            inputs = {k: v.to('cuda') for k, v in inputs.items()} 

        if self.args.fix_pretrained_encoder:
            with torch.no_grad():
                z = self.model(**inputs)
        else:
            z = self.model(**inputs)
        
        # first token is classification token
        z = z['last_hidden_state'][:,0] 

        # take [cls] token of each chunk in full text 
        z = torch.split(z, chunk_size, dim = 0)
        # run cls (num_chunks, dim) sequence through rnn
        batch['sent_len'] = chunk_size
        z = torch.stack( [ torch.vstack([y, self.pad.repeat(max(chunk_size) - len(y), 1)]) for y in z] )
        return self.chunk_aggregator(z, batch)
    
    @property
    def final_hidden_dim(self):
        return self.chunk_aggregator.final_hidden_dim
