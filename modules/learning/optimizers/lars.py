from torch import optim
import torch
from modules.utils.shared import register_object

@register_object("lars", 'optimizer')
class LARS(optim.Optimizer):
    '''
    Layerwise-Adaptive Rate Scaling: https://arxiv.org/pdf/1708.03888.pdf 
    Code adapted from FAIR: https://github.com/facebookresearch/barlowtwins 
    '''
    def __init__(self, params, args):
    #def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
    #             weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
                        eta=args.lars_eta, weight_decay_filter=args.weight_decay_filter,
                        lars_adaptation_filter=args.lars_adaptation_filter)
        param_weights = []
        param_biases = []
        for param in params:
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)
        parameters = [{'params': param_weights}, {'params': param_biases}]
        super().__init__(parameters, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
        return loss
