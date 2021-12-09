import torch

RESGISTRIES = {
    'LIGHTNING_REGISTRY': {},
    'BASE_MODEL_REGISTRY': {},
    'DATASET_REGISTRY': {},
    'MODEL_REGISTRY': {},
    'LOSS_REGISTRY': {},
    'METRIC_REGISTRY': {},
    'OPTIMIZER_REGISTRY': {},
    'SCHEDULER_REGISTRY': {},
    'CALLBACK_REGISTRY': {}
}

def get_object(object_name, object_type):
    if object_name not in RESGISTRIES['{}_REGISTRY'.format(object_type.upper())]:
        raise Exception('INVALID {} NAME: {}. AVAILABLE {}'.format(object_type.upper(), object_name, RESGISTRIES['{}_REGISTRY'.format(object_type.upper())].keys()))
    return RESGISTRIES['{}_REGISTRY'.format(object_type.upper())][object_name]

def register_object(object_name, object_type):
    def decorator(obj):
        RESGISTRIES['{}_REGISTRY'.format(object_type.upper())][object_name] = obj
        return obj 
    return decorator

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

