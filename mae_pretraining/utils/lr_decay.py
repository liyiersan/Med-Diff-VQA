# copied from https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py

import json

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = []
    num_stages = len(model.layers)
    for i in range(num_stages):
        num_layers.append(len(model.layers[i].blocks))
    total_layers = sum(num_layers) + 1

    layer_scales = list(layer_decay ** (total_layers - i) for i in range(total_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_mixmim(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_mixmim(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed', 'absolute_pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('layers'):
        block_idx = int(name.split('.')[1])
        if 'downsample' in name:
            return sum(num_layers[:block_idx + 1]) + 1
        layer_idx = int(name.split('.')[3])
        return sum(num_layers[:block_idx]) + layer_idx + 1
    else:
        return sum(num_layers)