import json
import numpy as np
from scipy.stats import norm

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, decay_type="right"):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    if decay_type == "right":
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    elif decay_type == "normal":
        #center_layer = num_layers // 2
        x = np.linspace(-3, 3, num_layers + 1)  # Adjust spread to match the required range
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # Normalize so the max is 1
        layer_scales = [s for s in normal_dist]  # Convert to list

    elif decay_type == "inverse_normal":
        x = np.linspace(-3, 3, num_layers + 1)  # Adjust spread to match the required range
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # Normalize so the max is 1

        inverted_dist = 1 - normal_dist  # Flip values so the middle becomes the lowest

        midpoint = len(inverted_dist) // 2
        position_counts = np.arange(1, num_layers + 1)  # Generate 1 to num_layers
        scaling_factors = layer_decay ** position_counts[::-1]  # Reverse to start from 0.75^13

        scaled_left = scaling_factors[:midpoint]  # Use appropriate portion of scaling factors
        right = inverted_dist[midpoint-1:]  # Keep the right side unchanged
        right[midpoint-1] += 0.1
        right[midpoint] += 0.1

        layer_scales = np.concatenate([scaled_left, right])

        layer_scales = layer_scales.tolist()

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
            
        layer_id = get_layer_id_for_vit(n, num_layers)
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

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('ppnet'):
        return num_layers
    else:
        return num_layers




import numpy as np
from torch.distributions.normal import Normal  # or from scipy.stats import norm
from math import exp

# If you are using scipy:
from scipy.stats import norm

def get_layer_id_for_vit_pp(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: 
    https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('ppnet'):
        # We'll skip these in the loop, but just in case
        return num_layers
    else:
        return num_layers


def param_groups_lrd_pp(
    model,
    weight_decay=0.05,
    no_weight_decay_list=[],
    layer_decay=0.75,
    decay_type="right",
    last_layer_lr=5e-4,
    prototype_lr=0.05,
):
    """
    Parameter groups for layer-wise lr decay, adapted to skip ppnet layers
    in favor of manual groups for add_on_layers, prototype_vectors, last_layer, etc.
    """
    param_group_names = {}
    param_groups = {}

    # For a ViT, the number of 'blocks' is typically the depth. 
    # We'll assume model.blocks exist. If your model differs, adapt accordingly.
    num_layers = len(model.blocks) + 1

    # Prepare the layer_scales depending on decay_type
    if decay_type == "right":
        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    elif decay_type == "normal":
        x = np.linspace(-3, 3, num_layers + 1)  # Adjust spread to match the required range
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # Normalize so the max is 1
        layer_scales = normal_dist.tolist()
    elif decay_type == "inverse_normal":
        x = np.linspace(-3, 3, num_layers + 1)
        normal_dist = norm.pdf(x) / max(norm.pdf(x))  # normalize
        inverted_dist = 1 - normal_dist  # flip
        midpoint = len(inverted_dist) // 2

        # For example: further scale it if you want
        position_counts = np.arange(1, num_layers + 1)
        scaling_factors = layer_decay ** position_counts[::-1]

        # Then slice & combine as you see fit
        scaled_left = scaling_factors[:midpoint]
        right = inverted_dist[midpoint - 1:]
        # small example modifications
        right[midpoint - 1] += 0.1
        right[midpoint] += 0.1

        layer_scales = np.concatenate([scaled_left, right]).tolist()

    # 1) Go through all named_parameters, but skip anything that starts with "ppnet"
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Skip ppnet parameters for manual grouping later
        if n.startswith("ppnet"):
            continue

        # Decide on weight decay
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit_pp(n, num_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

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

    # Convert to a list of param groups so far
    param_groups_list = list(param_groups.values())

    # 2) Manually add param groups for ppnet layers
    #    (following your snippet)

    # -> Add the add_on_layers group
    addon_params = list(model.ppnet.add_on_layers.parameters())
    param_groups_list.append({
        "params": addon_params,
        "lr": 3e-2,
        "weight_decay": 1e-4,
        # optionally, also "lr_scale": 1.0 if you want to unify format
    })

    # -> Add the prototype_vectors group
    #    (assuming it is a single tensor or list of Tensors)
    proto_params = [model.ppnet.prototype_vectors]  # or list(...)
    param_groups_list.append({
        "params": proto_params,
        "lr": prototype_lr,
        # e.g. no weight decay for prototypes, but you can set it if you prefer
    })

    # -> Add the last_layer group
    last_params = list(model.ppnet.last_layer.parameters())
    param_groups_list.append({
        "params": last_params,
        "lr": last_layer_lr,
        "weight_decay": 1e-4,
    })

    # -> Everything else that might be in ppnet but wasn't in the above
    #    Typically, "rest" is any leftover params not in addon_params, proto_params, or last_params
    all_params = set(model.ppnet.parameters())  # specifically from ppnet
    already_in_groups = set(addon_params + proto_params + last_params)
    rest = [p for p in all_params if p not in already_in_groups]
    if len(rest) > 0:
        param_groups_list.append({"params": rest})
    
    return param_groups_list