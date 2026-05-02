import copy
import torch
import torch.nn as nn
import math
import itertools

def clones(module, N):
    """
    From python's copy module: "A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original."
    nn.ModuleList holds submodules in a list
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_mask(seq_len, grid_freq=8, obj_masked=0.15, span=3):
    mask = torch.zeros(seq_len, dtype=torch.bool)
    prop = 0
    while prop < obj_masked:
        start = torch.randint(0, seq_len-span, (1,)).item()
        direction = (torch.rand(1) < 0.5)
        if direction: # mask horizontally 
            stop = start + (grid_freq * (span + 1))
            idxs = [i for i in range(start, stop, 8) if i < seq_len]
        else: # mask vertically
            idxs = [i for i in range(start, start+span+1) if i < seq_len]
        mask[list(idxs)] = True
        prop = mask.count_nonzero() / seq_len

    return mask


def get_alibi(attention_heads, num_patches_freq, num_patches_time):
    """
    Function adapted to rectangular-shaped input from original CROMA paper by Fuller et al. (2023)
    """
    points = list(itertools.product(range(num_patches_freq), range(num_patches_time)))
    num_patches = num_patches_freq * num_patches_time

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(attention_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, attention_heads, num_patches, num_patches)
