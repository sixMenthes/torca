import torch


def round_ste(z):
    zhat = torch.round(z)
    return z + (zhat - z).detach()
    

class FSQ:
    def __init__(self, levels: list[int]):
        self._levels = torch.tensor(levels)
        self._basis = torch.cat([torch.tensor([1]), torch.cumprod(self._levels[:-1], dim=0)])
        codebook_size = torch.prod(self._levels)

    def bound(self, z):


