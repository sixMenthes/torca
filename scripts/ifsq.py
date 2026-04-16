import torch
#import torch.nn as nn


def round_ste(z):
    zhat = torch.round(z)
    return z + (zhat - z).detach()

def compute_basis(levels):
    return torch.cat([torch.tensor([1]), torch.tensor(levels).cumprod(dim=-1)])
    

class FSQ:
    def __init__(self, levels: list[int], eps: float = 1e-3):
        self._levels = torch.tensor(levels)
        self._eps = eps
        self._basis = torch.cat([torch.tensor([1]), torch.cumprod(self._levels[:-1], dim=0)])
        self._implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size))

    @property
    def num_dimensions(self) -> int:
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        return torch.prod(self._levels)

    @property 
    def codebook(self):
        return self._implicit_codebook

    def bound(self, z: torch.Tensor):
        half_l = (self._levels - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0, 0.5)
        shift = torch.tan(offset/half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor):
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized): 
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).to(torch.int32)

    def indices_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self._basis), self._levels)
        return self._scale_and_shift_inverse(codes_non_centered)






