
import torch
import torch.nn as nn

def round_ste(z):
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    def __init__(self, levels, eps=1e-3):
        super().__init__()
        levels_t = torch.tensor(levels)
        basis = torch.cat([torch.tensor([1]), torch.cumprod(levels_t[:-1], dim=0)])
        self.register_buffer("_levels", levels_t)
        self.register_buffer("_basis", basis)
        self._eps = eps
        # _implicit_codebook depends on the buffers above; compute and register it too
        codebook = self._compute_codebook()
        self.register_buffer("_implicit_codebook", codebook)

    def _compute_codebook(self):
        return self.indices_to_codes(torch.arange(self.codebook_size))

    @property
    def num_dimensions(self):
        return len(self._levels)

    @property
    def codebook_size(self):
        return int(torch.prod(self._levels).item())

    @property
    def codebook(self):
        return self._implicit_codebook

    def bound(self, z):
        half_l = (self._levels - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
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
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self._basis),
self._levels)
        return self._scale_and_shift_inverse(codes_non_centered)