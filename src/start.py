import torch

def iFSQ(z, levels):
    '''
    z: visual features map (B*H*W, D)
    levels: list or tensor defining levels per dim (L)
    '''
    # 1. Bound input to [-1, 1]
    z = 2 * torch.nn.Sigmoid(1.6 * z) - 1

    # 2. Scale to the grid defined by levels
    half_width = (levels - 1) / 2
    z_scaled = z * half_width

    # 3. Quantization with Straight-Through Estimator
    z_rounded = torch.round(z_scaled)
    z_hat = z_rounded - z_scaled.detach() + z_scaled

    # 4. Compute indices for AR 
    # basis: [L^(d-1), ..., L^0]    

    z_ind = z_rounded + half_width
    basis = compute_basis(levels)
    indices = torch.sum(z_ind * basis, dim=-1).long()

    return indices 
