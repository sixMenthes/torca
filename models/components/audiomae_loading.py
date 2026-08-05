"""Shared AudioMAE / Bird-MAE checkpoint loading, factored out of vit.py.

`VIT.load_pretrained_weights` and `VIT_ppnet.load_pretrained_weights` each hard-code
a `target_length == 512` branch and a `target_length == 1024` branch; at any other
input length neither fires and the model silently keeps its random init. This module
holds the length-agnostic version of that logic so the two classes (and the
standalone `BirdMAEEncoder`) share one implementation.

Two facts drive the design:

1. **Only `pos_embed` is length-dependent.** `patch_embed.proj` is a 16x16 stride-16
   conv — its weights don't know how long the spectrogram is — and the transformer
   blocks are set-to-set. So a 5 s (512-frame) checkpoint transfers to a 3 s
   (304-frame) input as-is, *except* for the position table, which is 257 rows at
   512 and must be 153 at 304.

2. **`strict=False` does NOT forgive shape mismatches.** It ignores missing and
   unexpected keys only; a `pos_embed` of the wrong length raises RuntimeError from
   `load_state_dict`. So shape-mismatched keys have to be dropped *before* loading,
   which is exactly why the 512 branch gets away without it (shapes match there) and
   why copying that branch to a new length would break.

Dropping `pos_embed` loses nothing: torca (like AudioMAE) uses a *fixed* 2-D sincos
table, not a learned one — the 512 branch overwrites the loaded values with freshly
generated sincos immediately after loading. We regenerate at the new grid instead.
"""

import torch
from util.pos_embed import get_2d_sincos_pos_embed_flexible


def read_checkpoint(path):
    """Return the raw parameter dict from an AudioMAE / Bird-MAE / Lightning ckpt.

    weights_only=False: published Bird-MAE checkpoints pickle omegaconf configs,
    which PyTorch>=2.6 rejects under the default weights_only=True. Trusted source.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"unexpected checkpoint object {type(ckpt)} in {path}")
    for key in ("model", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt  # already a bare state dict


def remap_audiomae_keys(state_dict):
    """Strip the wrapper prefixes so keys line up with a plain VisionTransformer.

    Three checkpoint shapes are in play:
      * plain AudioMAE/Bird-MAE  -> `encoder.` prefix (or none), `decoder.` to drop;
      * MIM-Refiner             -> take `encoder_ema.` (the EMA teacher IS the model
        we want) and drop `encoder.`/`projectors.`/`predictors.`;
      * already-flat            -> passthrough.
    Mirrors the branching in vit.py so behaviour is unchanged, just shared.
    """
    mim_refiner = "encoder_ema.cls_token" in state_dict
    out = {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            continue
        if mim_refiner:
            if key.startswith(("encoder.", "projectors.", "predictors.")):
                continue
            new_key = key[len("encoder_ema."):] if key.startswith("encoder_ema.") else key
        else:
            new_key = key[len("encoder."):] if key.startswith("encoder.") else key
        out[new_key] = value
    return out


def audiomae_pos_embed(embed_dim, target_length, num_mel_bins=128, patch_size=16,
                       cls_token=True):
    """Fixed 2-D sincos position table for a (target_length, num_mel_bins) input.

    The grid is passed as (freq_patches, time_patches) — img_size[1]//p, img_size[0]//p —
    which is the order vit.py's 512 branch uses and, more importantly, the order
    AudioMAE itself used at pretraining time. Note this is *transposed* relative to
    the order `PatchEmbed` flattens tokens in (time-major, t*F_p + f). We reproduce
    it deliberately: the table is a fixed code, so what matters is that inference
    matches pretraining, not that the code reads left-to-right. Changing it would
    silently invalidate the pretrained weights.

    Returns (1, N+1, embed_dim) with the cls slot when `cls_token`, else (1, N, D).
    """
    patch_hw = (num_mel_bins // patch_size, target_length // patch_size)
    pos = get_2d_sincos_pos_embed_flexible(embed_dim, patch_hw, cls_token=cls_token)
    return torch.from_numpy(pos).float().unsqueeze(0)


def load_audiomae_weights(module, path, drop_head=True, verbose=True):
    """Load an AudioMAE/Bird-MAE checkpoint into `module`, length-agnostically.

    Drops the classifier head (`drop_head`) and every key whose shape disagrees with
    the target module — in practice `pos_embed`, whenever the input length differs
    from the checkpoint's. Returns (load_info, dropped) so callers can assert on what
    actually transferred rather than trusting a silent strict=False.
    """
    pretrained = remap_audiomae_keys(read_checkpoint(path))

    if drop_head:
        for k in ("head.weight", "head.bias"):
            pretrained.pop(k, None)

    current = module.state_dict()
    dropped = []
    for k, v in list(pretrained.items()):
        if k in current and hasattr(v, "shape") and current[k].shape != v.shape:
            dropped.append((k, tuple(v.shape), tuple(current[k].shape)))
            del pretrained[k]

    info = module.load_state_dict(pretrained, strict=False)

    if verbose:
        for k, ckpt_shape, model_shape in dropped:
            print(f"[audiomae] dropped {k}: checkpoint {ckpt_shape} != model {model_shape}")
        # Missing keys are the real signal that something didn't transfer. cls_token /
        # patch_embed / blocks.* showing up here means the checkpoint didn't match.
        print(f"[audiomae] loaded {path}: "
              f"{len(pretrained)} tensors, {len(info.missing_keys)} missing, "
              f"{len(info.unexpected_keys)} unexpected")
        if info.missing_keys:
            print(f"[audiomae]   missing: {info.missing_keys[:8]}"
                  f"{' ...' if len(info.missing_keys) > 8 else ''}")
    return info, dropped
