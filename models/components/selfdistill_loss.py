"""Loss and masking machinery for EMA self-distillation (iBOT-shaped, FSQ targets).

Kept free of lightning/hydra so it can be unit-tested and reasoned about on its own;
`mim_distillation.py` is the thin Lightning glue on top.

Three pieces:

  * `valid_token_mask` — which patch positions correspond to audio a human actually
    heard. 90% of DCLDE clips are shorter than the 3 s window and get zero-padded at
    the tail, so ~44% of positions are synthetic silence on average. Those must not
    enter the loss: the teacher assigns them one constant "silence" code, and a CE
    dominated by predicting it would fall beautifully while learning nothing.
  * `sample_student_mask` — which VALID positions to hide from the student.
  * `masked_ce` + `vicreg` — the objective itself.

The two backbones order tokens identically (time-major, index = t*F_p + f: timm
flattens NCHW row-major over (time, freq), and BEATs' reshape does the same), so all
of this is backbone-agnostic given the grid width.
"""

import torch
import torch.nn.functional as F


def valid_token_mask(n_valid_samples, num_patches, freq_patches, sample_rate,
                     frame_length_ms=25.0, frame_shift_ms=10.0, patch_size=16):
    """(B,) real-sample counts -> (B, num_patches) bool, True where the audio is real.

    Chains the two length transforms the signal goes through: waveform -> kaldi frames
    (a frame is real only if its whole 25 ms window predates the padding) -> patch rows
    (a row of 16 frames counted as real only if all 16 are). Both use floor, so the
    mask is conservative: a straddling row at the signal/padding boundary is dropped
    rather than half-believed.
    """
    win = sample_rate * frame_length_ms / 1000.0
    hop = sample_rate * frame_shift_ms / 1000.0

    n = n_valid_samples.to(torch.float32)
    valid_frames = torch.clamp(torch.floor((n - win) / hop) + 1, min=0.0)
    valid_rows = torch.floor(valid_frames / patch_size)                 # (B,)

    idx = torch.arange(num_patches, device=n_valid_samples.device)
    row_of_token = torch.div(idx, freq_patches, rounding_mode="floor")  # time-major
    return row_of_token.unsqueeze(0) < valid_rows.unsqueeze(1).to(idx.device)


def sample_student_mask(valid, mask_ratio=0.6, generator=None):
    """Choose the positions to hide, sampled only among valid ones.

    Restricting to valid positions is not a detail: a 1 s clip is two-thirds padding,
    so unrestricted sampling would routinely hide most of the real audio while leaving
    "context" that is entirely silence — an unanswerable prediction problem.

    Guarantees >=1 masked position per sample (a row with no masked position
    contributes no CE gradient and would silently shrink the effective batch).
    """
    B, N = valid.shape
    n_valid = valid.sum(dim=1)                                   # (B,)
    n_mask = torch.clamp((n_valid.float() * mask_ratio).round().long(), min=1)
    n_mask = torch.minimum(n_mask, n_valid.clamp(min=1))

    # random scores, invalid positions pushed to +inf so they sort last and are
    # never selected by the per-row top-k below
    scores = torch.rand(B, N, device=valid.device, generator=generator)
    scores = scores.masked_fill(~valid, float("inf"))

    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    return (ranks < n_mask.unsqueeze(1)) & valid


def codebook_logits(z, fsq, temperature=1.0):
    """Continuous pre-quant projections -> logits over the FSQ codebook.

    FSQ has no learned prototypes, so there is nothing to read logits off directly.
    Instead we score each code by negative squared distance in FSQ's own normalised
    space — the iBOT prototype-logit construction with a fixed scalar grid standing in
    for learned prototypes, which is precisely why this design resists collapse.

    `z` is PRE-bound (unbounded). `quantize` is round_ste(bound(z))/half_width, so the
    differentiable analogue that lives in the same space as the codebook is
    bound(z)/half_width — with the rounding, and only the rounding, dropped.
    """
    half_width = (fsq._levels // 2).to(z.dtype)
    z_norm = fsq.bound(z) / half_width                        # (..., L)
    codebook = fsq.codebook.to(z.dtype)                       # (codebook_size, L)

    d2 = torch.cdist(z_norm.flatten(0, -2), codebook).pow(2)  # (M, codebook_size)
    return (-d2 / temperature).view(*z_norm.shape[:-1], -1)


def masked_ce(logits, target_indices, loss_mask):
    """Cross-entropy at masked+valid positions only.

    Returns a zero (still graph-connected) if nothing is selected, so a degenerate
    batch can't produce NaN and kill the run.
    """
    if loss_mask.sum() == 0:
        return logits.sum() * 0.0
    sel = loss_mask.reshape(-1)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[sel], target_indices.reshape(-1)[sel]
    )


def _masked_select_tokens(z, valid):
    """(B, N, L) + (B, N) bool -> (M, L) of valid tokens only."""
    return z[valid]


def vicreg(z, valid, gamma=1.0, var_weight=1.0, cov_weight=0.04,
           within_clip_var_weight=0.0, eps=1e-4):
    """Anti-collapse guard on the student's pre-FSQ projections.

    Both statistics are computed over the flattened population of VALID patch tokens,
    because FSQ quantises each patch independently — the population whose collapse we
    fear is the token population, not the clip population. Pooling per clip first
    would hide the failure that matters: every patch inside a clip collapsing to one
    code still leaves healthy clip-level spread.

    variance: hinge on per-dimension std, pushing each of the L axes to stay spread.
    covariance: off-diagonal penalty on the LxL covariance, so the axes carry
      independent information and the product grid prod(levels) is actually used
      rather than a diagonal sliver of it. Computed globally, not per-clip-then-
      averaged: by the law of total covariance the latter constrains only the
      within-clip term and leaves between-clip correlation — which wastes codebook
      cells just as effectively — unpenalised.
    within_clip_var (optional, off by default): the complementary half of the
      variance decomposition. The global term can be satisfied by between-clip spread
      alone, so this is what actually forbids "every token in this clip is identical".

    Note this is VICReg's regularisers only — invariance is supplied by the masked CE
    against EMA targets, so the whole objective is iBOT-with-a-VICReg-guard rather
    than VICReg proper.
    """
    out = {}
    tokens = _masked_select_tokens(z, valid)                  # (M, L)

    if tokens.shape[0] < 2:                                   # degenerate batch
        zero = z.sum() * 0.0
        return zero, {"vic_var": zero.detach(), "vic_cov": zero.detach()}

    std = torch.sqrt(tokens.var(dim=0) + eps)                 # (L,)
    var_loss = F.relu(gamma - std).mean()
    out["vic_var"] = var_loss.detach()

    centered = tokens - tokens.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (tokens.shape[0] - 1)      # (L, L)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / tokens.shape[1]
    out["vic_cov"] = cov_loss.detach()

    total = var_weight * var_loss + cov_weight * cov_loss

    if within_clip_var_weight > 0:
        # per-clip std over that clip's valid tokens, then hinge; clips with <2 valid
        # tokens contribute nothing rather than a spurious zero-variance penalty.
        counts = valid.sum(dim=1)                              # (B,)
        usable = counts >= 2
        if usable.any():
            zc = z.masked_fill(~valid.unsqueeze(-1), 0.0)
            n = counts.clamp(min=1).unsqueeze(-1).to(z.dtype)
            mean = zc.sum(dim=1) / n                                       # (B, L)
            sq = ((zc - mean.unsqueeze(1)) * valid.unsqueeze(-1)).pow(2).sum(dim=1)
            within_std = torch.sqrt(sq / (n - 1).clamp(min=1) + eps)       # (B, L)
            within = F.relu(gamma - within_std[usable]).mean()
            total = total + within_clip_var_weight * within
            out["vic_within"] = within.detach()

    return total, out
