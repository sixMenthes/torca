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

import math

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
    z_norm = normalised_z(z, fsq)                             # (..., L)
    # cast to z_norm's dtype, NOT z's: fsq.bound() promotes to float32 (levels is an
    # integer buffer, so half_l is float32), so under bf16 the two cdist arguments
    # disagree and it raises. Autocast reconciles that during training, which is why
    # it only surfaces when this is called outside autocast — extracting features
    # from a checkpoint, say.
    codebook = fsq.codebook.to(z_norm.dtype)                  # (codebook_size, L)

    d2 = torch.cdist(z_norm.flatten(0, -2), codebook).pow(2)  # (M, codebook_size)
    return (-d2 / temperature).view(*z_norm.shape[:-1], -1)


def normalised_z(z, fsq):
    """Pre-bound projections -> the codebook's own space, with only the rounding gone.

    `quantize` is round_ste(bound(z))/half_width; this is that without the round. All
    FSQ codes live in [-1, 1] here, and bound()'s tanh means |z_norm| approaches but
    never reaches the outermost code.
    """
    half_width = (fsq._levels // 2).to(torch.float32)
    return fsq.bound(z) / half_width


def saturation_frac(z, fsq, valid, thresh=0.99):
    """Fraction of valid coordinates parked at the edge of the bounded range.

    The diagnostic for the collapse VICReg cannot see. Both VICReg terms are computed
    on the PRE-bound z, but the route from z to a code runs through tanh: drive ||z||
    up and every large coordinate saturates onto the same extreme. In that regime z
    has enormous per-dimension std (variance hinge satisfied, vic_var = 0) and
    independent dimensions (vic_cov ~ 0), while every token quantises to a cube
    CORNER — 2**L codes out of prod(levels).

    Both guards read healthy while the tokenizer is dead, so this is the number that
    tells them apart. Near 1.0 means saturated.
    """
    with torch.no_grad():
        sel = normalised_z(z, fsq)[valid]
        if sel.numel() == 0:
            return torch.zeros((), device=z.device)
        return (sel.abs() >= thresh).to(torch.float32).mean()


def codebook_diversity(logits, valid, softmax_scale=1.0, sample_entropy_weight=0.0,
                       eps=1e-8):
    """Entropy penalty on the soft code assignment: (w*E[H(q)] - H(E[q])) / log K.

    The term the objective is missing. VICReg variance measures SPREAD, and spread
    does not imply coverage — a distribution in two tight clumps at +/-1.2 has std 1.2
    and occupies two cells of the grid. Coverage is an entropy question and needs an
    entropy penalty.

    Form follows LFQ/MAGVIT-v2 (Yu et al. 2024), whose lookup-free quantizer is the
    same family as FSQ — fixed grid, no learned codebook. Two halves, each doing a
    distinct job:
      * -H(E[q])  : entropy of the BATCH-MEAN assignment, maximised. This is the
        anti-collapse half. Taken of the mean rather than as a mean of per-token
        entropies, because the latter is minimised by every token being confident,
        which says nothing about whether they are confident about the SAME code.
      * +E[H(q)]  : mean per-token entropy, minimised, so tokens still commit to one
        code rather than smearing across many to cheat the first term.
    wav2vec 2.0 (Baevski et al. 2020) is the ancestor, using only the first half,
    expressed as codebook perplexity.

    Unlike FSQNet's old diversity term — negative entropy of a bincount over integer
    codes, which has no gradient and therefore regularised nothing — this is computed
    on q = softmax(logits) and does.

    `softmax_scale` rescales the logits before the softmax, so coverage can be scored
    at a different temperature from the CE. MEASURED: leave it at 1.0, i.e. score at
    the CE's own sharp temperature. Softening looks like it should help (a near-one-hot
    q seems to have no gradient) and does the opposite on both counts — separation
    between healthy and collapsed regimes across 4 synthetic cases, and gradient
    magnitude:

        t_div   healthy   x16     x4      x1     |grad|
        0.05     -0.985  -0.766  -0.595  -0.381  1.4e-04   monotonic
        1.00     -0.993  -0.982  -0.984  -0.925  1.2e-05   NOT monotonic

    At the sharp temperature q_bar is close to the true code histogram, which is
    exactly the quantity we want; softening smears it toward uniform for every input
    and the signal disappears into the temperature.

    `sample_entropy_weight` likewise defaults OFF, departing from LFQ. At a sharp
    temperature the per-token entropy is already near zero, so the term adds nothing
    and dilutes the coverage signal — at w=1.0 the ordering inverts and a collapsed
    batch scores BETTER than a healthy one, because a collapsed z sits far from most
    codes and so has a sharper per-token softmax. Kept as a knob for softer regimes.

    Normalised by log K so the value is comparable across different `levels`, for the
    same reason token_bits_frac is.
    """
    sel = logits[valid] if valid is not None else logits.flatten(0, -2)
    if sel.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, {"diversity": zero.detach(), "soft_bits": zero.detach()}

    q = (sel.float() * softmax_scale).softmax(-1)             # (M, K)
    log_k = math.log(sel.shape[-1])

    batch_ent = -(q.mean(0) * torch.log(q.mean(0) + eps)).sum()        # H(E[q])
    sample_ent = -(q * torch.log(q + eps)).sum(-1).mean()              # E[H(q)]

    loss = (sample_entropy_weight * sample_ent - batch_ent) / log_k
    return loss, {
        "diversity": loss.detach(),
        # bits the code distribution actually carries at this temperature; compare
        # against log2(K). The hard-assignment counterpart is train/token_bits.
        "soft_bits": (batch_ent / math.log(2)).detach(),
        "soft_sample_bits": (sample_ent / math.log(2)).detach(),
    }


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


def code_counts(indices, valid, codebook_size):
    """Histogram of FSQ code usage over valid positions. (K,) long tensor.

    Returned rather than reduced so the caller can accumulate across a whole epoch:
    the entropy of pooled counts is NOT the mean of per-batch entropies (entropy is
    concave, so averaging per-batch values understates the pooled figure).
    """
    sel = indices[valid] if valid is not None else indices
    return torch.bincount(sel.reshape(-1), minlength=codebook_size)


def entropy_bits(counts):
    """Empirical entropy of a code-usage histogram, in bits: H = -sum p log2 p.

    This is what `codebook_frac` cannot see. Support size counts how many codes appear
    at all, so it scores "1000 codes, one of them used 99% of the time" as healthy.
    Entropy is the quantity that actually bounds what a token can carry, and it reads
    directly against the log2(K) ceiling.

    Concretely: a K=1000 run collapsed onto 50 codes carries 5.6 bits — LESS than the
    K=240 config it replaced (7.9) — while looking larger in the config file.

    Note this is the plug-in estimator, biased DOWNWARD when the sample is small
    relative to K (roughly (K-1)/(2 N ln2) bits). Accumulate over an epoch for the
    headline number; per-batch values are a live signal, not an estimate to quote.
    """
    total = counts.sum()
    if total == 0:
        return counts.new_zeros((), dtype=torch.float32)
    p = counts.float() / total.float()
    p = p[p > 0]
    return -(p * p.log2()).sum()


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
