"""Loss and masking machinery for EMA self-distillation (iBOT-shaped, FSQ targets).

Kept free of lightning/hydra so it can be unit-tested and reasoned about on its own;
`mim_distillation.py` is the thin Lightning glue on top.

Three pieces:

  * `valid_token_mask` — which patch positions correspond to audio a human actually
    heard. Clips are fixed-length windows centred on the annotation and clamped at the
    file start, so tail padding appears only when the source file ends first: measured
    train/valid_frac is 0.945, i.e. ~5% of positions are synthetic silence, NOT the
    ~44% an earlier version of this comment claimed. Those positions must still be kept
    out of the loss — the teacher assigns them one constant "silence" code, and a CE
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


def sample_student_mask(valid, mask_ratio=0.6, generator=None, strategy="random",
                        freq_patches=None, max_time_band_frac=0.25,
                        max_freq_band_frac=0.375, p_freq=0.5):
    """Choose the positions to hide, sampled only among valid ones.

    Restricting to valid positions is not a detail: a 1 s clip is two-thirds padding,
    so unrestricted sampling would routinely hide most of the real audio while leaving
    "context" that is entirely silence — an unanswerable prediction problem.

    Guarantees >=1 masked position per sample (a row with no masked position
    contributes no CE gradient and would silently shrink the effective batch) and
    >=1 UNmasked valid position (a sample with no visible context is unanswerable).

    strategy="random" scatters individual patches over the whole time-frequency grid,
    which is the MAE and iBOT convention and what every run before 2026-08-09 used. Note
    that it has always covered BOTH axes: the grid is time-major with frequency varying
    fastest, so index = time_row * freq_patches + freq_col, and a masked unit is one
    16-by-16 tile rather than a whole time frame.

    strategy="bands" is SpecAugment-shaped instead. It draws whole contiguous stripes,
    each one randomly a frequency band (every time step, a run of mel bins) or a time
    band (every mel bin, a run of frames), and keeps drawing until `mask_ratio` of the
    valid positions is covered. The difference from scattered tiles is what the model
    can do about it: a scattered tile can often be filled in by interpolating its
    immediate neighbours, whereas a band removes an entire region and forces inference
    from surrounding context. Bird-MAE's own pretraining used exactly this pair of
    operations at the spectrogram level, with frequency masking over up to 50 of 128
    mel bins and time masking over up to 100 of ~998 frames.

    Bands are drawn at PATCH granularity, not frame granularity, which is coarser than
    SpecAugment but is the right unit here: the mask is applied to patch embeddings, so
    a stripe narrower than one patch could not be represented anyway.

    `mask_ratio` is a target rather than an exact rate under "bands". Stripes are coarse
    (with 8 frequency columns a 3-wide band is already 37.5% of the grid) so coverage
    overshoots the target by however much the last band added. Watch train/mask_frac for
    what was actually achieved.

    freq_patches: number of patch columns along the frequency axis, needed only by
    "bands" to know the grid shape. Passing None falls back to "random".
    """
    B, N = valid.shape
    n_valid = valid.sum(dim=1)                                   # (B,)

    if strategy == "bands" and freq_patches:
        return _band_mask(valid, mask_ratio, freq_patches, generator,
                          max_time_band_frac, max_freq_band_frac, p_freq)

    n_mask = torch.clamp((n_valid.float() * mask_ratio).round().long(), min=1)
    n_mask = torch.minimum(n_mask, n_valid.clamp(min=1))

    # random scores, invalid positions pushed to +inf so they sort last and are
    # never selected by the per-row top-k below
    scores = torch.rand(B, N, device=valid.device, generator=generator)
    scores = scores.masked_fill(~valid, float("inf"))

    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    return (ranks < n_mask.unsqueeze(1)) & valid


def _band_mask(valid, mask_ratio, freq_patches, generator,
               max_time_band_frac, max_freq_band_frac, p_freq):
    """SpecAugment-shaped stripe masking over the patch grid. See sample_student_mask.

    A Python loop over the batch rather than a vectorised draw, because each sample
    needs a different NUMBER of bands to reach its coverage target (samples differ in
    how much of them is padding). At batch 32 against a ViT-B forward pass the cost is
    not measurable.
    """
    B, N = valid.shape
    dev = valid.device
    n_time = N // freq_patches
    n_freq = freq_patches
    max_t = max(1, int(round(n_time * max_time_band_frac)))
    max_f = max(1, int(round(n_freq * max_freq_band_frac)))

    def _draw(hi):
        return int(torch.randint(1, hi + 1, (1,), device=dev, generator=generator))

    out = torch.zeros_like(valid)
    for b in range(B):
        v = valid[b].view(n_time, n_freq)
        n_v = int(v.sum())
        if n_v == 0:
            continue
        # Cap below n_v so at least one valid position stays visible as context.
        target = min(max(1, int(round(n_v * mask_ratio))), n_v - 1) if n_v > 1 else 1
        m = torch.zeros(n_time, n_freq, dtype=torch.bool, device=dev)
        # Bounded rather than while-True: with heavy padding a band can repeatedly land
        # entirely on invalid rows and never advance the count.
        for _ in range(64):
            if int((m & v).sum()) >= target:
                break
            if float(torch.rand(1, device=dev, generator=generator)) < p_freq:
                w = _draw(max_f)
                s = int(torch.randint(0, max(1, n_freq - w + 1), (1,), device=dev,
                                      generator=generator))
                m[:, s:s + w] = True
            else:
                w = _draw(max_t)
                s = int(torch.randint(0, max(1, n_time - w + 1), (1,), device=dev,
                                      generator=generator))
                m[s:s + w, :] = True
        m &= v
        if not bool(m.any()):                     # every band missed the valid region
            idx = torch.nonzero(v.view(-1), as_tuple=False)[0]
            m.view(-1)[idx] = True
        elif n_v > 1 and int(m.sum()) == n_v:     # no context left, free one position
            idx = torch.nonzero(m.view(-1), as_tuple=False)[0]
            m.view(-1)[idx] = False
        out[b] = m.view(-1)
    return out


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


def quantisation_gap(z, fsq, logits, target_indices, mask, temperature):
    """How far the student lands from the teacher's code, measured WITHOUT the temperature.

    `masked_ce` and `train/masked_acc` are both poor instruments for this. Accuracy is
    a yes/no question that discards the size of an error, and cross-entropy is
    -ln q(true) where q is a softmax over -d^2/temperature, so at temperature=0.05 the
    softmax is effectively one-hot and the loss value becomes (1 - accuracy) times a
    near-constant miss cost. The eighteen-epoch run made that concrete: ce sat at 11.4
    against a nominal chance of ln(1000)=6.908, which reads as catastrophic, while
    accuracy sat at 137x chance. Nothing was catastrophic; the number is simply scaled
    by 1/temperature. Changing `temperature` moves it without changing the model.

    The three quantities here are all invariant to `temperature`, so they stay
    comparable across configurations that the loss curve does not.

      gap        mean of d^2(true) - d^2(nearest), in the codebook's own normalised
                 space. Zero when the teacher's code IS the nearest one. This is the
                 quantity the cross-entropy is a rescaling of: a miss costing
                 -ln q ~ 13 at temperature 0.05 corresponds to gap ~ 0.65.
      steps_l1   mean L1 distance in LATTICE INDEX units, i.e. how many single-axis
                 quantiser steps separate the student from the teacher's code. Axis
                 spacing is 1/(levels_d // 2) and so differs per axis (0.25 on the
                 8-level axis, 0.50 on the 5-level axes), which is exactly why this is
                 counted in index units rather than in distance.
      within_one every axis index within one step of the teacher's. The forgiving
                 sibling of masked_acc: `levels: [8,5,5,5]` gives axes with only five
                 positions, so being one step out is a near miss, and a model improving
                 from "three steps out" to "one step out" moves this while leaving
                 accuracy at zero.

    `gap` is read off the logits rather than recomputed, since logits = -d^2/temperature
    already contains every distance, and a second cdist over 1000 codes is not free.
    """
    with torch.no_grad():
        if not bool(mask.any()):
            zero = torch.zeros((), device=z.device)
            return {"gap": zero, "steps_l1": zero, "within_one": zero}

        lg = logits.float()
        best = lg.max(-1).values                                     # (B, N)
        true = lg.gather(-1, target_indices.unsqueeze(-1)).squeeze(-1)
        gap = ((best - true) * temperature)[mask]

        z_norm = normalised_z(z, fsq).float()                        # (B, N, L)
        c_true = fsq.codebook.to(z_norm.dtype)[target_indices]       # (B, N, L)
        half = (fsq._levels // 2).to(z_norm.dtype)
        steps = ((z_norm - c_true) * half).abs()                     # per-axis, index units

        return {
            "gap": gap.mean(),
            "steps_l1": steps.sum(-1)[mask].mean(),
            "within_one": (steps.amax(-1)[mask] <= 1.0).to(torch.float32).mean(),
        }


def codebook_diversity(logits, valid, softmax_scale=1.0, sample_entropy_weight=0.0,
                       eps=1e-8, floor=None):
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

    `floor` switches the term from a CONSTANT PULL to a HINGE, and the two are different
    objectives rather than two strengths of one. Without it the term is -H(E[q])/log K,
    which is minimised only when coverage is perfectly uniform, so it keeps pulling no
    matter how healthy the codebook already is. Measured consequence: coverage is slammed
    to 0.979 by epoch 2 and then spends sixteen epochs being clawed back down to 0.902 by
    the cross-entropy, which is a strange way to spend a run, and the term sits within two
    percent of its floor throughout so it is a near-constant added to the loss rather than
    a regulariser.

    With `floor` set the penalty is max(0, floor - coverage): exactly zero, with exactly
    no gradient, while coverage is above the floor, and linear below it. Entropy is then
    free to grow at whatever rate the cross-entropy wants, and the term does only the one
    job it is needed for, which is preventing the collapse the ablation found at weight
    zero (27 codes of 1000, coverage 0.065).

    Two things to watch, because a hinge is a weaker guard than a constant pull. It is
    REACTIVE, so it does nothing until coverage has already fallen through the floor, and
    if a collapse is fast it may catch it too late; watch train/coverage against
    train/codes_used_batch. And `diversity_weight` means something different under it: the
    hinge's largest possible value is `floor` itself, reached only at total collapse,
    where the constant-pull form sat near 1.0 at all times.

    The hinge is defined on H(E[q]) alone and ignores `sample_entropy_weight`, whose
    meaning inside a floor constraint is not clear. That term is off by default anyway.
    """
    sel = logits[valid] if valid is not None else logits.flatten(0, -2)
    if sel.numel() == 0:
        zero = logits.sum() * 0.0
        return zero, {"diversity": zero.detach(), "soft_bits": zero.detach(),
                      "coverage": zero.detach()}

    q = (sel.float() * softmax_scale).softmax(-1)             # (M, K)
    log_k = math.log(sel.shape[-1])

    batch_ent = -(q.mean(0) * torch.log(q.mean(0) + eps)).sum()        # H(E[q])
    sample_ent = -(q * torch.log(q + eps)).sum(-1).mean()              # E[H(q)]

    coverage = batch_ent / log_k                                       # in [0, 1]
    if floor is None:
        loss = (sample_entropy_weight * sample_ent - batch_ent) / log_k
    else:
        loss = torch.clamp(floor - coverage, min=0.0)
    return loss, {
        "diversity": loss.detach(),
        # H(E[q]) as a fraction of its ceiling. The quantity the floor is compared
        # against, and the one to watch to see whether the hinge ever fires.
        "coverage": coverage.detach(),
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
