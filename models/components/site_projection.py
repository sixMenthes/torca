"""Remove the recording-channel subspace from the TEACHER's tokens before quantisation.

Why this exists
---------------
The first adapted run made hydrophone identity MORE decodable, not less: balanced
accuracy from Background clips went 0.902 frozen to 0.947 adapted, against a chance of
0.100. The mechanism is visible in the tokenizer. The teacher view carried no
augmentation, so it tokenises a clean, channel-specific clip, and CodeUsageProbe measured
that the resulting codes carry 0.51 bits of site information above a shuffled null. The
student is then shown that clip with foreign noise mixed in and scored on reproducing
those exact codes, so the cross-entropy pays it to recover which hydrophone the clip came
from. This module removes the reward at its source, by stripping the channel subspace out
of the teacher's tokens so the target codes cannot encode it.

Why the subspace is estimated BETWEEN sites and not within one
--------------------------------------------------------------
The natural first idea is to run PCA on each hydrophone's samples separately and subtract
that site's leading components. That does not do what it sounds like it does. A channel
signature is approximately CONSTANT across the clips from one hydrophone, and a constant
contributes no variance, so it does not appear in the leading eigenvectors of a
site's own centred covariance at all. Those leading directions are within-site variation:
call against background, sea state, distance, time of day. Subtracting them removes
CONTENT and leaves the channel untouched.

The channel lives in the differences between site MEANS. So the estimate here is the
top-k left subspace of the matrix of per-site mean vectors, centred by their grand mean.
That is Nuisance Attribute Projection, the standard channel compensation of speaker
verification, and it targets exactly the quantity the nuisance probe reads.

One footnote, because it explains why the original intuition is not silly. An UNCENTRED
PCA of a single site's samples has a leading component close to that site's mean
direction, so the uncentred variant of the original idea does approximate the right thing
by accident. Estimating the between-site subspace directly is the deliberate version.

Measured, on synthetic tokens with a constant per-site offset, a shared content
direction and large within-site noise (scratch/test_site_projection.py, 12 sites, 768
dims). Site and content are read by linear probes; chance is 0.083 and 0.500.

    k removed   energy    site   content
         none        -   1.000     1.000
            1    0.062   1.000     1.000
            2    0.120   1.000     1.000
            4    0.229   1.000     1.000
            8    0.423   0.999     1.000
           10    0.511   0.921     1.000
           11    0.553   0.787     1.000
           16    0.553   0.787     1.000
     within-2        -   1.000     0.307

Three things follow, and the first overturned this module's original default of 2.
Removing a couple of components does nothing: the centred mean matrix of S sites has
rank S-1, so with 12 sites there are 11 directions separating them and deleting 2 leaves
9 that a linear probe reads perfectly. k has to approach S-1, which is 11 of 768
dimensions and still surgical. Second, content survives the full removal untouched at
1.000, so this is not a blunt instrument. Third, the within-site variant does exactly
what the argument above predicts: content collapses to 0.307 while site stays at 1.000,
which is the worst of both.

Site does not fall to chance even at k = S-1, stopping at 0.787, because the running
means are EMA estimates rather than exact ones and the residual is still decodable. On
real data the offsets are not exactly constant either, so expect the same. That is not a
defect: the goal is to stop the TARGET CODES from rewarding channel recovery, not to
zero the nuisance probe by construction, and the probe reads the student's unprojected
tokens regardless.

Applied to the teacher ONLY
---------------------------
The student's tokens are left alone. If both were projected, the student's encoder would
face no pressure to remove the channel itself, because the projection would have done the
work. Projecting only the target means the student must LEARN to produce tokens that land
on channel-stripped codes, which is the invariance being asked for. It also keeps the
measurement honest: the nuisance probe reads the student encoder's pooled tokens, a space
nothing here ever projects, so the metric cannot be satisfied by construction.
"""

import torch
from torch import nn


class SiteSubspaceProjection(nn.Module):
    """Running per-site means, and the projection that removes their shared subspace.

    dim:          token width, encoder.embed_dim (768 for both backbones here).
    n_components: how many between-site directions to remove. None, the default, means
                  S-1, the full rank of the centred mean matrix, which is equivalent to
                  removing each site's mean outright and is cepstral mean normalisation
                  in this space. Anything much below S-1 is measurably inert; see the
                  table above, where k=4 of 11 left site decodability at 1.000.
    momentum:     EMA rate for the running means. The encoder moves during training, so
                  the means have to track it rather than be estimated once.
    warmup_steps: the projection is DISABLED until every seen site has contributed at
                  least this many batches. Early means are noise, and projecting on noise
                  would corrupt the targets exactly when the codebook is forming.
    max_sites:    buffer width. Sites are indexed lazily in first-seen order, so this is
                  a capacity bound rather than a fixed vocabulary; the adaptation pool has
                  26 hydrophones.
    """

    def __init__(self, dim, n_components=None, momentum=0.05, warmup_steps=50,
                 max_sites=64):
        super().__init__()
        self.dim = int(dim)
        # None means "all S-1 of them", resolved at use time because S grows as sites
        # are first seen. Stored as None rather than a sentinel int so the intent survives
        # into a config dump.
        self.n_components = None if n_components is None else int(n_components)
        self.momentum = float(momentum)
        self.warmup_steps = int(warmup_steps)
        self.max_sites = int(max_sites)

        # Buffers, not plain tensors: these are state the run depends on, so they belong
        # in the checkpoint. A resumed run that re-estimated its means from scratch would
        # silently change the targets at the resume point.
        self.register_buffer("site_mean", torch.zeros(max_sites, self.dim))
        self.register_buffer("site_updates", torch.zeros(max_sites, dtype=torch.long))
        # Names cannot live in a buffer, so the index map is rebuilt from the order sites
        # are first seen. That is stable given a fixed dataset and seed, and the
        # consequence of it not being is a permuted-but-equivalent set of means.
        self._index = {}
        # Diagnostics, refreshed on every forward. Plain floats, not buffers: they are
        # observations about the last batch, not state the run resumes from.
        self.last_energy = 0.0
        self.last_k = 0

    def site_indices(self, names, device):
        """Hydrophone names -> long indices, assigning new ones on first sight."""
        out = []
        for n in names:
            if n not in self._index:
                if len(self._index) >= self.max_sites:
                    # Over capacity: bucket the overflow rather than crash a long run.
                    out.append(0)
                    continue
                self._index[n] = len(self._index)
            out.append(self._index[n])
        return torch.tensor(out, dtype=torch.long, device=device)

    @property
    def n_sites(self):
        return len(self._index)

    @torch.no_grad()
    def update(self, tokens, site_idx, valid=None):
        """EMA the per-site mean token. tokens (B, N, D), site_idx (B,)."""
        if valid is not None:
            w = valid.unsqueeze(-1).to(tokens.dtype)              # (B, N, 1)
            denom = w.sum(1).clamp_min(1.0)                       # (B, 1)
            per_clip = (tokens * w).sum(1) / denom                # (B, D)
        else:
            per_clip = tokens.mean(1)                             # (B, D)
        per_clip = per_clip.to(self.site_mean.dtype)

        for s in torch.unique(site_idx):
            sel = site_idx == s
            batch_mean = per_clip[sel].mean(0)
            i = int(s)
            if self.site_updates[i] == 0:
                self.site_mean[i] = batch_mean
            else:
                self.site_mean[i].mul_(1 - self.momentum).add_(
                    batch_mean, alpha=self.momentum
                )
            self.site_updates[i] += 1

    @torch.no_grad()
    def ready(self):
        """True once every seen site has enough updates for its mean to mean anything."""
        if self.n_sites < 2:
            return False
        if self.n_components is not None and self.n_components < 1:
            return False
        seen = self.site_updates[: self.n_sites]
        return bool((seen >= self.warmup_steps).all())

    @torch.no_grad()
    def basis(self):
        """(D, k) orthonormal basis of the top between-site directions, or None."""
        if not self.ready():
            return None
        means = self.site_mean[: self.n_sites].float()            # (S, D)
        centred = means - means.mean(0, keepdim=True)             # grand mean removed
        # Rank of the centred mean matrix is at most S-1, so asking for more components
        # than that returns directions that are numerically arbitrary.
        want = self.n_sites - 1 if self.n_components is None else self.n_components
        k = min(want, self.n_sites - 1, self.dim)
        if k < 1:
            return None
        # Right singular vectors of (S, D) span the row space, which is what we want to
        # remove from a (…, D) token.
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        return vh[:k].transpose(0, 1).contiguous()                # (D, k)

    @torch.no_grad()
    def forward(self, tokens):
        """Remove the between-site subspace. Identity while warming up."""
        v = self.basis()
        if v is None:
            self.last_energy = 0.0
            self.last_k = 0
            return tokens
        v = v.to(tokens.dtype)
        along = tokens @ v
        # Recorded here rather than recomputed by energy_removed, which would redo the
        # SVD. This is the number that says whether the arm is doing anything: near zero
        # means the between-site directions carry nothing and the run is C4 with extra
        # steps; near one means it is deleting most of the representation, which would be
        # a bug rather than a result.
        total = float(tokens.pow(2).sum())
        self.last_energy = float(along.pow(2).sum()) / total if total > 0 else 0.0
        self.last_k = int(v.shape[1])
        return tokens - along @ v.transpose(0, 1)

    @torch.no_grad()
    def energy_removed(self, tokens):
        """Fraction of squared token norm the projection strips. A diagnostic.

        Near zero means the between-site directions carry nothing and the arm is inert.
        Near one means it is removing most of the representation, which would be a bug
        or a badly chosen n_components rather than a result.
        """
        v = self.basis()
        if v is None:
            return None
        v = v.to(tokens.dtype)
        total = tokens.pow(2).sum()
        if total <= 0:
            return None
        return float((tokens @ v).pow(2).sum() / total)
