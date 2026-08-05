"""Standalone Bird-MAE encoder adapter: raw waveform -> (B, N, 768) patch tokens.

This is the Bird-MAE half of the uniform encoder interface the self-distillation
loop is built on (`tokens(waveform) -> (B, N, embed_dim)`); the BEATs half wraps
`BEATs.extract_features` the same way. Two things it deliberately is NOT:

  * not a LightningModule — no hydra/lightning import, so it can be driven from a
    plain script (see test_birdmae.py) exactly like BEATs can, and dropped into the
    SelfDistillMIM module later without dragging the finetune stack along;
  * not a classifier — no head, no pooling baked in. The self-distillation loss
    wants per-patch tokens, and the linear probe wants its own pooling choice.

Front-end is `KaldiFbank`, Bird-MAE's *native* kaldi fbank rather than torca VIT's
torchaudio Spectrogram->MelScale path — the mismatch that likely inflated the "frozen
Bird-MAE is poor" result. Length is a constructor argument: at 3 s / target_length
304 the patch grid is (19, 8) = 152 tokens, versus (32, 8) = 256 at the pretrained
5 s. Only `pos_embed` depends on that, and it is fixed sincos, so it is regenerated
rather than loaded (see audiomae_loading).
"""

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed

# Absolute (repo-root-relative) imports, matching util.pos_embed / test_beats.py:
# they resolve both when this is imported as a package module AND when the file is
# run directly (`ipython -i models/components/birdmae_encoder.py` from the repo
# root), which relative imports cannot do — a directly-run file is __main__ with no
# parent package.
from models.components.fbank_frontend import KaldiFbank
from models.components.audiomae_loading import audiomae_pos_embed, load_audiomae_weights


class BirdMAEEncoder(nn.Module):
    """Bird-MAE-B (ViT-B/16, AudioMAE-pretrained on XCL @32 kHz) as a token encoder.

    Args mirror configs/module/network/vit_base_16_dclde.yaml so the frozen baseline
    and the adapted encoder are the same network. `mask_token` is created up front
    (cheap, 768 floats) so the student path can mask patches without a second class:
    pass a bool mask of shape (B, N) to `tokens`.

    Input : (B, T) or (B, 1, T) waveform at `sample_rate`.
    Output: (B, N, embed_dim) patch tokens, cls token dropped (see `tokens`).
    """

    def __init__(
        self,
        target_length=304,
        num_mel_bins=128,
        sample_rate=32000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        eps=1e-6,
        drop_path=0.0,
        patch_size=16,
        fbank_mean=-45.198,
        fbank_std=14.855,
        pretrained_weights_path=None,
    ):
        super().__init__()
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        # kept (not just handed to the front-end) so callers can convert sample counts
        # to frames/patches without knowing which backbone they hold — BEATsEncoder
        # exposes the same attribute.
        self.sample_rate = sample_rate
        img_size = (target_length, num_mel_bins)

        self.frontend = KaldiFbank(
            sample_frequency=sample_rate,
            num_mel_bins=num_mel_bins,
            target_length=target_length,
            mean=fbank_mean,
            std=fbank_std,
        )

        # Built the same way VIT does, so checkpoint keys line up 1:1.
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            num_classes=0,
            drop_path_rate=drop_path,
        )
        self.vit.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)

        # Fixed (non-learned) sincos table at THIS grid — never loaded from the
        # checkpoint, which carries the pretraining length's table instead.
        num_patches = self.vit.patch_embed.num_patches
        self.vit.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.reset_pos_embed()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        if pretrained_weights_path:
            self.load_pretrained(pretrained_weights_path)

    @property
    def num_patches(self):
        return self.vit.patch_embed.num_patches

    @property
    def grid_size(self):
        """(time_patches, freq_patches) — (19, 8) at target_length 304."""
        return self.vit.patch_embed.grid_size

    @property
    def freq_patches(self):
        """Patches per frequency column — the stride for time-major token indexing."""
        return self.vit.patch_embed.grid_size[1]

    def reset_pos_embed(self):
        self.vit.pos_embed.data = audiomae_pos_embed(
            self.embed_dim, self.target_length, self.num_mel_bins, self.patch_size
        )

    def load_pretrained(self, path):
        """Load Bird-MAE weights into the ViT, then restore the sincos table.

        `load_audiomae_weights` drops the checkpoint's `pos_embed` whenever its length
        disagrees with ours; the reset afterwards covers the case where lengths DO
        agree, so the table is always this grid's sincos and never a stale copy.
        """
        info, dropped = load_audiomae_weights(self.vit, path)
        self.reset_pos_embed()
        return info, dropped

    def tokens(self, wave, mask=None, return_cls=False, layer=None):
        """Waveform -> patch tokens.

        mask: optional bool (B, N); True positions have their patch embedding
        replaced by `mask_token` BEFORE the position table is added — the iBOT/BEiT
        convention, so a masked token still knows where it is.

        layer: stop after this block index (0-based) instead of running the full
        stack — for probing where in the network the recording-condition confound
        lives. The final LayerNorm is applied ONLY to the full-depth output: it was
        trained to normalise the last block, so applying it to block 4 would be
        borrowing statistics from a distribution that block never sees. BEATs is
        truncated the same way (its encoder skips its own final norm when a target
        layer is given), so the two backbones stay comparable.
        """
        x = self.frontend(wave)  # (B, 1, target_length, mel)
        x = self.vit.patch_embed(x)  # (B, N, D)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)

        x = x + self.vit.pos_embed[:, 1:, :]
        cls = self.vit.cls_token + self.vit.pos_embed[:, :1, :]
        x = torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x)

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if layer is not None and i == layer:
                break
        else:
            x = self.vit.norm(x)

        return (x[:, 1:, :], x[:, 0]) if return_cls else x[:, 1:, :]

    def forward(self, wave, mask=None):
        return self.tokens(wave, mask=mask)

    @torch.no_grad()
    def pooled(self, wave, layer=None):
        """Mean-pooled clip embedding (B, D) — the feature the linear probe eats.

        This is the PRE-projector representation, which is what SSL evaluation
        protocols probe (SimCLR/DINO/VICReg all evaluate the backbone and discard the
        projection head, since the head sheds task-relevant information in service of
        the pretext objective).
        """
        return self.tokens(wave, layer=layer).mean(dim=1)
