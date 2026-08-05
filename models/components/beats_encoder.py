"""Standalone BEATs encoder adapter: raw waveform -> (B, N, 768) patch tokens.

The BEATs half of the uniform encoder interface (`tokens(waveform) -> (B, N, 768)`),
matching BirdMAEEncoder so the self-distillation loop is backbone-agnostic.

`BEATs.extract_features` bundles fbank -> patch_embedding -> encoder into one call,
which is fine for a frozen probe but leaves nowhere to inject a `[MASK]` token: the
student view needs masking applied to the *patch features*, after projection and
before the transformer. So this wrapper re-implements that call's body (it is ~10
lines) with the seam opened up, rather than wrapping it.

Note BEATs has no fixed input length — no positional table to resize, so nothing
here corresponds to the target_length surgery Bird-MAE needs. Frame count follows the
waveform: 3 s @ 16 kHz -> 298 kaldi frames -> a (18, 8) patch grid = 144 tokens
(the conv is stride-16 and does not pad, so the last 10 frames are dropped). That is
a different N from Bird-MAE's 152 at the same 3 s, which is expected and harmless:
the objective is per-token, and the two backbones are never compared token-for-token.
"""

import torch
import torch.nn as nn

# Absolute import so this file can also be run directly from the repo root
# (`ipython -i models/components/beats_encoder.py`), same as test_beats.py does.
from models.components.beats.beats import BEATs, BEATsConfig


class BEATsEncoder(nn.Module):
    """BEATs (iter3 SSL checkpoint) as a token encoder.

    Use the plain SSL iter3 checkpoint, NOT iter3+AS20K: the study compares SSL
    against SSL (Bird-MAE), and the AS20K-finetuned variant would smuggle in
    supervised AudioSet labels as a confound. A finetuned checkpoint also builds a
    `predictor` head, which this wrapper ignores in any case.

    Input : (B, T) or (B, 1, T) waveform at 16 kHz.
    Output: (B, N, 768) patch tokens.
    """

    def __init__(self, pretrained_weights_path=None, sample_rate=16000,
                 fbank_mean=15.41663, fbank_std=6.55582, cfg=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std

        if pretrained_weights_path:
            ckpt = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
            self.beats = BEATs(BEATsConfig(ckpt["cfg"]))
            info = self.beats.load_state_dict(ckpt["model"], strict=False)
            if info.missing_keys or info.unexpected_keys:
                print(f"[beats] {len(info.missing_keys)} missing, "
                      f"{len(info.unexpected_keys)} unexpected keys")
        else:
            self.beats = BEATs(BEATsConfig(cfg))

        self.embed_dim = self.beats.cfg.encoder_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @property
    def freq_patches(self):
        """Patches per frequency column (128 mel / patch size) — 8 for iter3.

        BEATs has no fixed time extent, so unlike Bird-MAE there is no total
        num_patches until a waveform arrives; only this stride is static, and it is
        what makes token index -> (time, freq) decodable the same way for both.
        """
        return 128 // self.beats.cfg.input_patch_size

    def features(self, wave):
        """Waveform -> projected patch features (B, N, 768), pre-encoder.

        Mirrors the first half of BEATs.extract_features. padding_mask is dropped:
        the SSL dataset serves fixed-length clips, so every position is valid.
        """
        if wave.dim() == 3:                       # (B, 1, T) -> (B, T)
            wave = wave.squeeze(1)
        with torch.no_grad():                     # fbank is a fixed front-end
            fbank = self.beats.preprocess(
                wave, fbank_mean=self.fbank_mean, fbank_std=self.fbank_std
            )
        x = self.beats.patch_embedding(fbank.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.beats.layer_norm(x)
        if self.beats.post_extract_proj is not None:
            x = self.beats.post_extract_proj(x)
        return x

    def tokens(self, wave, mask=None, layer=None):
        """Waveform -> encoder output tokens.

        mask: optional bool (B, N); True positions are replaced by `mask_token`
        after projection and before the encoder, so masked positions still receive
        context through attention — the whole point of masked self-distillation.

        layer: stop after this encoder layer (0-based) instead of running the full
        stack, for probing where the confound lives. BEATs supports this natively;
        note its encoder also skips its own final layer_norm when a target layer is
        given, which matches how BirdMAEEncoder truncates.
        """
        x = self.features(wave)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = self.beats.dropout_input(x)
        x, _ = self.beats.encoder(x, padding_mask=None, layer=layer)
        return x

    def forward(self, wave, mask=None):
        return self.tokens(wave, mask=mask)

    @torch.no_grad()
    def pooled(self, wave, layer=None):
        """Mean-pooled clip embedding (B, D) — the feature the linear probe eats."""
        return self.tokens(wave, layer=layer).mean(dim=1)
