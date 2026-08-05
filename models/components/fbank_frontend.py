import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import fbank


class KaldiFbank(nn.Module):
    """Bird-MAE / AudioMAE's native kaldi-fbank front-end, as an nn.Module.

    Faithful port of Bird-MAE/transforms.py::BaseTransform's `fbank` path
    (_process_waveforms -> _compute_fbank_features -> _pad_and_normalize ->
    standardize). Bird-MAE (ckpt AudioMAE_XCL_epoch=99_mixup) was pretrained on
    THIS front-end — log-mel kaldi fbank, htk_compat, hanning window, 128 mel,
    25/10 ms @ 32 kHz — NOT the torchaudio Spectrogram->MelScale path that torca's
    VIT currently feeds it. Using this removes the frozen-backbone OOD mismatch and
    matches the *form* of BEATs' own internal fbank, so both backbones can consume
    raw waveform (the self-distillation datamodule serves waveforms).

    Fidelity details taken from Bird-MAE's code (differ from BEATs.preprocess!):
      - the waveform is mean-SUBTRACTED and NOT scaled by 2**15 (BEATs scales by
        2**15 and does not mean-subtract);
      - window_type='hanning', htk_compat=True (BEATs uses kaldi defaults);
      - time padding uses the fbank's MIN value, not zero;
      - standardization is (fbank - mean) / (2*std).
    So this is ONE module class with per-backbone params, not a literally identical
    fbank across backbones. BEATs keeps its own BEATs.preprocess; this is for VIT.

    NOTE: this is the LOG-mel path. The PCEN ablation cell needs LINEAR mel power,
    so it uses the torchaudio mel + PCEN front-end instead of this module.

    Input : (B, T) or (B, 1, T) waveform at `sample_frequency`.
    Output: (B, 1, target_length, num_mel_bins), standardized, ready for patch_embed.
    """

    def __init__(self, sample_frequency=32000, num_mel_bins=128, target_length=304,
                 frame_shift=10.0, window_type="hanning", htk_compat=True,
                 use_energy=False, dither=0.0, mean=-45.198, std=14.855,
                 subtract_waveform_mean=True):
        super().__init__()
        self.sample_frequency = sample_frequency
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.frame_shift = frame_shift
        self.window_type = window_type
        self.htk_compat = htk_compat
        self.use_energy = use_energy
        self.dither = dither
        # (fbank - mean) / (2*std). Defaults = torca's DCLDE stats. For a FROZEN
        # Bird-MAE baseline these should instead be Bird-MAE's XCL PRETRAINING
        # stats (OPEN — not in the yaml; confirm from the feature-extractor /
        # checkpoint). Adaptation is far less sensitive since the weights move.
        self.mean = mean
        self.std = std
        self.subtract_waveform_mean = subtract_waveform_mean

    def _fit_time(self, fb):
        # fb: (num_frames, mel) -> pad/crop time to target_length.
        n = fb.shape[0]
        if n < self.target_length:            # Bird-MAE pads with the fbank min value
            fb = F.pad(fb, (0, 0, 0, self.target_length - n), value=float(fb.min()))
        elif n > self.target_length:
            fb = fb[: self.target_length]
        return fb

    def forward(self, wave):
        if wave.dim() == 3:                    # (B, 1, T) -> (B, T)
            wave = wave.squeeze(1)
        feats = []
        for w in wave:                          # kaldi fbank is not batched (per-sample)
            if self.subtract_waveform_mean:
                w = w - w.mean()
            with torch.no_grad():               # fixed front-end: no grad through fbank
                fb = fbank(
                    w.unsqueeze(0),
                    htk_compat=self.htk_compat,
                    sample_frequency=self.sample_frequency,
                    use_energy=self.use_energy,
                    window_type=self.window_type,
                    num_mel_bins=self.num_mel_bins,
                    dither=self.dither,
                    frame_shift=self.frame_shift,
                )                               # (num_frames, num_mel_bins)
            feats.append(self._fit_time(fb))
        x = torch.stack(feats, dim=0)           # (B, target_length, mel)
        x = (x - self.mean) / (self.std * 2.0)
        return x.unsqueeze(1)                    # (B, 1, target_length, mel)
