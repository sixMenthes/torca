from torch.utils.data import Dataset
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchvision
import os

from util.pylogger import get_pylogger
# Reuse the waveform-level augmentations already defined on this branch rather
# than duplicating them: the self-distillation views are exactly shift/gain/
# background-noise applied asymmetrically (weak on teacher, strong on student).
from torca_transforms import RandomShift, RandomGain, AddBackgroundNoise  # noqa: F401

log = get_pylogger(__name__)


class WaveformViewAug:
    """Compose waveform-level augmentations into a single view generator.

    Operates on a (1, T) waveform and returns a (1, T) waveform. The spectrogram
    / PCEN front-end deliberately lives in the *model*, so BEATs (raw wave ->
    internal kaldi fbank) and Bird-MAE (wave -> mel front-end) consume the same
    augmented waveform. That keeps the augmentation homogeneous across backbones
    and gives PCEN a single seam to slot into later.
    """

    def __init__(self, ops):
        self.aug = torchvision.transforms.Compose(ops) if ops else None

    def __call__(self, wave):
        return self.aug(wave) if self.aug else wave


class SelfDistillDataset(Dataset):
    """Two-view self-distillation dataset (iBOT/data2vec-style adaptation).

    Each item yields a *teacher* view (weak/clean) and a *student* view (strong
    augmentation, incl. cross-hydrophone background noise) of the SAME clip. No
    labels — this is unsupervised domain adaptation; the span masking that makes
    the objective non-trivial is applied to the student inside the model, not
    here. The hydrophone id (`dataset`) is carried through so post-adaptation
    diagnostics can group by recording condition.
    """

    def __init__(self, df, teacher_aug, student_aug, sample_rate, max_length):
        self.df = df
        self.teacher_aug = teacher_aug
        self.student_aug = student_aug
        self.sample_rate = int(sample_rate)
        self.max_length = int(max_length)

    def __len__(self):
        return self.df.height

    def _load_wave(self, path):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).T                         # (C, T)
        # highest-energy channel (mirrors BaseTransform's channel pick)
        idx = int(torch.argmax((wave ** 2).mean(1))) if wave.size(0) > 1 else 0
        wave = wave[idx].unsqueeze(0)                            # (1, T)
        # Resample to a common rate. Low-SR hydrophones upsample (their empty
        # high bands are real channel variety the invariance objective should
        # learn to ignore, not a bug); high-SR downsample. BEATs needs 16 kHz.
        if sr != self.sample_rate:
            wave = AF.resample(wave, sr, self.sample_rate)
        return self._fit_length(wave)          # (wave, n_valid_samples)

    def _fit_length(self, wave):
        """Fit to max_length, and report how much of the result is REAL audio.

        90% of DCLDE clips are shorter than a 3 s window (the slicer cuts
        [floor(begin), ceil(end)] around each annotation), so the tail padding is the
        common case, not the exception. The model needs the valid count to keep that
        synthetic silence out of the loss — without it the padding is
        indistinguishable from quiet ocean once it reaches the front-end.
        """
        t = wave.size(-1)
        if t > self.max_length:                                 # center crop
            start = (t - self.max_length) // 2
            wave = wave[..., start:start + self.max_length]
        elif t < self.max_length:                              # zero-pad tail
            wave = F.pad(wave, (0, self.max_length - t))
        return wave, min(t, self.max_length)

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        path = row["LocalPath"]
        if not os.path.exists(path):
            log.warning(f"Failed loading file \t {path}")
            return None
        wave, n_valid = self._load_wave(path)
        # clone() so the two aug pipelines can't alias the same underlying tensor
        return {
            "teacher": self.teacher_aug(wave.clone()),
            "student": self.student_aug(wave.clone()),
            "dataset": row["Dataset"],
            # samples of real audio before tail padding; the model turns this into a
            # per-patch validity mask. NOTE: RandomShift moves the signal inside the
            # window, so this is an upper bound on where real audio sits after
            # augmentation — fine, since the teacher (which defines the targets) sees
            # the unshifted clean view.
            "n_valid": n_valid,
        }
