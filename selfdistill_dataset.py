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

    def __init__(self, df, teacher_aug, student_aug, sample_rate, max_length,
                 frontend=None, deterministic=False, seed=59):
        self.df = df
        self.teacher_aug = teacher_aug
        self.student_aug = student_aug
        self.sample_rate = int(sample_rate)
        self.max_length = int(max_length)
        # Fbank front-end, applied HERE rather than in the model. kaldi's fbank is
        # not batched, so in the model it is a serial Python loop over the batch,
        # run twice per step (teacher + student). Measured at ~1.9 ms/sample scaling
        # 15.14x from batch 1 to 16 — negligible against a CPU forward, dominant
        # against an H100 one. In the dataset it runs inside the dataloader workers,
        # parallel across num_workers and overlapped with the GPU.
        #
        # Safe because the fbank is FIXED: no parameters, no gradient (it is already
        # under no_grad). Augmentation still happens on the waveform, before this.
        # The PCEN cell must keep its front-end in the model, since PCEN is trainable
        # and needs linear mel — pass frontend=None there.
        self.frontend = frontend
        # deterministic=True: every clip gets the SAME augmentation every epoch, so a
        # validation curve reflects the model changing rather than the noise draw
        # changing. All three augmentations use torch's global RNG (torch.rand /
        # randint / empty().uniform_), so seeding it per item is sufficient.
        self.deterministic = deterministic
        self.seed = int(seed)

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

        Padding is the EXCEPTION, not the common case. build_clip_manifest centres a
        fixed clip_duration window on the annotation midpoint and clamps it at the file
        start, so there is no head padding and a clip can only fall short when the SOURCE
        FILE ENDS first. Measured on a real run, train/valid_frac is 0.945 — about 5% of
        positions are synthetic silence.

        (An earlier comment here claimed "90% of DCLDE clips are shorter than a 3 s
        window (the slicer cuts [floor(begin), ceil(end)] around each annotation)". That
        described an OLDER slicer whose clip length followed the annotation length, and
        it is not what build_clip_manifest does. The same wrong figure had propagated
        into four other files.)

        The valid count is still worth reporting even at 5%: padding is indistinguishable
        from quiet ocean once it reaches the front-end, and it tokenises to one constant
        code, so it biases any per-subset code histogram by however much the subsets'
        durations differ.
        """
        t = wave.size(-1)
        if t > self.max_length:                                 # center crop
            start = (t - self.max_length) // 2
            wave = wave[..., start:start + self.max_length]
        elif t < self.max_length:                              # zero-pad tail
            wave = F.pad(wave, (0, self.max_length - t))
        return wave, min(t, self.max_length)

    def _view(self, aug, wave):
        """Augment on the WAVEFORM, then apply the fixed front-end.

        Order is not negotiable: the cross-hydrophone noise has to be mixed into the
        audio, not into a spectrogram, or the de-confounding lever is measuring
        something else entirely.

        The front-end is fed a 1-D waveform and returns (1, frames, mel), so
        default_collate stacks views into (B, 1, frames, mel) — exactly the rank the
        encoders treat as "already preprocessed".
        """
        out = aug(wave.clone())
        return out if self.frontend is None else self.frontend(out.squeeze(0))

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        path = row["LocalPath"]
        if not os.path.exists(path):
            log.warning(f"Failed loading file \t {path}")
            return None
        wave, n_valid = self._load_wave(path)
        # clone() so the two aug pipelines can't alias the same underlying tensor
        if self.deterministic:
            # fork_rng(devices=[]) forks only the CPU generator — augmentation runs in
            # dataloader workers on CPU — and restores it after, so seeding here can't
            # perturb the training RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + index)
                teacher = self._view(self.teacher_aug, wave)
                student = self._view(self.student_aug, wave)
            return {"teacher": teacher, "student": student,
                    "dataset": row["Dataset"], "n_valid": n_valid}
        return {
            "teacher": self._view(self.teacher_aug, wave),
            "student": self._view(self.student_aug, wave),
            "dataset": row["Dataset"],
            # samples of real audio before tail padding; the model turns this into a
            # per-patch validity mask. NOTE: RandomShift moves the signal inside the
            # window, so this is an upper bound on where real audio sits after
            # augmentation — fine, since the teacher (which defines the targets) sees
            # the unshifted clean view.
            "n_valid": n_valid,
        }
