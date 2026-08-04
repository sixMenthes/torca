import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

warnings.filterwarnings("ignore", message=".*has been deprecated.*")


class LocalDataset(Dataset):
    """
    Reads pre-sliced local audio clips (one file per annotation, cut and centered
    on the annotation by LabelDataModule) and returns a raw waveform + integer
    label. The clip is decoded, resampled to `sample_rate`, and center-cropped or
    symmetrically padded to `clip_duration` seconds. BEATs computes its own fbank,
    so no spectrogram transform happens here.

    Because the source clips are centered on the annotation, center-cropping keeps
    the call centered when `clip_duration` (e.g. 3 s) is shorter than the source
    clip (e.g. 5 s). Set `clip_duration` equal to the source length to use the
    whole clip. (Edge annotations near a file boundary were cut off-center by
    LabelDataModule, so center-crop is approximate there.)

    Expected columns: clip_path, Labels.
    """

    def __init__(self, polars_df, label_map=None, clip_duration=3.0, sample_rate=16000):
        super().__init__()
        self.df = polars_df
        self.labels = self.df.get_column("Labels")
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.target_samples = int(clip_duration * sample_rate)
        unique_labels = self.labels.unique().to_list()
        if not label_map:
            self.label_map = {name: i for i, name in enumerate(unique_labels)}
        else:
            self.label_map = label_map

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        audio_path = self.df["clip_path"][idx]
        decoder = AudioDecoder(audio_path, sample_rate=self.sample_rate)
        duration = decoder.metadata.duration_seconds
        audio_data = decoder.get_samples_played_in_range(0, duration).data
        audio_data = audio_data[0]  # mono: first channel
        audio_data = self._center_fit(audio_data, self.target_samples)
        label = torch.tensor(self.label_map[self.labels[idx]], dtype=torch.long)
        return audio_data, label

    @staticmethod
    def _center_fit(audio, target):
        n = audio.shape[-1]
        if n == target:
            return audio
        if n > target:  # center crop
            start = (n - target) // 2
            return audio[start:start + target]
        # symmetric pad to keep the annotation centered
        total = target - n
        left = total // 2
        return F.pad(audio, (left, total - left))


def collate_fn(batch):
    samples, labels = zip(*batch)
    labels = torch.stack(labels)
    lengths = [s.shape[0] for s in samples]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len)
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i, (s, L) in enumerate(zip(samples, lengths)):
        padded[i, :L] = s
        mask[i, :L] = False
    return padded, mask, labels
