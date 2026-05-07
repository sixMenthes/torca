import torch
import polars as pl
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

import warnings
warnings.filterwarnings("ignore", message=".*has been deprecated.*")


class LocalDataset(Dataset):
    def __init__(self, polars_df, label_map=None, transform=None):
        super().__init__()
        self.df = polars_df
        self.labels = self.df.get_column("Labels")
        self.transform = transform
        self.params = {
                        'clipDur':3,
                        'outSR': 16000,
                        'fmin':0
                    }
        unique_labels = self.labels.unique().to_list()
        if not label_map:
            self.label_map = {name: i for i, name in enumerate(unique_labels)}
        else:
            self.label_map = label_map

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        audio_path = self.df["clip_path"][idx]
        decoder = AudioDecoder(audio_path, sample_rate=self.params['outSR'])
        metadata = decoder.metadata
        start_time, end_time = self.get_start(metadata, idx)
        audio_data = decoder.get_samples_played_in_range(start_time, end_time).data
        audio_data = audio_data[0]
        label = torch.tensor(self.label_map[self.labels[idx]], dtype=torch.long)
        return audio_data, label
        
    def get_start(self, metadata, idx):
        start_time = self.df['FileBeginSec'][idx]
        end_time = self.df['FileEndSec'][idx]
        file_duration = metadata.duration_seconds
        sample_rate = metadata.sample_rate
        duration = end_time - start_time
        center_time = start_time + duration / 2.0
        new_start_time = max(0, center_time - (self.params['clipDur'] / 2.0))
        new_end_time = new_start_time + self.params['clipDur']
        if new_end_time > file_duration:
            new_start_time = max(0, file_duration - self.params['clipDur'])
            new_end_time = file_duration
        return (new_start_time, new_end_time)

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





