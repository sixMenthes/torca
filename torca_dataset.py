from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
from util.pylogger import get_pylogger
import torch.nn.functional as F
import polars as pl
import os 
import torch
from torca_transforms import BaseTransform

log = get_pylogger(__name__)

class TorcaDataset(Dataset):
    def __init__(self, df:pl.DataFrame, transform:BaseTransform, label_map: dict):
        self.df = df
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return self.df.height

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        path = row["LocalPath"]
        if os.path.exists(path):
            label = self.label_map[row["Labels"]]
            label = F.one_hot(torch.tensor(label))
            wave = AudioDecoder(path, sample_rate=32000).get_all_samples()
            features = self.transform(wave.data)
            return {"audio": features, "label": label}
        else:
            log.warning(f"Failed loading file \t {path}")
            return None