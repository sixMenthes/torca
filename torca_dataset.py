from torch.utils.data import Dataset
import soundfile as sf
from util.pylogger import get_pylogger
import torch.nn.functional as F
import polars as pl
import os 
import torch
from torca_transforms import BaseTransform

log = get_pylogger(__name__)

class LabelDataset(Dataset):
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
            num_classes = len(self.label_map.keys())
            label = F.one_hot(torch.tensor(label), num_classes = num_classes)
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            wave = torch.from_numpy(audio).T
            features = self.transform(wave.data)
            return {"audio": features, "label": label}
        else:
            log.warning(f"Failed loading file \t {path}")
            return None
