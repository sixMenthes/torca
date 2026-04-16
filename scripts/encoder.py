import torch
import torch.nn as nn
import torchaudio
from ifsq import FSQ
from foundation_model.BEATs import BEATs, BEATsConfig

def load_beats(beats_ckpt):
    # load the pre-trained checkpoints
    checkpoint = torch.load(beats_ckpt, map_location="cpu")

    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    for p in BEATs_model.parameters():
        p.requires_grad_(False)

    return BEATs_model

    

class Encoder(nn.Module):
    def __init__(self, beats_ckpt, levels):
        super().__init__()
        self.backbone = load_beats(beats_ckpt)
        self.proj = nn.Linear(768, len(levels))
        self.fsq = FSQ(levels)

    def forward(self, path_to_audio):
        audio = torchaudio.load_with_torchcodec(path_to_audio)[0]
        h = self.backbone.extract_features(audio, padding_mask=torch.zeros_like(audio).bool())[0]
        h = self.proj(h)
        h = self.fsq.quantize(h)
        return self.fsq.codes_to_indices(h)

enc = Encoder('../models/BEATs_iter3.pt', [8, 6, 6])
over = '/Users/leo/projects/orcas/ds/overlaps.wav'
x = enc.forward(over)
