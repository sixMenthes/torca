import torch
import torch.nn as nn
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
        #self.fsq = FSQ(levels)

    def forward(self, h):
        h = self.backbone(h)
        return h

