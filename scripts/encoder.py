import torch
import torch.nn as nn
import torchaudio
import copy
import math
import itertools
from ifsq import FSQ
from foundation_model.BEATs import BEATs, BEATsConfig

def load_beats(beats_ckpt):
    # load the pre-trained checkpoints
    checkpoint = torch.load(beats_ckpt, map_location="cpu")

    cfg_beats = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg_beats)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval()

    for p in BEATs_model.parameters():
        p.requires_grad_(False)

    return BEATs_model
     

class Quantizer(nn.Module):
    def __init__(self, beats_ckpt, levels):
        super().__init__()
        self.encoder = load_beats(beats_ckpt)
        self.proj = nn.Linear(768, len(levels))
        self.fsq = FSQ(levels)

    def forward(self, path_to_audio):
        audio = torchaudio.load_with_torchcodec(path_to_audio)[0]
        h = self.encoder.extract_features(audio, padding_mask=torch.zeros_like(audio).bool())[0]
        h = self.proj(h)
        h = self.fsq.quantize(h)
        return h

class Classifier(nn.Module):
    def __init__(self, levels, d_model, heads, d_ff, d_o, dropout):
        super().__init__()
        self.emb = nn.Linear(len(levels), d_model)
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.classif_head = nn.Linear(d_model, d_o)
        self.masked_loss_head = nn.Linear(d_model, math.prod(levels))
    
    def forward(self, x, mask=None):
        h = self.emb(x)
        h = self.ffn(self.attn(h, h, h, mask))
        c = self.classif_head(h.mean(dim=1)) # for classification you need to aggregate the prediction for the patches, worth trying mean pooling too for instance
        m = self.masked_loss_head(h)
        return c, m




enc = Quantizer('../models/BEATs_iter3.pt', [8, 6, 6])
cla = Classifier([8, 6, 6], 64, 4, 72, 15, dropout=0.1)
over = '/Users/leo/projects/orcas/ds/overlaps.wav'
x = enc.forward(over)
