import torch
import torch.nn as nn
import torchaudio
import copy
import math
import itertools
from ifsq import FSQ
from foundation_model.BEATs import BEATs, BEATsConfig
from classifier import TransformerStack

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

def spec_augment(x):
    assert x.size() == (1, 408, 3)
    
     

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
    def __init__(self, levels, d_model, heads, d_ff, d_o, num_trans, dropout):
        super().__init__()
        self.quant = Quantizer('../models/BEATs_iter3.pt', levels)
        self.emb = nn.Linear(len(levels), d_model)
        self.trans = nn.ModuleList([TransformerStack(d_model=d_model, number_heads=heads, d_ff=d_ff, dropout=dropout) for _ in range(num_trans)])
        self.classif_head = nn.Linear(d_model, d_o)
        self.masked_loss_head = nn.Linear(d_model, math.prod(levels))
    
    def forward(self, x, mask=None):
        x = self.quant(x) # FSQ object => we have the indices
        indices = x.codes_to_indices() #(maybe I need to clone them)
        x = self.emb(x)
        x = self.trans(x)
        clas = self.classif_head(x.mean(dim=1)) # for classification you need to aggregate the prediction for the patches, worth trying mean pooling too for instance
        masked_prediction = self.masked_loss_head(x)
        return clas, masked_prediction


enc = Quantizer('../models/BEATs_iter3.pt', [8, 6, 6])
cla = Classifier([8, 6, 6], 64, 4, 72, 15, dropout=0.1)
over = '/Users/leo/projects/orcas/ds/3secs.wav'
x = enc.forward(over)
