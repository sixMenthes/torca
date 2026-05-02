import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from utils import clones, make_mask, get_alibi
from fsq import FSQ
from cfg import TorcaConfig
from foundation_model.BEATs import BEATs, BEATsConfig
from classifier import TransformerStack

"""
# Params:
- levels: list of int that determines the size of the output vector through len(levels) and the quantization steps
- d_model: modulo heads == 0, size of the embeddings
- heads: number of attention heads
- d_ff: dimensions in the FFN
- d_o: number of classes in the classification task
- num_trans: how many transformer layers are stacked
- clas_loss_coef: coefficient for the classification loss
- dropout: percent of neurons to drop
# Given by the backbone (fbank + patch function)
- grid_time = ((seconds * 1000) / frame_shift) // patch_size
- grid_freq = num_mel_bins // patch_size
- seq_len = grid_time * grid_freq
 
"""

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

    def forward(self, path_to_audio): # repair padding mask here
        audio, sr = torchaudio.load_with_torchcodec(path_to_audio)
        audio = torchaudio.functional.resample(audio, sr, 16000)
        h = self.encoder.extract_features(audio, padding_mask=torch.zeros_like(audio).bool())[0]
        h = self.proj(h)
        h = self.fsq.quantize(h)
        return self.fsq.codes_to_indices(h)

class Torca(nn.Module):
    #def __init__(self, chunk_duration, patch_size, time_step, num_mel_bins, num_classes, d_model, num_heads, d_ff, num_layers, dropout, fsq_levels, mask_prob, span_len, class_loss_weight, beats_ckpt):
    def __init__(self, cfg):
        super().__init__()
        self.grid_freq = int(cfg.num_mel_bins // cfg.patch_size)
        self.grid_time = int((cfg.chunk_duration * 1000) / cfg.time_step // cfg.patch_size)
        self.seq_len = self.grid_freq * self.grid_time
        self.class_loss_weight = cfg.class_loss_weight
        self.alibi_bias = get_alibi(cfg.num_heads, self.grid_freq, self.grid_time)
        self.quant = Quantizer(cfg.beats_ckpt, cfg.fsq_levels)
        self.emb = nn.Embedding(math.prod(cfg.fsq_levels), cfg.d_model)
        self.mask = make_mask(seq_len=self.seq_len, grid_freq=self.grid_freq, obj_masked=cfg.mask_prob, span=cfg.span_len)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.trans = clones(TransformerStack(d_model=cfg.d_model, number_heads=cfg.num_heads, d_ff=cfg.d_ff, bias=self.alibi_bias, dropout=cfg.dropout), cfg.num_layers)
        self.classif_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.masked_head = nn.Linear(cfg.d_model, math.prod(cfg.fsq_levels))

    
    def forward(self, x, padding_mask=None, labels=None):
        indices = self.quant(x) 
        h = self.emb(indices)
        if self.training:
            #batch_size = x.size(0)
            batch_size = 1
            tgt = indices.clone().long()
            masks = torch.stack([self.mask for _ in range(batch_size)])
            h[masks] = self.mask_token
        for layer in self.trans:
            h = layer(h)
        clas_logits = self.classif_head(h.mean(dim=1)) # for classification aggregate the prediction for all patches, worth trying mean pooling too
        if self.training:
            mask_logits = self.masked_head(h)
            print(f"mask_logits = {mask_logits.size()}\n")
            mask_loss = F.cross_entropy(mask_logits[masks], tgt[masks])
            clas_loss = F.cross_entropy(clas_logits, labels)
            return mask_loss + self.clas_loss_coef * clas_loss
        return clas_logits

cfgkw = TorcaConfig()
torkw = Torca(cfgkw)
torkw.eval()
over = '/Users/leo/projects/orcas/ds/3secs.wav'
test_logits = torkw(over)