import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import clones, make_mask, get_alibi
from fsq import FSQ
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

    def forward(self, soundwave, mask): # repair padding mask here
        h, m = self.encoder.extract_features(soundwave, padding_mask=mask)
        h = self.proj(h)
        h = self.fsq.quantize(h)
        return self.fsq.codes_to_indices(h), m

    def train(self, mode=True):
        #override train to stop layer behaviour on BEATs. Besides the weights that are frozen (requires_grad_ == False), dropout would've messed things up
        super().train(mode)
        self.encoder.eval()
        return self

class Torca(nn.Module):
    #def __init__(self, chunk_duration, patch_size, time_step, num_mel_bins, num_classes, d_model, num_heads, d_ff, num_layers, dropout, fsq_levels, mask_prob, span_len, class_loss_weight, beats_ckpt):
    def __init__(self, cfg, class_weights):
        super().__init__()
        self.grid_freq = int(cfg.num_mel_bins // cfg.patch_size)
        self.grid_time = int((cfg.chunk_duration * 1000) / cfg.time_step // cfg.patch_size)
        self.seq_len = self.grid_freq * self.grid_time
        self.mask_prob = cfg.mask_prob
        self.span_len = cfg.span_len
        self.register_buffer("alibi_bias", get_alibi(cfg.num_heads, self.grid_freq, self.grid_time)) #register buffer so that torch keeps track of the tensor
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.to(torch.float32))
        else:
            self.class_weights = None
        self.quant = Quantizer(cfg.beats_ckpt, cfg.fsq_levels)
        self.emb = nn.Embedding(math.prod(cfg.fsq_levels), cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        self.trans = clones(TransformerStack(d_model=cfg.d_model, number_heads=cfg.num_heads, d_ff=cfg.d_ff, bias=self.alibi_bias, dropout=cfg.dropout), cfg.num_layers)
        self.classif_head = nn.Linear(cfg.d_model, cfg.num_classes)
        self.masked_head = nn.Linear(cfg.d_model, math.prod(cfg.fsq_levels))

    
    def forward(self, x, labels=None, padding_mask=None):
        indices, patch_mask = self.quant(x, padding_mask) 
        print(f"Padding mask sanity: \n {patch_mask.float().mean()}")
        print(f"Unique indices: \n {indices.unique().numel()}")
        h = self.emb(indices.long())
        print(f"Pre-classifier features: \n {h.mean(dim=1).std(dim=0)}")
        batch_size = x.size(0)
        tgt = indices.clone().long()
        masks = torch.stack([make_mask(seq_len=self.seq_len, grid_freq=self.grid_freq, obj_masked=self.mask_prob, span=self.span_len)for _ in range(batch_size)]).to(h.device)
        h[masks] = self.mask_token
        for layer in self.trans:
            h = layer(h, padding_mask=patch_mask)
        clas_logits = self.classif_head(h.mean(dim=1)) # for classification aggregate the prediction for all patches
        
        print(f"Logits spread: \n {clas_logits.std(dim=0).mean()}")
        mask_logits = self.masked_head(h)
        mask_loss = F.cross_entropy(mask_logits[masks], tgt[masks])
        clas_loss = F.cross_entropy(clas_logits, labels, weight=self.class_weights)
        return mask_loss, clas_loss, clas_logits
