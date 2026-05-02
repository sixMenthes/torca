from dataclasses import dataclass

@dataclass
class TorcaConfig:
    # data
    chunk_duration: float = 3.0
    #sample_rate: int = 16000
    patch_size: int = 16
    time_step: int = 10
    num_mel_bins: int = 128
    num_classes: int = 10

    # model
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    fsq_levels: list = (8, 6, 6)

    # masking
    mask_prob: float = 0.15
    span_len: int = 3

    # training
    #lr: float = 1e-4
    #batch_size: int = 32
    #num_epochs: int = 100
    class_loss_weight: float = 1.0

    # paths
    beats_ckpt: str = '../models/BEATs_iter3.pt' 