import torch
import torchaudio
from models.components.fsq import FSQ

from models.components.beats.beats import BEATs, BEATsConfig

# load the pre-trained checkpoints
checkpoint = torch.load("/Users/leo/projects/orcas/pretrained_models/BEATs_iter3.pt")

cfg = BEATsConfig(checkpoint["cfg"])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint["model"])
BEATs_model.eval()

# extract the the audio representation
sw, sr = torchaudio.load_with_torchcodec("/Users/leo/projects/orcas/ds/3secs.wav")
transform = torchaudio.transforms.Resample(sr, 16000)
sw = transform(sw)
mask = torch.zeros_like(sw)

representation = BEATs_model.extract_features(sw, padding_mask=mask)[0]
proj = torch.nn.Linear(768, 3)
fsq = FSQ([8, 6, 5])
