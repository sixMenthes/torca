"""Bird-MAE counterpart of test_beats.py: waveform -> tokens -> FSQ, at 3 seconds.

Run:  python test_birdmae.py [/path/to/Bird-MAE-B] [/path/to/clip.wav]

Where BEATs bundles fbank -> patch -> encoder inside `extract_features`, Bird-MAE
needs the three stages wired up explicitly; BirdMAEEncoder does that and exposes the
same `(B, N, 768)` interface. The point of this script is to confirm the checkpoint
really loads at target_length=304 — a 5 s (256-patch) checkpoint driving a 3 s
(152-patch) model — which it did NOT before the general branch was added to
VIT.load_pretrained_weights. Watch the [audiomae] lines: `pos_embed` SHOULD be
dropped (it is fixed sincos, regenerated at the new grid); anything else in the
missing list means the checkpoint didn't match and the backbone is partly random.
"""

import sys

import torch
import torchaudio

from models.components.fsq import FSQ
from models.components.birdmae_encoder import BirdMAEEncoder

# Bird-MAE-B; on the workstation this is /home/tundra/projects/torca/pretrained/Bird-MAE-B
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/Users/leo/projects/orcas/pretrained_models/Bird-MAE-B"
WAV = sys.argv[2] if len(sys.argv) > 2 else "/Users/leo/projects/orcas/ds/3secs.wav"

SAMPLE_RATE = 32000  # Bird-MAE pretrained on XCL @32 kHz (BEATs is 16 kHz)
TARGET_LENGTH = 304  # 3 s @ 10 ms hop = 300 frames, padded to a multiple of 16

encoder = BirdMAEEncoder(
    target_length=TARGET_LENGTH,
    sample_rate=SAMPLE_RATE,
    pretrained_weights_path=CKPT,
)
encoder.eval()
print(f"grid (time, freq) = {encoder.grid_size}, num_patches = {encoder.num_patches}")

try:
    sw, sr = torchaudio.load_with_torchcodec(WAV)
except Exception:
    # torchcodec is optional here; the SSL dataset reads with soundfile anyway.
    # Catch broadly on purpose: a torchcodec that is INSTALLED but cannot load its
    # shared library raises RuntimeError, not ImportError. That is the normal state
    # on ROCm — the published torchcodec wheels link CUDA, so loading dies on a
    # missing libnvrtc regardless of which FFmpeg is present.
    import soundfile as sf
    audio, sr = sf.read(WAV, dtype="float32", always_2d=True)
    sw = torch.from_numpy(audio).T
sw = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(sw)

with torch.no_grad():
    representation = encoder.tokens(sw)      # (B, 152, 768) — same contract as BEATs
print("tokens:", tuple(representation.shape))
print("pooled:", tuple(encoder.pooled(sw).shape))

# The tokenizer path: project to L dims, quantize on the fixed FSQ grid.
proj = torch.nn.Linear(768, 3)
fsq = FSQ([8, 6, 5])
codes = fsq.quantize(proj(representation))
indices = fsq.codes_to_indices(codes)
print("codes:", tuple(codes.shape), "indices:", tuple(indices.shape),
      "distinct:", int(indices.unique().numel()), "of", fsq.codebook_size)
