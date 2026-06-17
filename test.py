from torca_transforms import BaseTransform, TrainTransform
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from torchcodec.decoders import AudioDecoder

my_conf = OmegaConf.load("configs/data/transform/melbank_dclde.yaml")
wave = AudioDecoder("/Users/leo/projects/orcas/ds/try_birdmae.wav").get_all_samples().data
base = BaseTransform(my_conf)
train = TrainTransform(my_conf)
spec1 = base(wave).squeeze(0).permute(1,0)
spec2 = train(wave).squeeze(0).permute(1,0)

fig, ax = plt.subplots(2, 1)
ax[0].imshow(spec1, origin="lower", aspect="auto", interpolation="nearest")
ax[1].imshow(spec2, origin="lower", aspect="auto", interpolation="nearest")
plt.show()
