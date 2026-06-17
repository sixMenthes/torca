from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
import hydra
import torchvision
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB

# input: B, C, H, W
# cyclic rolling!!


class BaseTransform:
    def __init__(self,
                 transform_params:DictConfig):

        self.input_params = transform_params.input #spectrogram params
        self.transform_params = transform_params
        self.sampling_rate = self.input_params.sample_rate
        self.target_length = transform_params.target_length
        self.clip_duration = transform_params.clip_duration
        self.mean = self.input_params.mean
        self.std = self.input_params.std
        self.max_length = int(int(self.sampling_rate) * self.clip_duration)

        self.spectrogram_conversion = Spectrogram(
            n_fft=self.input_params.n_fft,
            hop_length=self.input_params.hop_length, 
            power=self.input_params.power)
        self.melscale_conversion = MelScale(
            n_mels=self.input_params.n_mels, 
            sample_rate=self.sampling_rate, 
            n_stft=self.input_params.n_stft)
        self.dbscale_conversion = AmplitudeToDB("power", 80.0)

        self.freqm = None
        self.timem = None

        if transform_params.freqm:
            self.freqm = FrequencyMasking(freq_mask_param=self.transform_params.freqm)
        if transform_params.timem:
            self.timem = TimeMasking(time_mask_param=self.transform_params.timem)

    def __call__(self, waveform):

        if waveform.size(0) > 1:
            idx = torch.argmax((waveform**2).mean(1))
        else:
            idx = 0

        waveform = waveform[idx].unsqueeze(0) # remove channels, 1D tensor

        waveform = self._process_waveform(waveform)
        fbank = self._compute_spectrogram_features(waveform)
        fbank = self._pad_and_normalize(fbank)
        if self.freqm:
            fbank = self.freqm(fbank)
        if self.timem:
            fbank = self.timem(fbank)

        return ((fbank - self.mean) / (self.std * 2)).permute(0, 2, 1)


    def _process_waveform(self, waveform, return_attention_mask=False):
        num_samples = waveform.size(0)
        if num_samples > self.max_length:
            new_waveform = waveform[:self.max_length]
            attention_mask = torch.ones_like(new_waveform)

        elif num_samples < self.max_length:
            padding = (0, self.max_length - waveform.size(-1))
            new_waveform = F.pad(waveform, padding, mode="constant", value=0)
            attention_mask = torch.zeros_like(new_waveform)
            attention_mask[:waveform.size(0)] = 1

        else:
            new_waveform = waveform
            attention_mask = torch.ones_like(new_waveform, dtype=torch.bool)

        if return_attention_mask:
            return new_waveform, attention_mask
        else:
            return new_waveform

    def _compute_spectrogram_features(self, waveform):
        spec = self.spectrogram_conversion(waveform)
        spec = self.melscale_conversion(spec)
        fbank_features = self.dbscale_conversion(spec)
        
        return fbank_features #H, W

    def _pad_and_normalize(self, fbank_features):
        length = fbank_features.size(-1) #Dims = B, H, W
        if self.target_length > length:
            difference = self.target_length - length
            min_value = fbank_features.min()
            padding = (0, difference)
            fbank_features = F.pad(fbank_features, padding, value=min_value.item())
        return fbank_features

class TrainTransform(BaseTransform):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.transform_params.get("spectrogram_augmentations"):
            spec_augs = []
            for names, augs in self.transform_params.spectrogram_augmentations.items():
                spec_augs.append(hydra.utils.instantiate(augs))

            self.spec_aug = torchvision.transforms.Compose(transforms=spec_augs)
        else:
            self.spec_aug = None

    def __call__(self, waveform):

        if waveform.size(0) > 1:
            idx = torch.argmax((waveform**2).mean(1))
        else:
            idx = 0

        waveform = waveform[idx].unsqueeze(0) # remove channels, 1D tensor

        waveform = self._process_waveform(waveform)
        fbank = self._compute_spectrogram_features(waveform)
        fbank = self._pad_and_normalize(fbank)
        if self.freqm:
            fbank = self.freqm(fbank)
        if self.timem:
            fbank = self.timem(fbank)

        if self.spec_aug:
            fbank = self.spec_aug(fbank)


        return ((fbank - self.mean) / (self.std * 2)).permute(0, 2, 1)
    








    


