from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import soundfile as sf
import hydra
import torchvision
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB

# input: B, C, H, W


# ---------------------------------------------------------------------------
# Waveform-level augmentations. These operate on a (C, T) waveform *before*
# the spectrogram conversion, unlike specAugment which acts on the fbank.
# ---------------------------------------------------------------------------

class RandomShift:
    """Offset jitter: shift a (C, T) waveform in time by a random amount and
    zero-fill the exposed edge. Non-cyclic on purpose — a cyclic roll would wrap
    the tail of a call onto the front and corrupt its temporal structure."""
    def __init__(self, max_shift_samples, p=0.5):
        self.max_shift = int(max_shift_samples)
        self.p = p

    def __call__(self, wave):
        if self.max_shift > 0 and torch.rand(1).item() < self.p:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            t = wave.size(-1)
            if shift >= 0:                              # move later: pad front, drop tail
                wave = F.pad(wave, (shift, 0))[..., :t]
            else:                                       # move earlier: drop front, pad tail
                k = -shift
                wave = F.pad(wave, (0, k))[..., k:]
        return wave


class RandomGain:
    """Volume jitter: scale the waveform by a random gain drawn in dB."""
    def __init__(self, min_gain_db, max_gain_db, p=0.5):
        self.min_db = float(min_gain_db)
        self.max_db = float(max_gain_db)
        self.p = p

    def __call__(self, wave):
        if torch.rand(1).item() < self.p:
            gain_db = torch.empty(1).uniform_(self.min_db, self.max_db).item()
            wave = wave * (10.0 ** (gain_db / 20.0))
        return wave


class AddBackgroundNoise:
    """Mix in a random clip from a bank of background waveforms at a random SNR.

    The bank is a list of file paths (here: the DCLDE `Background`-labelled
    clips). Each call loads one clip, matches it to the signal length, and adds
    it scaled to hit an SNR sampled uniformly in [min_snr_db, max_snr_db].
    """
    def __init__(self, background_paths, target_length, sample_rate,
                 min_snr_db=5.0, max_snr_db=15.0, p=0.5):
        self.paths = list(background_paths) if background_paths else []
        self.target_length = int(target_length)
        self.sample_rate = int(sample_rate)
        self.min_snr_db = float(min_snr_db)
        self.max_snr_db = float(max_snr_db)
        self.p = p

    def _load(self, path):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        noise = torch.from_numpy(audio).T          # (C, T)
        noise = noise.mean(0, keepdim=True) if noise.size(0) > 1 else noise[:1]
        if sr != self.sample_rate:
            noise = AF.resample(noise, sr, self.sample_rate)
        return noise

    def _fit(self, noise):
        t = noise.size(-1)
        if t > self.target_length:
            start = int(torch.randint(0, t - self.target_length + 1, (1,)).item())
            noise = noise[..., start:start + self.target_length]
        elif t < self.target_length:
            noise = F.pad(noise, (0, self.target_length - t))
        return noise

    def __call__(self, wave):
        if not self.paths or torch.rand(1).item() >= self.p:
            return wave
        path = self.paths[int(torch.randint(0, len(self.paths), (1,)).item())]
        try:
            noise = self._fit(self._load(path)).to(wave.dtype)
        except Exception:
            return wave
        sig_power = wave.pow(2).mean()
        noise_power = noise.pow(2).mean()
        if sig_power <= 0 or noise_power <= 0:
            return wave
        snr = 10.0 ** (torch.empty(1).uniform_(self.min_snr_db, self.max_snr_db).item() / 10.0)
        scale = torch.sqrt(sig_power / (noise_power * snr))
        return wave + scale * noise


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


    def __call__(self, waveform):

        if waveform.size(0) > 1:
            idx = torch.argmax((waveform**2).mean(1))
        else:
            idx = 0

        waveform = waveform[idx].unsqueeze(0) # remove channels, 1D tensor

        waveform = self._process_waveform(waveform)
        fbank = self._compute_spectrogram_features(waveform)
        fbank = self._pad_and_normalize(fbank)

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
    def __init__(self, transform_params, background_paths=None):

        super().__init__(transform_params)

        if self.transform_params.get("spectrogram_augmentations"):
            spec_augs = []
            for names, augs in self.transform_params.spectrogram_augmentations.items():
                spec_augs.append(hydra.utils.instantiate(augs))

            self.spec_aug = torchvision.transforms.Compose(transforms=spec_augs)
        else:
            self.spec_aug = None

        self.wave_aug = self._build_wave_aug(background_paths)

    def _build_wave_aug(self, background_paths):
        cfg = self.transform_params.get("wave_augmentations")
        if not cfg:
            return None
        ops = []
        if cfg.get("shift"):
            s = cfg.shift
            ops.append(RandomShift(
                max_shift_samples=int(s.max_shift_seconds * self.sampling_rate),
                p=s.get("p", 0.5)))
        if cfg.get("gain"):
            g = cfg.gain
            ops.append(RandomGain(g.min_gain_db, g.max_gain_db, p=g.get("p", 0.5)))
        if cfg.get("background") and background_paths:
            b = cfg.background
            ops.append(AddBackgroundNoise(
                background_paths=background_paths,
                target_length=self.max_length,
                sample_rate=self.sampling_rate,
                min_snr_db=b.get("min_snr_db", 5.0),
                max_snr_db=b.get("max_snr_db", 15.0),
                p=b.get("p", 0.5)))
        return torchvision.transforms.Compose(ops) if ops else None

    def __call__(self, waveform):

        if waveform.size(0) > 1:
            idx = torch.argmax((waveform**2).mean(1))
        else:
            idx = 0

        waveform = waveform[idx].unsqueeze(0) # remove channels, 1D tensor

        waveform = self._process_waveform(waveform)
        if self.wave_aug:
            waveform = self.wave_aug(waveform)
        fbank = self._compute_spectrogram_features(waveform)
        fbank = self._pad_and_normalize(fbank)

        if self.spec_aug:
            fbank = self.spec_aug(fbank)


        return ((fbank - self.mean) / (self.std * 2)).permute(0, 2, 1)
    








    


