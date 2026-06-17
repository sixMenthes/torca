import numpy as np
import hydra
from omegaconf import DictConfig
import torch 
import torchvision
import torch.nn.functional as F
import torch_audiomentations
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.augmentations import  NoCallMixer
from birdset.datamodule.components.augmentations import PowerToDB

from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.transforms import Spectrogram, MelScale
from typing import List, Union

import numpy as np
from transformers import BatchFeature
from transformers import SequenceFeatureExtractor
from transformers.utils import logging, PaddingStrategy
import torch 
import random

from util.mixup import SpecMixup


logger = logging.get_logger(__name__)

class BaseTransform:
    def __init__(self, 
                 transform_params: DictConfig,         
                 target_length: int,
                 sampling_rate:int,
                 mean: float,
                 std: float,
                 columns: List[str],
                 clip_duration: float
        ):
        self.sampling_rate = sampling_rate  
        self.target_length = target_length 
        self.mean = mean
        self.std = std

        self.columns = columns
        self.clip_duration = clip_duration
        #self.fbank_params = transform_params.fbank
        self.input_params = transform_params.input
        self.transform_params = transform_params

        self.feature_extractor = DefaultFeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            return_attention_mask=False
        )    
    
        self.mixup_fn = None
        if self.transform_params.mixup.prob > 0:
            self.mixup_fn = SpecMixup(
                alpha=self.transform_params.mixup.alpha, 
                prob=self.transform_params.mixup.prob, 
                num_mix=self.transform_params.mixup.num_mix, 
                full_target=self.transform_params.mixup.full_target)
        
        self.freqm = None
        self.timem = None
        if self.transform_params.freqm:
            self.freqm = FrequencyMasking(freq_mask_param=self.transform_params.freqm)
        if self.transform_params.timem:
            self.timem = TimeMasking(time_mask_param=self.transform_params.timem)
        
        self.event_decoder = EventDecoding(min_len=5, max_len=5, sampling_rate=self.sampling_rate)

        if self.input_params.type == "birdset":
            self.spectrogram_conversion = Spectrogram(
                n_fft=self.input_params.n_fft, 
                hop_length=self.input_params.hop_length, 
                power=self.input_params.power)
            self.melscale_conversion = MelScale(
                n_mels=self.input_params.n_mels, 
                sample_rate=self.sampling_rate, 
                n_stft=self.input_params.n_stft)
            self.dbscale_conversion = PowerToDB()
    


    def _process_waveforms(self, waveforms):
        max_length = int(int(self.sampling_rate) * self.clip_duration)
        waveform_batch = self.feature_extractor(
            waveforms,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=False
        )
        # add std
        waveform_batch["input_values"] = waveform_batch["input_values"] - waveform_batch["input_values"].mean(axis=1, keepdims=True)
        #waveform_batch["input_values"] = (waveform_batch["input_values"] - waveform_batch["input_values"].mean(axis=1, keepdims=True)) / (waveform_batch["input_values"].std(axis=1, keepdims=True) + 1e-8)
        return waveform_batch 
    
    def _compute_fbank_features(self, waveforms):
        fbank_features = [
            fbank(
                waveform.unsqueeze(0),
                htk_compat=self.input_params.htk_compat,
                sample_frequency=self.sampling_rate,
                use_energy=self.input_params.use_energy,
                window_type=self.input_params.window_type,
                num_mel_bins=self.input_params.num_mel_bins,
                dither=self.input_params.dither,
                frame_shift=self.input_params.frame_shift
            )
            for waveform in waveforms
        ]
        return torch.stack(fbank_features)

    def _compute_birdset_features(self, waveforms):
        spectrograms = self.spectrogram_conversion(waveforms)
        spectrograms = self.melscale_conversion(spectrograms)
        fbank_features = self.dbscale_conversion(spectrograms).permute(0, 2, 1) # batch, 128, 501 --> batch, 501, 128
        return fbank_features


    def _pad_and_normalize(self, fbank_features):
        difference = self.target_length - fbank_features[0].shape[0]
        min_value = fbank_features.min()
        #min_value = -80
        if self.target_length > fbank_features.shape[0]:
            padding = (0, 0, 0, difference)
            fbank_features = F.pad(fbank_features, padding, value=min_value.item()) #no difference! 
            #m = torch.nn.ZeroPad2d((0, 0, 0, difference))
            #fbank_features = m(fbank_features)
            

        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        # fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1)
        # fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return fbank_features
    
    def __call__(self, batch):
        try:
            waveform_batch = [audio["array"] for audio in batch["audio"]]
        except:
            waveform_batch = self.event_decoder(batch)
            waveform_batch = [audio["array"] for audio in batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch)

        if self.input_params.type == "birdset":
            fbank_features = self._compute_birdset_features(waveform_batch["input_values"])
        
        elif self.input_params.type == "fbank":
            fbank_features = self._compute_fbank_features(waveform_batch["input_values"])     

        elif self.input_params.type == "audio":
            #audio = self._pad_and_normalize(waveform_batch["input_values"])
            # pad or truncate
            audio = waveform_batch["input_values"]
            return {
                "audio": audio,
                "label": torch.Tensor(batch[self.columns[1]])
            }
        fbank_features = self._pad_and_normalize(fbank_features)
        fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return {
            "audio": fbank_features.unsqueeze(1),
            "label": torch.Tensor(batch[self.columns[1]])
        }

class TrainTransform(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def cyclic_rolling_start(self, waveforms):
        batch_size, waveform_length = waveforms.shape
        idx = torch.randint(0, waveform_length, (batch_size,), device=waveforms.device)
        arange = torch.arange(waveform_length, device=waveforms.device).unsqueeze(0).expand(batch_size, -1)
        rolled_indices = (arange + idx.unsqueeze(1)) % waveform_length
        rolled_waveforms = waveforms[torch.arange(batch_size).unsqueeze(1), rolled_indices]
        volume_mag = torch.distributions.Beta(10, 10).sample((batch_size, 1)).to(waveforms.device) + 0.5
        waveforms = rolled_waveforms * volume_mag
        
        return waveforms  
    
    def __call__(self, batch):
        try:
            waveform_batch = [audio["array"] for audio in batch["audio"]]
        except:
            waveform_batch = self.event_decoder(batch)
            waveform_batch = [audio["array"] for audio in batch["audio"]]
        waveform_batch = self._process_waveforms(waveform_batch) # list of arrays with shape len wav --> stacked torch tensor batch x 320_000 torch float 32
        waveform_batch["input_values"] = self.cyclic_rolling_start(waveform_batch["input_values"]) # same shape as before

        # waveform mixup
        # if self.mixup: 
        #     waveform_batch["input_values"], batch[self.columns[1]] = self.mix_fn(waveform_batch["input_values"], batch[self.columns[1]])

        fbank_features = self._compute_fbank_features(waveform_batch["input_values"]) #shape now: batch, 998 height, 128 width

        # self.mixup_fn = SpecMixupN()
        if self.mixup_fn: #spec mxup
            if torch.rand(1) < 0.75:
                fbank_features, batch[self.columns[1]] = self.mixup_fn(fbank_features, batch[self.columns[1]]) # shape now: batch, 998, 128

        fbank_features = self._pad_and_normalize(fbank_features) # shape: batch, time(1024) padded, freq(128)
        
        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        if self.freqm: 
            fbank_features = fbank_features.permute(0, 2, 1).unsqueeze(1) # batch, 1, 128, 1024
            fbank_features = torch.stack([self.freqm(feature) for feature in fbank_features])
            fbank_features = torch.stack([self.timem(feature) for feature in fbank_features])
            #fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1) # time, freq
            fbank_features = fbank_features.squeeze(1)  # Remove the channel dimension
            fbank_features = fbank_features.permute(0, 2, 1)  # batch, 1, 1024, 128

        fbank_features = (fbank_features - self.mean) / (self.std * 2) # need: batch, 1024, 128

        return {
            "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
            "label": torch.Tensor(batch[self.columns[1]]),
        }
    
    def _mixup(self, waveform_batch, labels):
        mixed_audio = []
        mixed_labels = []
        batch_length = len(labels)
        for idx in range(batch_length):
            if random.random() < self.mixup:
                mix_sample_idx = random.randint(0, batch_length - 1)
                mix_lambda = np.random.beta(10, 10)
                mix_waveform = mix_lambda * waveform_batch["input_values"][idx] + (1 - mix_lambda) * waveform_batch["input_values"][mix_sample_idx]

                mix_waveform = mix_waveform - mix_waveform.mean()
                mixed_audio.append(mix_waveform)
                mixed_labels.append([mix_lambda * l1 + (1 - mix_lambda) * l2 for l1, l2 in zip(labels[idx], labels[mix_sample_idx])])
            else:
                mixed_audio.append(waveform_batch["input_values"][idx])
                mixed_labels.append(labels[idx])

        waveform_batch["input_values"] = torch.stack(mixed_audio)
        return torch.stack(mixed_audio), mixed_labels


class EvalTransform(BaseTransform):
    pass

class DefaultFeatureExtractor(SequenceFeatureExtractor):
    """
    A class used to extract features from audio data.

    Attributes
    ----------
    _target_ : str
        Specifies the feature extractor component used in the pipeline.
    feature_size : int
        Determines the size of the extracted features.
    sampling_rate : int
        The sampling rate at which the audio data should be processed.
    padding_value : float
        The value used for padding shorter sequences to a consistent length.
    return_attention_mask : bool
        Indicates whether an attention mask should be returned along with the processed features.
    """
    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 32000,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.return_attention_mask = return_attention_mask

    def __call__(
        self,
        waveform: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: int = None,
        truncation: bool = False,
        return_attention_mask: bool = False):
        #return_tensors: str = "pt"):

        waveform_encoded = BatchFeature({"input_values": waveform})

        padded_inputs = self.pad(
            waveform_encoded,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask
        )

        padded_inputs["input_values"] = torch.tensor(
            padded_inputs["input_values"])
        attention_mask = padded_inputs.get("attention_mask")

        if attention_mask is not None:
            padded_inputs["attention_mask"] = attention_mask


        return padded_inputs


class ImageTrainTransform(BaseTransform): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, batch):
        image_batch = [torch.tensor(np.array(images).T) for images in batch["input_values"]] # list of tensors with shape 998, 128
        image_batch = [self._pad_and_normalize(image) for image in image_batch] # list of tensors with shape 1024, 128
        image_batch = torch.stack(image_batch) # batch, 1024, 128
        image_batch = image_batch - image_batch.mean(axis=(1, 2), keepdims=True) # batch, 1024, 128
        fbank_features = self.cyclic_rolling_start_images(image_batch) # batch, 1024, 128

        if self.mixup_fn: #spec mxup
            fbank_features, batch["label"] = self.mixup_fn(fbank_features, batch["label"]) # shape now: batch, 1024, 128

         # shape: batch, time(1024), freq(128)
        
        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        if self.freqm: 
            fbank_features = fbank_features.permute(0, 2, 1).unsqueeze(1) # batch, 1, 128, 1024
            fbank_features = torch.stack([self.freqm(feature) for feature in fbank_features])
            fbank_features = torch.stack([self.timem(feature) for feature in fbank_features])
            #fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1) # time, freq
            fbank_features = fbank_features.squeeze(1)  # Remove the channel dimension
            fbank_features = fbank_features.permute(0, 2, 1)  # batch, 1, 1024, 128

        fbank_features = (fbank_features - self.mean) / (self.std * 2) # need: batch, 1024, 128

        return {
            "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
            "label": torch.Tensor(batch["label"]),
        }

    def cyclic_rolling_start_images(self, images):
        # Assuming images is of shape (batch_size, width, height)
        batch_size, width, height = images.shape
        idx = np.random.randint(0, width, size=batch_size)  # Random starting indices for each image in the batch
        
        # Create an array of indices for the width dimension
        rolled_indices = [(np.arange(width) + start_idx) % width for start_idx in idx]  # Roll indices for each image
        
        # Roll the images along the width dimension
        rolled_images = np.array([images[i, rolled_indices[i], :] for i in range(batch_size)])  # Shape: (batch_size, width, height)
        
        # Generate volume magnitude
        volume_mag = np.random.beta(10, 10, size=(batch_size, 1)) + 0.5  # Shape: (batch_size, 1)
        
        # Apply the volume magnitude to the rolled images
        images = rolled_images * volume_mag[:, None]  # Broadcasting volume magnitude across height
        
        return torch.tensor(images, dtype=torch.float32)
    
    def _mixup(self, waveform_batch, labels):
        mixed_audio = []
        mixed_labels = []
        batch_length = len(labels)
        for idx in range(batch_length):
            if random.random() < self.mixup:
                mix_sample_idx = random.randint(0, batch_length - 1)
                mix_lambda = np.random.beta(10, 10)
                mix_waveform = mix_lambda * waveform_batch["input_values"][idx] + (1 - mix_lambda) * waveform_batch["input_values"][mix_sample_idx]

                mix_waveform = mix_waveform - mix_waveform.mean()
                mixed_audio.append(mix_waveform)
                mixed_labels.append([mix_lambda * l1 + (1 - mix_lambda) * l2 for l1, l2 in zip(labels[idx], labels[mix_sample_idx])])
            else:
                mixed_audio.append(waveform_batch["input_values"][idx])
                mixed_labels.append(labels[idx])

        waveform_batch["input_values"] = torch.stack(mixed_audio)
        return torch.stack(mixed_audio), mixed_labels
    
    def _pad_and_normalize(self, fbank_features):
        difference = self.target_length - fbank_features.shape[0]
        if self.target_length > fbank_features.shape[0]:
            m = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank_features = m(fbank_features)

        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        # fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1)
        # fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return fbank_features
    
class ImageEvalTransform(BaseTransform): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, batch):
        image_batch = [torch.tensor(np.array(images).T) for images in batch["input_values"]] # list of tensors with shape 998, 128
        image_batch = [self._pad_and_normalize(image) for image in image_batch] # list of tensors with shape 1024, 128
        image_batch = torch.stack(image_batch) # batch, 1024, 128
        fbank_features = image_batch - image_batch.mean(axis=(1, 2), keepdims=True) # batch, 1024, 128
        fbank_features = (fbank_features - self.mean) / (self.std * 2) # need: batch, 1024, 128

        return {
            "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
            "label": torch.Tensor(batch["label"]),
        }

    def _pad_and_normalize(self, fbank_features):
        difference = self.target_length - fbank_features.shape[0]
        if self.target_length > fbank_features.shape[0]:
            m = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank_features = m(fbank_features)

        #fbank_features = fbank_features.transpose(0,1).unsqueeze(0)
        # fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1)
        # fbank_features = (fbank_features - self.mean) / (self.std * 2)
        return fbank_features


class BirdSetTrainTransform(TrainTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.no_call_mixer_params = self.transform_params.no_call_mixer
        self.event_decoder = EventDecoding(min_len=5, max_len=5, sampling_rate=self.sampling_rate)

        try:
            self.no_call_mixer = NoCallMixer(
                directory=self.no_call_mixer_params.directory,
                p=self.no_call_mixer_params.p,
                sampling_rate=self.no_call_mixer_params.sampling_rate,
                length=self.no_call_mixer_params.length
            )
        except:
            print("no no_call_mixer")
            self.no_call_mixer = None #!TODO FIX this!
        

        # waveform augmentations
        if self.transform_params.get('waveform_augmentations'):
            wave_augs = []
            for names, augs in self.transform_params.waveform_augmentations.items():
                wave_augs.append(hydra.utils.instantiate(augs))

            self.wave_aug = torch_audiomentations.Compose(wave_augs, output_type="object_dict")
        else:
            self.wave_aug = None
        
        if self.transform_params.get("spectrogram_augmentations"):
            spec_augs = []
            for names, augs in self.transform_params.spectrogram_augmentations.items():
                spec_augs.append(hydra.utils.instantiate(augs))

            self.spec_aug = torchvision.transforms.Compose(transforms=spec_augs)
        else:
            self.spec_aug = None

    def __call__(self, batch):
        try:
            waveform_batch = [audio["array"] for audio in batch["audio"]]
        except:
            waveform_batch = self.event_decoder(batch)
            waveform_batch = [audio["array"] for audio in batch["audio"]]

        waveform_batch = self._process_waveforms(waveform_batch)
        waveform_batch["input_values"] = self.cyclic_rolling_start(waveform_batch["input_values"])

        #waveform augmentations
        if self.wave_aug:
            if self.transform_params.get("pretrain") is None: #if fine-tuning
                output_dict = self.wave_aug(
                    waveform_batch["input_values"].unsqueeze(1), 
                    sample_rate=self.sampling_rate, 
                    targets=torch.Tensor(batch[self.columns[1]]).unsqueeze(1).unsqueeze(1))
                waveform_batch["input_values"] = output_dict["samples"].squeeze(1)
                batch[self.columns[1]] = output_dict["targets"].squeeze(1).squeeze(1)
            
            else: # if pre-training, no labels used
                output_dict = self.wave_aug(
                    waveform_batch["input_values"].unsqueeze(1), 
                    sample_rate=self.sampling_rate, 
                    targets=None)
                waveform_batch["input_values"] = output_dict["samples"].squeeze(1)

        if self.no_call_mixer:
            waveform_batch["input_values"], batch[self.columns[1]] = self.no_call_mixer(
                waveform_batch["input_values"], 
                batch[self.columns[1]])
            
        if self.input_params.type == "birdset":
            fbank_features = self._compute_birdset_features(waveform_batch["input_values"])
        
        elif self.input_params.type == "fbank":
            fbank_features = self._compute_fbank_features(waveform_batch["input_values"])

        elif self.input_params.type == "audio":
            audio = waveform_batch["input_values"]
            assert len(audio.shape) == 2, f"audios have shape {audio.shape} but expected to be of (batch, -1)"
            audio = F.pad(audio, (audio.size(0), self.target_length - audio.size(1)))
            return {
                "audio": audio,
                "label": torch.Tensor(batch[self.columns[1]])
            }
        else:
            raise ValueError("Invalid input type for BirdSetTrainTransform")

        if self.spec_aug: # normally in birdset on the spectrograms, not on the fbanks at the end.
            fbank_features = self.spec_aug(fbank_features.permute(0,2,1))
            fbank_features = fbank_features.permute(0,2,1)
     
        # self.mixup_fn = SpecMixupN(
        #     num_mix=2,
        #     min_snr_in_db=5.0,
        #     max_snr_in_db=25.0,
        #     full_target= True
        # )
        # if self.mixup_fn: #spec mxup
        #     if torch.rand(1) < 0.9:
        #         fbank_features, batch[self.columns[1]] = self.mixup_fn(fbank_features, batch[self.columns[1]]) 
        fbank_features = self._pad_and_normalize(fbank_features)

        if self.freqm: 
            fbank_features = fbank_features.permute(0, 2, 1).unsqueeze(1) # batch, 1, 128, 1024
            fbank_features = torch.stack([self.freqm(feature) for feature in fbank_features])
            fbank_features = torch.stack([self.timem(feature) for feature in fbank_features])
            #fbank_features = torch.transpose(fbank_features.squeeze(), 0, 1) # time, freq
            fbank_features = fbank_features.squeeze(1)  # Remove the channel dimension
            fbank_features = fbank_features.permute(0, 2, 1)  # batch, 1, 1024, 128

        fbank_features = (fbank_features - self.mean) / (self.std * 2) # need: batch, 1024, 128

        if self.transform_params.get("pretrain"):
            return {
                "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
            }
        else:
            return {
                "audio": fbank_features.unsqueeze(1), # batch, 1, 1024, 128
                "label": torch.Tensor(batch[self.columns[1]]),
        }
    
    