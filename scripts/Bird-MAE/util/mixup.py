import torch 
import numpy as np 

# class SpecMixup:
#     def __init__(self,
#                  alpha=1.0, 
#                  prob=1.0,
#                  num_mix=2
#                  ):
#         self.alpha = alpha
#         self.prob = prob
#         self.num_mix = num_mix

#     def __call__(self, x, target):
#         x, target = self._mix_batch(x, target)
#         return x, target
    
#     def _mix_batch(self, x, target):

#         batch_size = x.size(0)
#         # which samples to apply mixup to 
#         apply_mixup = np.random.rand(batch_size) < self.prob


#         # Initialize mixed inputs and targets
#         x_mix = x.clone()
#         target = torch.tensor(target, dtype=torch.float32)
#         target_mix = target.clone()

#         for i in range(batch_size):
#             if apply_mixup[i]:
#                 # Generate mixing coefficients from a Dirichlet distribution
#                 mix_weights = np.random.dirichlet([self.alpha] * self.num_mix)
#                 mix_weights = torch.from_numpy(mix_weights).float()  # Shape: (num_mix,)

#                 # Randomly select indices for mixing, including the current sample
#                 mix_indices = torch.randperm(batch_size)[:self.num_mix]
#                 # Ensure the current sample is included
#                 if i not in mix_indices:
#                     mix_indices[0] = i

#                 # Mix inputs
#                 x_mix_i = mix_weights[0] * x[mix_indices[0]]
#                 for j in range(1, self.num_mix):
#                     x_mix_i += mix_weights[j] * x[mix_indices[j]]
#                 x_mix[i] = x_mix_i

#                 # Mix targets
#                 target_mix_i = mix_weights[0] * target[mix_indices[0]]
#                 for j in range(1, self.num_mix):
#                     target_mix_i += mix_weights[j] * target[mix_indices[j]]
#                 target_mix[i] = target_mix_i
#             else:
#                 # If Mixup is not applied, keep the original target
#                 target_mix[i] = target[i]
#         return x_mix, target_mix.tolist()


class SpecMixup:
    def __init__(self, alpha=1.0, prob=1.0, num_mix=2, full_target=False):
        self.alpha = alpha
        self.prob = prob
        self.num_mix = num_mix
        self.full_target = full_target

    def __call__(self, x, target):
        return self._mix_batch(x, target)
    
    def _mix_batch(self, x, target):
        batch_size = x.size(0)
        device = x.device
        is_waveform = len(x.shape) == 2  # True for waveforms, False for spectrograms

        # Determine which samples to apply mixup to
        apply_mixup = torch.rand(batch_size, device=device) < self.prob

        if not apply_mixup.any():
            return x, target

        # Convert target to tensor if it's not already
        target = torch.tensor(target, dtype=torch.float32, device=device)

        # Generate mixing coefficients from a Dirichlet distribution
        mix_weights = torch.from_numpy(
            np.random.dirichlet([self.alpha] * self.num_mix, size=batch_size)
        ).float().to(device)

        # Generate random indices for mixing, excluding self-indices
        mix_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, self.num_mix)
        for i in range(batch_size):
            pool = torch.cat([torch.arange(0, i), torch.arange(i+1, batch_size)])
            mix_indices[i, 1:] = pool[torch.randperm(batch_size-1)[:self.num_mix-1]]
            #mix_indices[i, 1:] = 14

        # Perform mixup
        x_mix = torch.zeros_like(x)
        target_mix = torch.zeros_like(target)
        
        for i in range(self.num_mix):
            if is_waveform:
                x_mix += apply_mixup.unsqueeze(1) * mix_weights[:, i].unsqueeze(1) * x[mix_indices[:, i]]
            else:
                x_mix += apply_mixup.unsqueeze(1).unsqueeze(2) * mix_weights[:, i].unsqueeze(1).unsqueeze(2) * x[mix_indices[:, i]]
            
            if self.full_target:
                # For full_target, use hard labels
                target_mix = torch.max(target_mix, target[mix_indices[:, i]])
            else:
                # For soft labels, use weighted sum
                target_mix += apply_mixup.unsqueeze(1) * mix_weights[:, i].unsqueeze(1) * target[mix_indices[:, i]]

        # Only replace mixed samples
        if is_waveform:
            x = torch.where(apply_mixup.unsqueeze(1), x_mix, x)
        else:
            x = torch.where(apply_mixup.unsqueeze(1).unsqueeze(2), x_mix, x)
        target = torch.where(apply_mixup.unsqueeze(1), target_mix, target)

        return x, target.tolist()
    

class SpecMixupN:
    def __init__(self, alpha=1.0, prob=1.0, num_mix=1, full_target=False, min_snr_in_db=3.0, max_snr_in_db=30.0):
        self.alpha = alpha
        self.prob = prob
        self.num_mix = num_mix
        self.full_target = full_target
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db  

    def __call__(self, x, target):
        return self._mix_batch(x, target)
    
    def _mix_batch(self, x, target):
        batch_size = x.size(0)
        device = x.device
        is_waveform = len(x.shape) == 2  # True for waveforms, False for spectrograms

        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=x.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=x.device,
            ),
            validate_args=True,
        ) # sample uniformly from this distribution (low and high values)

        # randomize SNRs
        snr = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # Convert target to tensor if it's not already
        target = torch.tensor(target, dtype=torch.float32, device=device)

        # Generate mixing coefficients from a Dirichlet distribution
        mix_weights = torch.from_numpy(
            np.random.dirichlet([self.alpha] * self.num_mix, size=batch_size)
        ).float().to(device)

        # Generate random indices for mixing, excluding self-indices
        mix_indices = torch.empty((batch_size, self.num_mix), dtype=torch.long, device=device)
        for i in range(batch_size):
            # Create a pool of indices excluding the current index
            pool = torch.cat([torch.arange(0, i), torch.arange(i + 1, batch_size)])  # Exclude current index
            mix_indices[i] = pool[torch.randperm(batch_size - 1)[:self.num_mix]]  # Randomly select from the pool

        # Perform mixup
        #x_mix = torch.zeros_like(x)
        target_mix = target.clone()

        # Initialize x_mix with zeros or use torch.zeros_like(x) if you want to start from zero
        # x_mix = torch.zeros_like(x)  # Uncomment if you want to start from zero
        x_mix = x.clone()  # If you want to start with the original values

        batch_size, width, height = x_mix.shape

        for i in range(self.num_mix):
            if is_waveform:
                #x_mix += apply_mixup.unsqueeze(1) * mix_weights[:, i].unsqueeze(1) * x[mix_indices[:, i]]
                return
            else:
                current_indices = mix_indices[:, i]
                background_samples = rms_normalize_spectrogram(x[current_indices])

                idx = torch.randint(0, width, size=(batch_size,), device=device)  # Random starting indices for each image
                
                # Create an array of indices for the width dimension
                rolled_indices = [(torch.arange(width, device=device) + start_idx) % width for start_idx in idx]  # Roll indices for each image
                
                # Roll the images along the width dimension
                background_samples = torch.stack([background_samples[i, rolled_indices[i], :] for i in range(batch_size)])  # Shape: (batch_size, width, height)

                background_rms = calculate_rms_spectrogram(x) / (
                    10 ** (snr.unsqueeze(dim=-1) / 20)
                )
                background_rms = background_rms.unsqueeze(2).expand(-1, -1, background_samples.shape[1], -1)

                # Mix the background samples into x_mix
                x_mix += (background_rms * background_samples).sum(dim=0)
                target_mix = torch.max(target_mix, target[current_indices])
                #x_mix += apply_mixup.unsqueeze(1).unsqueeze(2) * mix_weights[:, i].unsqueeze(1).unsqueeze(2) * x[mix_indices[:, i]]

        return x_mix, target_mix.tolist()

def calculate_rms(tensor):
    return torch.sqrt(torch.mean(tensor ** 2, dim=-1, keepdim=True))

def calculate_rms_spectrogram(spectrogram):
    # Calculate RMS across the time dimension (width)
    return torch.sqrt(torch.mean(spectrogram ** 2, dim=1, keepdim=True)) 

def rms_normalize_spectrogram(samples):
    rms = samples.square().mean(dim=1, keepdim=True).sqrt()  # Shape: (batch_size, 1, height)
    return samples / (rms + 1e-8)  # Broadcasting will handle the dimensions