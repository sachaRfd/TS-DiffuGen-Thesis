"""

Code for the noise schedules:


Also includes plots of denoising of a molecule


"""

import os
import numpy as np
import torch
import math

from Dataset_W93.dataset_class import * 
import Diffusion.utils as diffusion_utils
from Diffusion.saving_sampling_functions import return_xyz, write_xyz_file





def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def sigmoid_beta_schedule(timesteps, raise_to_power: float = 1, S: float = 6):
    """
    Sigmoid schedule

    S controls the steepness of the sigmoid function:
    - Lower values make the function less steep, and it slowly transitions from low to high values.
    - Higher values make the function steeper, and it quickly transitions from low to high values.


    Examples:
    - If you choose S = 1 --> You will be returned a linear noise schedule
    """
    steps = timesteps + 2
    x = np.linspace(-S, S, steps)
    alphas2 = (1 / (1 + np.exp(-x)))**2
    alphas2 = (alphas2 - alphas2[0]) / (alphas2[-1] - alphas2[0])
    alphas2 = np.clip(alphas2, a_min=0, a_max=0.999)

    if raise_to_power != 1:
        alphas2 = np.power(alphas2, raise_to_power)

    return 1 - alphas2[1:]


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        elif "sigmoid" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            steepness = float(splits[1])
            alphas2 = sigmoid_beta_schedule(timesteps=timesteps, S=steepness)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        print("sigmas2", sigmas2)

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]
    


# Extra Functions for this files plotting purposes: 
def inflate_batch_array(array, target):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
    return array.view(target_shape)


def sigma(gamma, target_tensor):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

def alpha(gamma, target_tensor):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)



if __name__ == "__main__":
    print("Running Script")


    # Setup the noising variables: 
    noising = "sigmoid_5"
    timesteps = 1000
    precision= 1e-4


    # Setup the noise schedule: 
    gamma = PredefinedNoiseSchedule(noise_schedule=noising, timesteps=timesteps, precision=precision)

    # Create an array between 0 and 10 to show how the noising goes: 
    t_int = torch.linspace(0, timesteps, timesteps)

    # Normalise t_int bewtween 0 and 1: 
    t = t_int / timesteps


    # Get a sample X to perform the noising on:
    dataset = QM90_TS(directory="Dataset_W93/data/Clean_Geometries/")
    example_sample, node_mask = dataset[0]
    x = example_sample[:, -3:]
    ohe = example_sample[:, :4]

    # Repeat the same compound 10 times so that we can visualise the noising process: 
    x = x.repeat(timesteps, 1, 1)  
    ohe = ohe.repeat(timesteps, 1, 1)

    # Get the inflated Gamma, Alpha and Sigma Values:
    gamma_t = inflate_batch_array(gamma(t), x)
    alpha_t = alpha(gamma_t, x)
    sigma_t = sigma(gamma_t, x)


    # Sample Random Noise for calculation: 
    eps = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(size=(x.size(0), x.size(1), 3), device="cpu", node_mask=node_mask.unsqueeze(1).expand(x.size()))


    # Perform Equation Alpha * x + sigma * eps:
    noised_x = alpha_t * x + sigma_t * eps
    
    # Concatenate the h vector to X for saving purposes: 
    samples = torch.concatenate((ohe, noised_x), dim=2)


    # Save the samples: 
    path_to_save = f"plots_and_images/{noising}/"
    # Create the directory if it doesn't exist
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f"Directory '{path_to_save}' created.")
    else:
        print(f"Directory '{path_to_save}' already exists.")

    # Save alpha_t and sigma_t arrays in appropriate folders
    np.save(os.path.join(path_to_save, "alpha_t.npy"), alpha_t.cpu().numpy())
    np.save(os.path.join(path_to_save, "sigma_t.npy"), sigma_t.cpu().numpy())


    for i, sample in enumerate(samples):
        sample_path = path_to_save +  f"sample_{i}.xyz"
        
        # Convert to XYZ format: 
        xyz_data = return_xyz(sample=[sample], dataset=dataset)
 
        # Write the file: 
        write_xyz_file(xyz_data[0], sample_path)

    print("Finished Creating samples")