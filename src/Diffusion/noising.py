# Sacha Raffaud sachaRfd and acse-sr1022

"""

Code for the noise schedules:


Also includes plots of denoising of a molecule


"""

import os
import numpy as np
import torch

from data.Dataset_W93.dataset_class import W93_TS
import src.Diffusion.utils as diffusion_utils
from src.Diffusion.saving_sampling_functions import return_xyz, write_xyz_file


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    Clips the noise schedule given by alpha^2 to improve stability during sampling.     # noqa

    Parameters:
        alphas2 (numpy.ndarray): 1-dimensional array containing the values of alpha^2 for the noise schedule.
        clip_value (float, optional): The lower bound to which alpha_t / alpha_t-1 will be clipped.
                                      Defaults to 0.001.

    Returns:
        numpy.ndarray: A 1-dimensional array containing the clipped noise schedule, where alpha_t / alpha_t-1
                        is clipped to be greater than or equal to 'clip_value'.

    """
    alphas2 = np.concatenate(
        [np.ones(1), alphas2], axis=0
    )  # Add 1 to left of array # noqa

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    Generate a noise schedule based on a simple polynomial equation: 1 - (x / timesteps)^power. # noqa

    Parameters:
        timesteps (int): The total number of time steps for the noise schedule.
        s (float, optional): The minimum value added to the noise schedule. Defaults to 1e-4.
        power (float, optional): The power value in the polynomial equation. Defaults to 3.

    Returns:
        numpy.ndarray: A 1-dimensional array representing the noise schedule, containing alpha^2 values.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    Generate a cosine beta schedule as proposed in the paper:
    (https://openreview.net/forum?id=-NEXDKk8gZ)

    Parameters:
        timesteps (int): The total number of time steps for the beta schedule.
        s (float, optional): A scaling parameter to control the schedule's behavior. Defaults to 0.008. # noqa
        raise_to_power (float, optional): A power value to raise the beta values to. Defaults to 1.

    Returns:
        numpy.ndarray: A 1-dimensional array representing the cosine beta schedule.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def sigmoid_beta_schedule(timesteps, raise_to_power: float = 1, S: float = 6):
    """
    Generate a sigmoid beta schedule with optional power scaling.

    The sigmoid schedule transitions from low to high values based on the sigmoid function. # noqa

    Parameters:
        timesteps (int): The total number of time steps for the beta schedule.
        raise_to_power (float, optional): A power value to raise the beta values to. Defaults to 1.
        S (float, optional): Controls the steepness of the sigmoid function. Lower values make the function less steep,
                             and it slowly transitions from low to high values. Higher values make the function steeper,
                             and it quickly transitions from low to high values. Defaults to 6.

    Returns:
        numpy.ndarray: A 1-dimensional array representing the sigmoid beta schedule.

    Examples:
        - If you choose S = 1, you will be returned a linear noise schedule.
    """
    steps = timesteps + 2
    x = np.linspace(-S, S, steps)
    alphas2 = (1 / (1 + np.exp(-x))) ** 2
    alphas2 = (alphas2 - alphas2[0]) / (alphas2[-1] - alphas2[0])
    alphas2 = np.clip(alphas2, a_min=0, a_max=0.999)

    if raise_to_power != 1:
        alphas2 = np.power(alphas2, raise_to_power)

    return 1 - alphas2[1:]


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule.

    This class creates a lookup array for predefined (non-learned) noise schedules based on the input noise_schedule. # noqa
    The class supports three predefined noise schedules: 'cosine', 'polynomial_power', and 'sigmoid_steepness'.

    Parameters:
        noise_schedule (str): The predefined noise schedule to use. Possible values are: 'cosine',
                              'polynomial_<power>', or 'sigmoid_<steepness>'.
        timesteps (int): The total number of time steps for the noise schedule.
        precision (float): A parameter to control the noise schedule's behavior.
                           Used in the 'polynomial_power' and 'sigmoid_steepness' schedules.

    Raises:
        ValueError: If the noise_schedule provided is not one of the predefined options.

    Attributes:
        gamma (torch.nn.Parameter): The parameter storing the lookup array for the noise schedule.

    Note:
        The class does not use learned parameters; instead, it creates predefined noise schedules
        using the specified options.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
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

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False,
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


# Extra Functions for this files plotting purposes:
def inflate_batch_array(array, target):
    """
    Inflate the batch array to match the target shape.

    This function inflates the batch array 'array' with only a single axis (i.e., shape=(batch_size,), or possibly # noqa
    more empty axes (i.e., shape=(batch_size, 1, ..., 1)) to match the shape of the target tensor.

    Parameters:
        array (torch.Tensor): The input batch array with a single axis (shape=(batch_size,)).
        target (torch.Tensor): The target tensor with the desired shape.

    Returns:
        torch.Tensor: The inflated batch array with the same shape as the target tensor.
    """
    target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
    return array.view(target_shape)


def sigma(gamma, target_tensor):
    """
    Compute the sigma function given gamma.

    The sigma function is defined as the square root of the sigmoid of gamma.

    Parameters:
        gamma (torch.Tensor): Input tensor containing gamma values.
        target_tensor (torch.Tensor): The target tensor with the desired shape.

    Returns:
        torch.Tensor: The output tensor representing the computed sigma values with the same shape as the target tensor. # noqa
    """
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)


def alpha(gamma, target_tensor):
    """
    Compute the alpha function given gamma.

    The alpha function is defined as the square root of the sigmoid of negative gamma.  # noqa

    Parameters:
        gamma (torch.Tensor): Input tensor containing gamma values.
        target_tensor (torch.Tensor): The target tensor with the desired shape.

    Returns:
        torch.Tensor: The output tensor representing the computed alpha values with the same shape as the target tensor.
    """
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)  # noqa


if __name__ == "__main__":
    print("Running Script")

    # Setup the noising variables:
    noising = "sigmoid_5"
    timesteps = 1000
    precision = 1e-4

    # Setup the noise schedule:
    gamma = PredefinedNoiseSchedule(
        noise_schedule=noising, timesteps=timesteps, precision=precision
    )

    # Create an array between 0 and 10 to show how the noising goes:
    t_int = torch.linspace(0, timesteps, timesteps)

    # Normalise t_int bewtween 0 and 1:
    t = t_int / timesteps

    # Get a sample X to perform the noising on:
    dataset = W93_TS(directory="data/Dataset_W93/data/Clean_Geometries/")
    example_sample, node_mask = dataset[0]
    x = example_sample[:, -3:]
    ohe = example_sample[:, :4]

    # Repeat the same compound 10 times so that we can visualise the noising process:   # noqa
    x = x.repeat(timesteps, 1, 1)
    ohe = ohe.repeat(timesteps, 1, 1)

    # Get the inflated Gamma, Alpha and Sigma Values:
    gamma_t = inflate_batch_array(gamma(t), x)
    alpha_t = alpha(gamma_t, x)
    sigma_t = sigma(gamma_t, x)

    # Sample Random Noise for calculation:
    eps = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(
        size=(x.size(0), x.size(1), 3),
        device="cpu",
        node_mask=node_mask.unsqueeze(1).expand(x.size()),
    )

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
        sample_path = path_to_save + f"sample_{i}.xyz"

        # Convert to XYZ format:
        xyz_data = return_xyz(sample=[sample], dataset=dataset)

        # Write the file:
        write_xyz_file(xyz_data, sample_path)

    print("Finished Creating samples")
