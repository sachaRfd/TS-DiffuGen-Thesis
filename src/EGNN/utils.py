# Sacha Raffaud sachaRfd and acse-sr1022

import torch

"""# noqa
Code from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/fce07d701a2d2340f3522df588832c2c0f7e044a/equivariant_diffusion/utils.py
"""


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device is: ", device)
    else:
        device = torch.device("cpu")
        print("Device is: ", device)

    return device


def sum_except_batch(x):
    """# noqa
    Sums the elements of each tensor in the input `x`, except the batch dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the sum of elements in each tensor of `x`,
                      excluding the batch dimension.
    """
    # Remove batch from dim
    x = x.reshape(x.size(0), -1)
    # Sum over all molecules/atoms
    sum = x.sum(dim=-1)
    return sum


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"  # noqa
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    # print(rel_error)
    assert rel_error < 1e-5, f"Mean is not zero, relative_error {rel_error}"


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask)
    ).abs().max().item() < 1e-4, "Variables not masked properly."
