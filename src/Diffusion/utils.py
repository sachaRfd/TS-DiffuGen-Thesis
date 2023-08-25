import torch
import numpy as np
from src.EGNN.utils import remove_mean_with_mask


def assert_mean_zero(x):
    """# noqa
    Asserts that the mean of each row in the input tensor `x` is approximately zero.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features).

    Raises:
        AssertionError: If the absolute value of the maximum mean is >= 1e-6.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    mean = mean.abs().max().item()
    assert mean < 1e-5


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    """# noqa
    Generates a tensor following a Gaussian distribution, centered at zero,
    while considering a node mask.

    Args:
        size (tuple): Size of the tensor to generate (batch_size, num_nodes, features).
        device (torch.device): Device for tensor placement.
        node_mask (torch.Tensor): Node mask of shape (batch_size, num_nodes).

    Returns:
        x_projected (torch.Tensor): Generated tensor with masked center gravity.
    """
    assert len(size) == 3
    x = torch.randn(size, device=device)

    if len(node_mask.size()) == 2:
        node_mask = node_mask.unsqueeze(2).expand(size)

    x_masked = x * node_mask.to(device)

    x_projected = remove_mean_with_mask(x_masked, node_mask.to(device))
    return x_projected


def random_rotation(x, h):
    """# noqa


    Algorithm:
    1. Input sample with coordinates x and node features H
    2. Seperate coordinates of reactant, product and TS
    1. Sample random angle Theta
    2. Rotate Reactant, product and TS coordinates depending on rotation matrix and angle Theta:
    3. Return rotated coordinates and same node features

    Adapted from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/utils.py

    Apply random rotations to 3D coordinates and corresponding reactant and product information.

    This function performs random rotations on input 3D coordinates along with associated reactant and product
    information for data augmentation during training. It generates random rotation matrices around each axis,
    applies these rotations to the input data, and returns the rotated data.

    Args:
        x (torch.Tensor): Input tensor of 3D coordinates with shape (batch_size, num_nodes, 3).
        h (torch.Tensor): Input tensor containing additional information, including reactant and product details.

    Returns:
        torch.Tensor, torch.Tensor: Rotated 3D coordinates tensor and updated information tensor.
    """

    # assert that this is only possible with h vector of size 9 or 10:
    assert h.shape[2] == 9 or h.shape[2] == 10

    bs, _, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2

    # Seperate the Reactant/Product from h:
    product = h[:, :, -3:]
    reactant = h[:, :, -6:-3]

    assert n_dims == 3  # Make sure we have 3D Coordinates

    # Build Rx: Rotations in X dimension
    Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = (
        torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
    )  # Random theta angle to rotate
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Rx[:, 1:2, 1:2] = cos
    Rx[:, 1:2, 2:3] = sin
    Rx[:, 2:3, 1:2] = -sin
    Rx[:, 2:3, 2:3] = cos

    # Build Ry
    Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Ry[:, 0:1, 0:1] = cos
    Ry[:, 0:1, 2:3] = -sin
    Ry[:, 2:3, 0:1] = sin
    Ry[:, 2:3, 2:3] = cos

    # Build Rz
    Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Rz[:, 0:1, 0:1] = cos
    Rz[:, 0:1, 1:2] = sin
    Rz[:, 1:2, 0:1] = -sin
    Rz[:, 1:2, 1:2] = cos

    # Perform the rotation on the Transition state
    x = x.transpose(1, 2)
    # Perform the rotation in X direction
    x = torch.matmul(Rx, x)
    # Perform the rotation in X direction
    x = torch.matmul(Ry, x)
    # Perform the rotation in X direction
    x = torch.matmul(Rz, x)
    # Transpose back the to the original shape
    x = x.transpose(1, 2)

    # Perform the rotation on the Reactant state
    reactant = reactant.transpose(1, 2)

    # Perform the rotation in X direction
    reactant = torch.matmul(Rx, reactant)
    # Perform the rotation in X direction
    reactant = torch.matmul(Ry, reactant)
    # Perform the rotation in X direction
    reactant = torch.matmul(Rz, reactant)
    # Transpose back the to the original shape
    reactant = reactant.transpose(1, 2)

    # Perform the rotation on the Product state
    product = product.transpose(1, 2)
    # Perform the rotation in X direction
    product = torch.matmul(Rx, product)
    # Perform the rotation in X direction
    product = torch.matmul(Ry, product)
    # Perform the rotation in X direction
    product = torch.matmul(Rz, product)
    # Transpose back the to the original shape
    product = product.transpose(1, 2)

    # print(product)

    # Concatenate the reactant and product back together
    h_rot = h.clone()
    h_rot[:, :, -3:] = product
    h_rot[:, :, -6:-3] = reactant

    return x.contiguous(), h_rot.contiguous()


# if __name__ == "__main__":
# print("Running Tests")

# # Let's check that all the above functions work as intended here !!!
# dataset = W93_TS()
# train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# example_sample = next(iter(train_loader))

# # print(next(iter(train_loader)))

# # Now we can verify that each function works as intended:

# # Sum_except_batch:
# sum_batch = sum_except_batch(example_sample)
# # print(sum_batch)

# # Remove Mean:
# fake_batch_mean_removed = remove_mean_including_reactants_and_products(
#     example_sample
# )
# # print((fake_batch_mean_removed == example_sample).sum())
# assert_mean_zero(fake_batch_mean_removed[:, :, 4:7])

# # Remove mean of TS:
# TS_removed_mean = remove_mean_just_ts(example_sample[:, :, 10:13])
# # print(TS_removed_mean)

# # Assert mean is 0 --> it should already be the case:
# assert_mean_zero(example_sample[:, :, 4:7])

# # Sample random gaussian like the shape of TS:
# random_ts_noise = sample_center_gravity_zero_gaussian(
#     example_sample[:, :, 10:].shape, "cpu"
# )
# # print(random_ts_noise)
# assert_mean_zero(random_ts_noise)

# # Testing the other Log likelihood function:
# log_hood_2 = standard_gaussian_log_likelihood(example_sample[:, :, 10:])
# # print(log_hood_2)

# # Testing the Rajdom gaussian sampling that is not centred on 0:
# random_noise = sample_gaussian(example_sample[:, :, 10:13].shape, "cpu")
# # assert_mean_zero(random_noise)  # Should cause an error
