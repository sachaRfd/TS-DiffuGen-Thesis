# Sacha Raffaud sachaRfd and acse-sr1022

import torch
from torch import autograd  # noqa
from torch.nn import functional as F
import src.Diffusion.utils as diffusion_utils
import src.EGNN.utils as EGNN_utils
from tqdm import tqdm
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split

from data.Dataset_W93.dataset_class import W93_TS
from src.EGNN import dynamics
from src.EGNN.dynamics_with_graph import (
    EGNN_dynamics_graph,
)
from src.Diffusion.noising import PredefinedNoiseSchedule

from src.Classifier.classifier import EGNN_graph_prediction

from torch.utils.data import DataLoader
from torch import expm1
from torch.nn.functional import softplus


"""

This script contains the classes for 2 diffusion models. 
The first class repreents the simple diffusion model that does not use 
reaction graphs during training and inference.

The second inherits from the first but includes slight changes that 
allow it to use edge attributes during training.

The diffusion backbone was adapted from the following repo:
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py

The adaptations include cleaning the whole script, removing functions
and methods and making everything a bit more modular. Noising was also 
adapted to not be performed with H node features, as we are only trying to
predict X part of Xh.


This file also includes get_node_features function which controls how many node features are included
in the different diffusion processes.


"""  # noqa


class DiffusionModel(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
        self,
        dynamics: dynamics.EGNN_dynamics_QM9,
        in_node_nf: int,
        n_dims: int = 3,
        device: str = "cpu",
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        noise_precision: float = 1e-4,
        use_mse=True,
    ):
        super().__init__()

        # Setup CPU-GPU:
        self.device = device

        # MSE is used as the loss function from paper:
        self.use_mse = use_mse
        self.mse = MSELoss()

        # Setup the noise schedule and get Gamma Values:
        self.gamma = PredefinedNoiseSchedule(
            noise_schedule, timesteps=timesteps, precision=noise_precision
        )

        # Denoising EGNN Network:
        self.dynamics = dynamics

        # Setup variables:
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims

        # Number of Sampling Timesteps:
        self.T = timesteps

        # Not sure what is the use.
        self.register_buffer("buffer", torch.zeros(1))

        # Check that the noise schedule allows for the total noising process
        #  to reach Gaussian Noise:
        self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        """
        Checks that the number of timesteps is large enough
        to converge to white noise.
        """
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        if sigma_0 * num_stdevs > 1.0:
            raise ValueError("Please use larger amount of Timesteps")

    def phi(self, x, t, node_mask, edge_mask):
        """
        Function to get Predicted noise from denoising model.
        """
        # Predicted H is not needed in conformation generation
        _, net_out = self.dynamics._forward(
            t,
            x,
            node_mask,
            edge_mask,
        )
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1))
        to match the target shape.
        """  # noqa
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """
        Computes sigma given gamma.

        The sigma variable controls how much noise is
        added at a given timestep.
        """
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(gamma)),
            target_tensor,
        )

    def alpha(self, gamma, target_tensor):
        """
        Computes alpha given gamma.

        The alpha variable controls how much of the
        original sample is kept at a given
        timestep.
        """
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(
        self,
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor,
        target_tensor: torch.Tensor,
    ):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2).
        """  # noqa
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s,
            target_tensor,
        )

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def compute_x_pred(self, net_out, zt, gamma_t):
        """
        Commputes x_pred, i.e. the most likely prediction of x.
        """
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = (
            1.0 / alpha_t * (zt - sigma_t * eps_t)
        )  # This is from the prediction formula fron noise to prediction value
        return x_pred

    def compute_inter_atomic_distance(self, coordinates):
        """
        Calculate pairwise distance matrix from 3D coordinates.

        Args:
            coordinates (torch.Tensor): Batch of 3D coordinates with shape (bs, num_atoms, 3).

        Returns:
            torch.Tensor: Pairwise distance matrix with shape (bs, num_atoms, num_atoms).
        """  # noqa

        # Expand dimensions to allow broadcasting
        expanded_coords = coordinates.unsqueeze(
            2
        )  # Shape: (bs, num_atoms, 1, 3)  # noqa
        expanded_coords_transpose = coordinates.unsqueeze(
            1
        )  # Shape: (bs, 1, num_atoms, 3)

        # Compute the pairwise distance matrix
        diff = (
            expanded_coords - expanded_coords_transpose
        )  # Shape: (bs, num_atoms, num_atoms, 3)
        squared_distances = torch.sum(
            diff**2, dim=-1
        )  # Shape: (bs, num_atoms, num_atoms)
        distance_matrix = torch.sqrt(squared_distances + 1e-8)

        return distance_matrix

    def compute_error(self, net_out, eps, use_mse=True):
        """
        Computes the error (MSE) between the true and predicted noise, otherwise
        use the Inter-atomic distance norm
        """  # noqa

        if use_mse:
            error = self.mse(eps, net_out)
            return error
        else:
            # Use both in loss function:
            error = self.mse(eps, net_out)
            # Use Inter-atomic-distance matrix:
            eps_iad = self.compute_inter_atomic_distance(eps)
            net_out_iad = self.compute_inter_atomic_distance(net_out)

            # Get number of atoms:
            bs, N_atom, _ = eps.shape

            error_2 = torch.tensor(0.0, device=eps.device)

            for i in range(bs):
                # Divide by 2 as we are dealing with a symmetric matrix:
                dmae_sum = torch.sum(torch.abs(eps_iad[i] - net_out_iad[i])) / 2  # noqa

                # Calculate the error for this sample and add it to the list:
                error_2 += 2.0 / (N_atom * (N_atom - 1)) * dmae_sum

            # Divide by Bath_size:
            error_2 /= bs

            error = error.clone().requires_grad_(True)

            return error + error_2

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, guided=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        net_out = self.phi(
            t=zeros,
            x=z0,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(
            net_out=net_out.to(self.device),
            zt=z0[:, :, -3:].to(self.device),
            gamma_t=gamma_0.to(self.device),
        )

        if guided:
            x = self.sample_normal(
                mu=mu_x.to(self.device),
                sigma=sigma_x.to(self.device),
                node_mask=node_mask.unsqueeze(2)
                .view(z0.shape[0], -1, 1)
                .expand(mu_x[:, :, -3:].size()),
            )
        else:
            x = self.sample_normal(
                mu=mu_x.to(self.device),
                sigma=sigma_x.to(self.device),
                node_mask=node_mask.to(self.device),
            )

        return x

    def sample_normal(self, mu, sigma, node_mask, guidance=False):
        """
        Samples from a Normal distribution and returns the noisy X_t sample.
        """
        bs = mu.size(0)
        eps = self.sample_position(bs, mu.size(1), node_mask)

        if guidance:
            return mu + sigma * eps, eps
        else:
            return mu + sigma * eps

    def sample_position(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x.
        """
        z_x = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=self.device,
            node_mask=node_mask,
        )
        return z_x

    def compute_loss(self, x, h, node_mask, edge_mask):
        """Computes the loss using simple MSE loss."""

        lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device
        ).float()

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        # gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_position(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t.to(self.device) * x.to(self.device) + sigma_t.to(
            self.device
        ) * eps.to(self.device)

        EGNN_utils.assert_mean_zero_with_mask(
            z_t, node_mask.unsqueeze(2).expand(z_t.size()).to(self.device)
        )

        # Now Concatenate x with H before feeding it into the model:
        z_t = torch.cat([h.to(self.device), z_t.to(self.device)], dim=2)

        # Neural net prediction.
        net_out = self.phi(
            z_t.to(self.device),
            t.to(self.device),
            node_mask.to(self.device),
            edge_mask.to(self.device),
        )

        # Compute the error.
        error = self.compute_error(net_out, eps, use_mse=self.use_mse)

        return error

    def forward(self, x, h, node_mask=None, edge_mask=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """  # noqa
        loss = self.compute_loss(x, h, node_mask, edge_mask)

        neg_log_pxh = loss

        # Correct for normalization on x.
        neg_log_pxh = neg_log_pxh.to(self.device)

        return neg_log_pxh

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask)

        # Compute mu for p(zs | zt).
        EGNN_utils.assert_mean_zero_with_mask(
            zt[:, :, -3:], node_mask.unsqueeze(2).expand(zt[:, :, -3:].size())
        )
        EGNN_utils.assert_mean_zero_with_mask(
            eps_t[:, :, -3:],
            node_mask.unsqueeze(2).expand(eps_t[:, :, -3:].size()),  # noqa
        )

        mu = zt[:, :, -3:].to(self.device) / alpha_t_given_s.to(self.device) - (  # noqa
            sigma2_t_given_s.to(self.device)
            / alpha_t_given_s.to(self.device)
            / sigma_t.to(self.device)
        ) * eps_t.to(self.device)

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(
            mu.to(self.device),
            sigma.to(self.device),
            node_mask.to(self.device),
        )

        # Project down to avoid numerical runaway of the center of gravity.
        # Have to concatenate the OLD H:
        # first remove mean:
        zs = torch.cat(
            [
                zt[:, :, :-3].to(self.device),
                EGNN_utils.remove_mean_with_mask(
                    zs[:, :, -3:],
                    node_mask.unsqueeze(2).expand(zs[:, :, -3:].size()),  # noqa
                ).to(self.device),
            ],
            dim=2,
        )
        return zs

    @torch.no_grad()
    def sample(
        self,
        h,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context_size=0,
    ):
        """
        Draw samples from the generative model.
        """

        z = self.sample_position(n_samples, n_nodes, node_mask)

        # Concatenate the predefined H:
        z = torch.cat([h.to(self.device), z.to(self.device)], dim=2)

        # Check the coordinates have zero mean
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, -3:], node_mask.unsqueeze(2).expand(z[:, :, -3:].size())
        )

        # Assert that the product is centred
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, 7 + context_size : 10 + context_size],  # noqa
            node_mask.unsqueeze(2).expand(
                z[:, :, 7 + context_size : 10 + context_size].size()  # noqa
            ),
        )

        # Assert that the reactant is centred
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, 4 + context_size : 7 + context_size],  # noqa
            node_mask.unsqueeze(2).expand(
                z[:, :, 4 + context_size : 7 + context_size].size()  # noqa
            ),
        )

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array.to(self.device),
                t_array.to(self.device),
                z.to(self.device),
                node_mask.to(self.device),
                edge_mask.to(self.device),
            )

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(
            z,
            node_mask.to(self.device),
            edge_mask.to(self.device),
        )

        EGNN_utils.assert_mean_zero_with_mask(
            x, node_mask.unsqueeze(2).expand(x.size())
        )

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = diffusion_utils.remove_mean_with_mask(
                x, node_mask.unsqueeze(2).expand(x.size())
            )

        return x

    def sample_p_zs_given_zt_guidance(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        # target_function,
        # scale=0.9,
    ):
        """
        Adapted from:
        https://gitlab.com/porannegroup/gaudi/

        Samples from zs ~ p(zs | zt), with
        extra conditioning on y.
        Only used during sampling.
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        with torch.no_grad():
            eps_t = self.phi(zt, t, node_mask, edge_mask)

        # Calculate the predicted coordinates of zs:
        x_pred_s = self.compute_x_pred(
            net_out=eps_t,
            zt=zt[:, :, -3:],
            gamma_t=gamma_t,
        )

        # Compute mu for p(zs | zt).
        mu = zt[:, :, -3:].to(self.device) / alpha_t_given_s.to(self.device) - (  # noqa
            sigma2_t_given_s.to(self.device)
            / alpha_t_given_s.to(self.device)
            / sigma_t.to(self.device)
        ) * eps_t.to(self.device)

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt:
        # Also returns the noise that was added for future use
        zs, eps_used = self.sample_normal(
            mu,
            sigma,
            node_mask.unsqueeze(2)
            .view(zt.shape[0], -1, 1)
            .expand(zt[:, :, -3:].size()),
            guidance=True,
        )

        # Concatenate the reactant and product to the
        # x prediction:
        x_pred_s = torch.cat(
            [zt[:, :, :-3].to(self.device), x_pred_s[:, :, -3:]],
            dim=2,
        )

        # # guidance:
        # with torch.enable_grad():
        #     x_pred_s = x_pred_s.requires_grad_()
        #     energy = (
        #         scale
        #         * target_function(
        #             x_pred_s,
        #             node_mask,
        #             edge_mask,
        #             # t,
        #         ).sum()
        #     )
        #     grad = autograd.grad(energy, x_pred_s)[0]

        # max_norm = 10
        # grad_norm = grad.norm(dim=[1, 2])
        # clip_coef = max_norm / (grad_norm + 1e-6)
        # clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        # grad *= clip_coef_clamped[:, None, None]

        # # grad = torch.cat(
        # #     [
        # #         # grad[:, :, :-3],
        # #         diffusion_utils.remove_mean_with_mask(
        # #             grad[:, :, -3:],
        # #             node_mask,
        # #         ),
        # #     ],
        # #     dim=2,
        # # )
        # grad = diffusion_utils.remove_mean_with_mask(
        #     grad[:, :, -3:],
        #     node_mask,
        # )

        # # Sigma Reduces as we get better and better samples:
        # zs = zs - sigma * grad

        # # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                zt[:, :, :-3].to(self.device),
                EGNN_utils.remove_mean_with_mask(
                    zs[:, :, -3:],
                    node_mask.unsqueeze(2)
                    .view(zt.shape[0], -1, 1)
                    .expand(zt[:, :, -3:].size()),  # noqa
                ).to(self.device),
            ],
            dim=2,
        )
        return zs, x_pred_s

    def guided_sampling(
        self,
        h,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        edges,
        classifier_model=EGNN_graph_prediction,
    ):
        """
        Draw samples from the generative model, and guide
        them using the pre-trained classifier model.
        """

        # Generate random noise:
        z = self.sample_position(n_samples, n_nodes, node_mask)

        # Concatenate the predefined reactant and product:
        z = torch.cat([h.to(self.device), z.to(self.device)], dim=2)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        preds = []
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z, current_prediction_x = self.sample_p_zs_given_zt_guidance(
                s_array.to(self.device),
                t_array.to(self.device),
                z.to(self.device),
                node_mask.to(self.device),
                edge_mask.to(self.device),
            )

            # Calculate the proba that D-MAE is below 0.1:
            # current_prediction = current_prediction_x.view(-1, 13)
            # x_ = current_prediction_x[:, :, -3:].view(-1, 3)
            # node_mask = node_mask.view(-1, 1)

            # edge_mask = edge_mask.view(n_samples * 23 * 23, 1)

            pred_label = classifier_model(
                h=current_prediction_x.view(-1, 13),
                x=current_prediction_x[:, :, -3:].view(-1, 3),
                edge_index=edges,
                node_mask=node_mask.view(-1, 1),
                edge_mask=edge_mask.view(n_samples * 23 * 23, 1),
            )
            pred_label = torch.argmax(pred_label, 1)
            if s % 100 == 0:
                print(f"Timestep:\t{s}\t {pred_label}")
                preds.append(current_prediction_x)

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(
            z,
            node_mask.to(self.device),
            edge_mask.to(self.device),
            guided=True,
        )

        return x, preds

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = diffusion_utils.remove_mean_with_mask(
                x, node_mask.unsqueeze(2).expand(x.size())
            )

        return x

    @torch.no_grad()
    def sample_chain(
        self,
        h,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        keep_frames=None,
        context_size=0,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """  # noqa

        predefined_h = h.to(self.device)

        z = self.sample_position(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero(z[:, :, -3:])  # noqa

        # Concatenate predefined H:
        z = torch.cat([predefined_h, z], dim=2)

        # Check the coordinates have zero mean
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, -3:], node_mask.unsqueeze(2).expand(z[:, :, -3:].size())
        )

        # Assert that the product is centred
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, 7 + context_size : 10 + context_size],  # noqa
            node_mask.unsqueeze(2).expand(
                z[:, :, 7 + context_size : 10 + context_size].size()  # noqa
            ),
        )

        # Assert that the reactant is centred
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, 4 + context_size : 7 + context_size],  # noqa
            node_mask.unsqueeze(2).expand(
                z[:, :, 4 + context_size : 7 + context_size].size()  # noqa
            ),
        )

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array.to(self.device),
                t_array.to(self.device),
                z.to(self.device),
                node_mask.to(self.device),
                edge_mask.to(self.device),
            )

            diffusion_utils.assert_mean_zero(z[:, :, -3:])

            # Write to chain tensor
            write_index = (s * keep_frames) // self.T
            chain[write_index] = z

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(
            z, node_mask.to(self.device), edge_mask.to(self.device)
        )

        diffusion_utils.assert_mean_zero(x[:, :, -3:])

        xh = torch.cat([h, x], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat


# Class that inherits the above but
#  with slight changes for reaction graph input:
class DiffusionModel_graph(DiffusionModel):
    def __init__(
        self,
        dynamics: EGNN_dynamics_graph,
        in_node_nf: int,
        n_dims: int = 3,
        device: str = "cpu",
        timesteps: int = 1000,
        noise_schedule: str = "cosine",
        noise_precision: float = 0.0001,
    ):
        super().__init__(
            dynamics,
            in_node_nf,
            n_dims,
            device,
            timesteps,
            noise_schedule,
            noise_precision,
        )

    def phi(self, x, t, node_mask, edge_mask, edge_attributes):
        # Predicting updated node features is not important.
        _, net_out = self.dynamics._forward(
            t,
            x,
            node_mask,
            edge_mask,
            edge_attributes=edge_attributes,
        )
        return net_out

    def sample_p_xh_given_z0(
        self,
        z0,
        node_mask,
        edge_mask,
        edge_attributes=None,
    ):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        net_out = self.phi(
            t=zeros,
            x=z0,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attributes=edge_attributes,
        )

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(
            net_out=net_out.to(self.device),
            zt=z0[:, :, -3:].to(self.device),
            gamma_t=gamma_0.to(self.device),
        )

        x = self.sample_normal(
            mu=mu_x.to(self.device),
            sigma=sigma_x.to(self.device),
            node_mask=node_mask.to(self.device),
        )

        return x

    def compute_loss(
        self,
        x,
        h,
        node_mask,
        edge_mask,
        edge_attributes,
    ):
        """Computes the the simple loss (MSE)."""
        lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device
        ).float()

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        # gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_position(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t.to(self.device) * x.to(self.device) + sigma_t.to(
            self.device
        ) * eps.to(self.device)

        EGNN_utils.assert_mean_zero_with_mask(
            z_t, node_mask.unsqueeze(2).expand(z_t.size()).to(self.device)
        )

        # Now Concatenate x with H before feeding it into the model:
        z_t = torch.cat([h.to(self.device), z_t.to(self.device)], dim=2)

        # Neural net prediction.
        net_out = self.phi(
            z_t.to(self.device),
            t.to(self.device),
            node_mask.to(self.device),
            edge_mask.to(self.device),
            edge_attributes,
        )

        # Compute the error.
        error = self.compute_error(net_out, eps)

        return error

    def forward(
        self,
        x,
        h,
        node_mask=None,
        edge_mask=None,
        edge_attributes=None,  # Include Edge Attributes
    ):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.           # noqa
        """
        # Normalize data, take into account volume change in x.

        loss = self.compute_loss(
            x,
            h,
            node_mask,
            edge_mask,
            edge_attributes=edge_attributes,
        )

        neg_log_pxh = loss

        # Correct for normalization on x.
        neg_log_pxh = neg_log_pxh.to(self.device)
        return neg_log_pxh

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        edge_attributes=None,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(
            zt,
            t,
            node_mask,
            edge_mask,
            edge_attributes,
        )

        # Compute mu for p(zs | zt).
        EGNN_utils.assert_mean_zero_with_mask(
            zt[:, :, -3:], node_mask.unsqueeze(2).expand(zt[:, :, -3:].size())
        )
        EGNN_utils.assert_mean_zero_with_mask(
            eps_t[:, :, -3:],
            node_mask.unsqueeze(2).expand(eps_t[:, :, -3:].size()),  # noqa
        )

        mu = zt[:, :, -3:].to(self.device) / alpha_t_given_s.to(self.device) - (  # noqa
            sigma2_t_given_s.to(self.device)
            / alpha_t_given_s.to(self.device)
            / sigma_t.to(self.device)
        ) * eps_t.to(self.device)

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(
            mu.to(self.device),
            sigma.to(self.device),
            node_mask.to(self.device),
        )

        # Project down to avoid numerical runaway of the center of gravity.
        # Have to concatenate the OLD H:
        # first remove mean:
        zs = torch.cat(
            [
                zt[:, :, :-3].to(self.device),
                EGNN_utils.remove_mean_with_mask(
                    zs[:, :, -3:],
                    node_mask.unsqueeze(2).expand(zs[:, :, -3:].size()),  # noqa
                ).to(self.device),
            ],
            dim=2,
        )
        return zs

    @torch.no_grad()
    def sample(
        self,
        h,
        edge_attributes,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
    ):
        """
        Draw samples from the generative model.
        """
        # Predefined_h compared to theirs
        predefined_h = h

        z = self.sample_position(n_samples, n_nodes, node_mask)

        # Concatenate the predefined H:
        z = torch.cat([predefined_h.to(self.device), z.to(self.device)], dim=2)

        # Check the coordinates have zero mean
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, -3:], node_mask.unsqueeze(2).expand(z[:, :, -3:].size())
        )
        # Currently are not using Reactant or product
        EGNN_utils.assert_mean_zero_with_mask(
            z[:, :, 4:7],  # noqa
            node_mask.unsqueeze(2).expand(z[:, :, 4:7].size()),  # noqa
        )

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array.to(self.device),
                t_array.to(self.device),
                z.to(self.device),
                node_mask.to(self.device),
                edge_mask.to(self.device),
                edge_attributes,
            )

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(
            z,
            node_mask.to(self.device),
            edge_mask.to(self.device),
            edge_attributes,
        )

        EGNN_utils.assert_mean_zero_with_mask(
            x, node_mask.unsqueeze(2).expand(x.size())
        )

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = EGNN_utils.remove_mean_with_mask(
                x, node_mask.unsqueeze(2).expand(x.size())
            )  # noqa
        return x


def get_node_features(
    remove_hydrogens=False,
    include_context=False,
    no_product=False,
):
    """
    Determine the number of node features based on input options.

    This function calculates the number of node features for a given molecular node based on
    the input options provided. The node features are determined by the presence or absence
    of hydrogen atoms, inclusion of context information, and the no_product option.

    Args:
        remove_hydrogens (bool, optional): If True, hydrogens are removed, and the resulting
            node features will not include a 4D one-hot encoding for hydrogen positions.
        include_context (bool, optional): If True, an additional dimension for context information
            is added to the node features.
        no_product (bool, optional): If True, a few dimensions are excluded from the node features
            to account for the no_product option.

    Returns:
        int: The calculated number of node features based on the provided options.
    """  # noqa
    # Time embedding is already included
    if remove_hydrogens:
        in_node_nf = 10
    else:
        in_node_nf = 11

    if include_context:
        in_node_nf += 1

    if no_product:
        in_node_nf -= 3
    return in_node_nf


if __name__ == "__main__":
    print("running script")

    # Setup the device:
    device = dynamics.setup_device()

    remove_hydrogens = False
    include_context = False

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens, include_context=include_context
    )

    noise_schedule = "sigmoid_2"
    timesteps = 2_000
    batch_size = 64
    n_layers = 4
    hidden_features = 64
    lr = 4e-4
    epochs = 100

    # Setup for clear model Tracking:
    model_name = f"{n_layers}_layers_{hidden_features}_hiddenfeatures_{lr}_lr_{noise_schedule}_{timesteps}_timesteps_{batch_size}_batch_size_{epochs}_epochs_{remove_hydrogens}_Rem_Hydrogens"  # noqa
    folder_name = "Diffusion/Clean/" + model_name + "/"

    # # Setup WandB:
    # wandb.init(project="Diffusion_context_test", name=model_name)
    # wandb.config.name = model_name
    # wandb.config.batch_size = batch_size
    # wandb.config.epochs = epochs
    # wandb.config.lr = lr
    # wandb.config.hidden_node_features = hidden_features
    # wandb.config.number_of_layers = n_layers

    denoising_model = dynamics.EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_features,
        sin_embedding=True,
        n_layers=n_layers,
        device=device,
    )

    # denoising_model_graph = EGNN_dynamics_graph(
    #     in_node_nf=in_node_nf,
    #     context_node_nf=0,
    #     in_edge_nf=2,
    #     n_dims=3,
    #     out_node=3,
    #     hidden_nf=hidden_features,
    #     sin_embedding=True,
    #     n_layers=n_layers,
    #     device=device,
    # )

    dataset = W93_TS(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
    )

    # Calculate the sizes for each split
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.3, random_state=42
    )

    # Splitting the train dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.1, random_state=42
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
    )  # Not shuffled so that we can visualise the same samples

    # Setup the diffusion model:
    diffusion_model = DiffusionModel(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=noise_schedule,
        use_mse=False,
    )

    # diffusion_model_graph = DiffusionModel_graph(
    #     dynamics=denoising_model_graph,
    #     in_node_nf=in_node_nf,
    #     timesteps=timesteps,
    #     device=device,
    #     noise_schedule=noise_schedule,
    # )

    # Now let's try training the function and see if it works for now
    optimiser = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    diffusion_model.to(device)
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0

        # Setup training mode:
        diffusion_model.train()
        for batch, node_mask in tqdm(train_loader):
            optimiser.zero_grad()

            h = batch[:, :, :-3].to(device)
            x = batch[:, :, -3:].to(device)

            # setup the edge_mask:
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

            # Create mask for diagonal, as atoms cannot connect to themselves:
            diag_mask = (
                ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
                .unsqueeze(0)
                .bool()
            )

            # Expand to batch size:
            diag_mask = diag_mask.expand(edge_mask.size())

            # Multiply the edge mask by the diagonal mask to not have connections with itself:# noqa
            edge_mask *= diag_mask

            # Calculate the loss:
            nll = diffusion_model(
                x.to(device),
                h.to(device),
                node_mask.to(device),
                edge_mask.to(device),  # noqa
            )

            loss = nll.to(device)

            loss.backward()
            optimiser.step()

            total_train_loss += nll

        total_train_loss /= len(train_loader)
        print(f"At epoch {epoch} \t Train Loss = {total_train_loss}")
        # wandb.log({"Train_loss": total_train_loss})

        # Setup Validation part:
        diffusion_model.eval()
        with torch.no_grad():
            for batch, node_mask in tqdm(val_loader):
                h = batch[:, :, :-3].to(device)
                x = batch[:, :, -3:].to(device)

                # setup the edge_mask:
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

                # Create mask for diagonal, as atoms cannot connect to themselves:# noqa
                diag_mask = (
                    ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
                    .unsqueeze(0)
                    .bool()
                )

                # Expand to batch size:
                diag_mask = diag_mask.expand(edge_mask.size())

                # Multiply the edge mask by the diagonal mask to not have connections with itself:# noqa
                edge_mask *= diag_mask

                # Calculate the loss:
                nll = diffusion_model(
                    x.to(device),
                    h.to(device),
                    node_mask.to(device),
                    edge_mask.to(device),
                )
                loss = nll.to(device)
                total_val_loss += nll

        total_val_loss /= len(val_loader)
        print(f"At epoch {epoch} \t Val Loss = {total_val_loss}")
        # wandb.log({"val_loss": total_val_loss})
