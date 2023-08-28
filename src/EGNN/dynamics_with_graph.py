"""# noqa
Script for EGNN denoising model which only contains reactant coordinates and graph information about the product: 
--------------------------------

Code was adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/models.py

Main adaptations include the dynamics taking edge attributes as inputs.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import numpy as np
from torch.utils.data.dataset import random_split

from data.Dataset_W93.dataset_reactant_and_product_graph import (
    W93_TS_coords_and_reacion_graph,
)

from src.EGNN.egnn_with_bond_info import EGNN_with_bond
from src.EGNN.utils import (
    remove_mean,
    remove_mean_with_mask,
    assert_mean_zero_with_mask,
    setup_device,
)


class EGNN_dynamics_graph(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        n_dims: int = 3,
        out_node: int = 3,
        hidden_nf: int = 64,
        device: str = "cpu",
        act_fn=torch.nn.SiLU(),
        n_layers: int = 4,
        attention: bool = True,
        condition_time: bool = True,
        tanh: bool = False,
        norm_constant: int = 0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: int = 100,
        aggregation_method: str = "sum",
    ):
        super().__init__()
        self.egnn = EGNN_with_bond(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            out_node_nf=out_node,
            coords_range=10,
        )
        self.in_node_nf = in_node_nf

        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, edge_attributes):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, edge_attributes):
        def fwd(time, state):
            return self._forward(
                time,
                state,
                node_mask,
                edge_mask,
                edge_attributes,
            )

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, edge_attributes):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        # Re-view the edge_attributes so they match the whole batch:
        edge_attributes = edge_attributes.view(-1, 2)

        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]  # Send edges to device
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, h_dims:].clone()
        h = xh[:, :h_dims].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        # Here we ignore the h_final as we do not need it but keep in case of future use    # noqa
        _, x_final = self.egnn(
            h.to(xh.device),
            x.to(xh.device),
            edges,
            node_mask=node_mask.to(xh.device),
            edge_mask=edge_mask.to(xh.device),
            edge_attributes=edge_attributes.to(xh.device),
        )
        vel = (x_final - x) * node_mask

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        assert_mean_zero_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        return _, vel

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


if __name__ == "__main__":
    print("Running Script Model with MASK")

    device = setup_device()

    # Load the dataset:
    dataset = W93_TS_coords_and_reacion_graph(graph_product=True)

    # Calculate the sizes for each split
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Perform the train-test split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = 64

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True
    )  # noqa
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )  # noqa

    in_node_nf = 10 + 1  # Number of h features

    out_node = 3
    context_nf = 0
    n_dims = 3  # Dimension of the TS coordinates at each node (X, Y, Z)
    in_edge_nf = 2  # 2 edge features given indside then we add the distance later --> But changed EGNN function    # noqa

    model = EGNN_dynamics_graph(
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        n_dims=n_dims,
        out_node=out_node,
        sin_embedding=False,
        n_layers=5,
        device=device,
    )
    print("Model Setup")

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )  # noqa
    print(f"Number of Trainable Parameters is: {trainable_parameters}")

    example_sample, node_masks, edge_attributes = next(iter(train_loader))

    # Setup a basic training:
    epochs = 60
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        model.train()

        for batch, node_masks, edge_attributes in tqdm(train_loader):
            optimizer.zero_grad()

            batch.to(device)

            assert_mean_zero_with_mask(
                batch[:, :, -3:],
                node_masks.unsqueeze(2).expand(batch[:, :, -3:].size()),
            )

            true_coordinates = batch[:, :, -3:]

            true_noise = torch.randn_like(
                true_coordinates
            ) * node_masks.unsqueeze(  # noqa
                2
            ).expand(
                true_coordinates[:, :, -3:].size()
            )

            true_noise = remove_mean_with_mask(
                true_noise,
                node_masks.unsqueeze(2).expand(true_noise[:, :, -3:].size()),  # noqa
            )
            assert_mean_zero_with_mask(
                true_noise[:, :, -3:],
                node_masks.unsqueeze(2).expand(true_noise[:, :, -3:].size()),
            )

            noisy_coordinates = true_coordinates + true_noise
            assert_mean_zero_with_mask(
                noisy_coordinates[:, :, -3:],
                node_masks.unsqueeze(2).expand(
                    noisy_coordinates[:, :, -3:].size()
                ),  # noqa
            )

            batch[:, :, -3:] = noisy_coordinates.to(device)

            # Get the Edge mask:
            edge_mask = node_masks.unsqueeze(1) * node_masks.unsqueeze(2)
            # Create mask for diagonal, as atoms cannot connect to themselves:  # noqa
            diag_mask = (
                ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
                .unsqueeze(0)
                .bool()
            )
            # Expand to batch size:
            diag_mask = diag_mask.expand(edge_mask.size())
            # Multiply the edge mask by the diagonal mask to not have connections with itself:# noqa
            edge_mask *= diag_mask

            # Always the same timestep:
            t = torch.tensor([1]).to(device)
            _, x_final = model._forward(
                t=t,
                xh=batch.to(device),
                node_mask=node_masks.to(device),
                edge_mask=edge_mask.to(device),
                context=None,
                edge_attributes=edge_attributes,
            )
            loss = loss_fn(x_final, true_noise.to(device))
            loss.backward()  # BackProp the loss
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"At epoch {epoch}:\t train_loss = {train_loss}")
