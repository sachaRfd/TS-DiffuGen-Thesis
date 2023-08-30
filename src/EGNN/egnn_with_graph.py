# Sacha Raffaud sachaRfd and acse-sr1022

from torch import nn
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.EGNN.egnn import (
    SinusoidsEmbeddingNew,
    setup_device,
    get_adj_matrix,
    EquivariantBlock,
)

from data.Dataset_W93.dataset_reaction_graph import (
    W93_TS_coords_and_reacion_graph,
)


"""

This adapted EGNN class from the EGNN file.
The main adaptation being that the EGNN is now able
to take as input extra edge attributes.

These edge attributes contain information about the 
reaction graph of the input molecules.

Oringial proof-of-concept was kept in the script for future
reference. May require slight modifications if certain functions
are changed.

"""  # noqa


class EGNN_with_bond(nn.Module):
    """
    Enhanced Graph Neural Network (EGNN) with Bond Information.

    This class defines an EGNN model with support for bond information in addition to node and edge features.
    It applies equivariant graph neural network layers to process node and edge features, considering bond information
    and interaction between nodes. The model supports multiple layers, attention mechanisms, and various configuration
    options to customize the behavior of the network.

    Args:
        in_node_nf (int): Dimensionality of input node features.
        in_edge_nf (int): Dimensionality of input edge features.
        hidden_nf (int): Dimensionality of hidden features in the equivariant blocks.
        device (str, optional): Device to which the model will be moved (default is "cpu").
        act_fn (torch.nn.Module, optional): Activation function to be used in the equivariant blocks (default is nn.SiLU()).
        n_layers (int, optional): Number of equivariant blocks (default is 3).
        attention (bool, optional): Whether to use attention mechanism in the equivariant blocks (default is False).
        out_node_nf (int, optional): Dimensionality of output node features. If not provided, it will be set to in_node_nf.
        tanh (bool, optional): Whether to apply tanh activation to the output node features (default is False).
        coords_range (float, optional): Range of coordinate values for positional encodings (default is 5).
        norm_constant (float, optional): Normalization constant for positional encodings (default is 1).
        inv_sublayers (int, optional): Number of layers in the equivariant block's inverse feature extraction (default is 2).
        sin_embedding (bool, optional): Whether to use sinusoidal embeddings for edge features (default is False).
        normalization_factor (int, optional): Normalization factor for equivariant block's aggregation (default is 100).
        aggregation_method (str, optional): Method for aggregating node information (default is "sum").
    """  # noqa

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=3,
        attention=False,
        out_node_nf=None,
        tanh=False,
        coords_range=5,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EGNN_with_bond, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim + in_edge_nf
        else:
            self.sin_embedding = None
            # Add the extra dimension that will be added by the distance being added    # noqa
            edge_feat_nf = in_edge_nf + 1

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    device=device,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.to(self.device)

    def forward(
        self,
        h,
        x,
        edge_index,
        node_mask=None,
        edge_mask=None,
        edge_attributes=None,
    ):
        """
        Forward pass of the EGNN_with_bond model.

        Args:
            h (torch.Tensor): Input node features.
            x (torch.Tensor): Input edge features.
            edge_index (torch.Tensor): Graph edge indices.
            node_mask (torch.Tensor, optional): Mask for node features (default is None).
            edge_mask (torch.Tensor, optional): Mask for edge features (default is None).
            edge_attributes (torch.Tensor, optional): Additional attributes associated with edges (default is None).

        Returns:
            h (torch.Tensor): Output node features.
            x (torch.Tensor): Output edge features.
        """  # noqa
        # Instead of concatenating the distances twice, let's just add information about bonds:     # noqa
        edge_attr = edge_attributes
        # distances, _ = coord2diff(x, edge_index)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = (
                h * node_mask
            )  # Squeeze as you want everything to be one dimension here# noqa
        return h, x


if __name__ == "__main__":
    print("Running Script")

    dataset = W93_TS_coords_and_reacion_graph()

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
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )  # noqa
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )  # noqa

    device = setup_device()

    # Dummy parameters
    n_nodes = 23
    n_feat = 4  # Should be 7 as we dont want to use the product coordinates
    x_dim = 3
    in_edge_nf = 2  # Number of features in the edges (Distance + reactant bond OHE, product Bond OHE)  # noqa
    egnn = EGNN_with_bond(
        in_node_nf=n_feat,
        hidden_nf=64,
        out_node_nf=3,
        in_edge_nf=in_edge_nf,
        n_layers=2,
        sin_embedding=False,  # noqa
    )
    egnn.to(device)

    print(
        f"There are {sum(p.numel() for p in egnn.parameters() if p.requires_grad)} parameters in the model."  # noqa
    )

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(egnn.parameters(), lr=0.003)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        egnn.train()  # Setup the training mode

        for batch, node_mask, edge_features in tqdm(train_loader):
            # Concatenate all the edges together into one big tensor:
            edge_features = edge_features.view(-1, 2)

            optimizer.zero_grad()

            batch_size = batch.shape[0]
            n_nodes = batch.shape[1]

            edge_mask = (node_mask.unsqueeze(1)) * (node_mask.unsqueeze(2))

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
            edge_mask = edge_mask.view(batch_size * 23 * 23, 1)

            h = batch[:, :, :4]
            x = batch[:, :, -3:]

            # Add noise to x samples
            noise = torch.randn_like(x)  # Adding Gaussian noise

            noise_mean = torch.mean(noise, dim=1, keepdim=True)

            noise = noise - noise_mean

            # Mulitply noise  with the node mask
            noise = noise * node_mask.unsqueeze(2).expand(noise.size())

            # Reshape the nodemask:
            node_mask = node_mask.view(batch_size * n_nodes, 1)

            x_noisy = x + noise

            h = h.view(-1, n_feat)
            x_noisy = x_noisy.view(-1, x_dim)

            edges = get_adj_matrix(
                n_nodes=n_nodes, batch_size=batch_size, device=device
            )

            edges = [
                edge.to(device) for edge in edges
            ]  # Convert each tensor in the list to GPU tensor

            h_out, x_out = egnn(
                h.to(device),
                x_noisy.to(device),
                edges,
                node_mask=node_mask.to(device),
                edge_mask=edge_mask.to(device),
                edge_attributes=edge_features.to(device),
            )

            loss = loss_fn(x_out, noise.view(-1, x_dim).to(device))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"At epoch {epoch}:\t train_loss = {train_loss}")
