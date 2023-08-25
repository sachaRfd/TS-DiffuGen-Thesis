"""    # noqa
Script for Equivariant Graph Neural Networks:
---------------------------------------------

Code was adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/egnn/models.py


Main Adaptations: 
    1. Clearn up code
    2. Debug
    3. Allowed for setting up training and Proof of Concept
    4. Adapted code to allow for extra edge-attributes (edge features to be added) --> Check Other EGNN script for this
"""

from torch import nn
import torch
import math
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.EGNN.utils import setup_device


from data.Dataset_W93.dataset_class import W93_TS


class GCL(nn.Module):
    """# noqa
    Graph Convolutional Layer (GCL) module.

    Args:
        input_nf (int): Number of input node features.
        output_nf (int): Number of output node features.
        hidden_nf (int): Number of hidden node features.
        normalization_factor (int): Normalization factor for aggregation.
        aggregation_method (str): Aggregation method for aggregating edge information.
        edges_in_d (int): Dimensionality of additional edge attributes.
        nodes_att_dim (int): Dimensionality of additional node attributes.
        act_fn (nn.Module): Activation function applied to MLP layers.
        attention (bool): Whether to apply attention mechanism.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                input_edge + edges_in_d, hidden_nf
            ),  # Difference here does not include + edge_coords_nf in input
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(
        self, source, target, edge_attr, edge_mask
    ):  # Here it uses edge attr instead of radial
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, _ = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    """#noqa
    Equivariant Update module for EGNN.

    Args:
        hidden_nf (int): Number of hidden node features.
        normalization_factor (int): Normalization factor for aggregation.
        aggregation_method (str): Aggregation method for aggregating edge information.  # noqa
        edges_in_d (int): Dimensionality of additional edge attributes.
        act_fn (nn.Module): Activation function applied to MLP layers.
        tanh (bool): Whether to apply hyperbolic tangent to coordinates.
        coords_range (float): Range of coordinate values.
    """

    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr,
        edge_mask,
    ):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        coord = self.coord_model(
            h, coord, edge_index, coord_diff, edge_attr, edge_mask
        )  # noqa
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    """# noqa
    Equivariant Block module for EGNN.

    Args:
        hidden_nf (int): Number of hidden node features.
        edge_feat_nf (int): Number of edge feature dimensions.
        device (str): Device to run the module on.
        act_fn (nn.Module): Activation function applied to MLP layers.
        n_layers (int): Number of layers in the block.
        attention (bool): Whether to apply attention mechanism.
        norm_diff (bool): Whether to normalize coordinate differences.
        tanh (bool): Whether to apply hyperbolic tangent to coordinates.
        coords_range (float): Range of coordinate values.
        norm_constant (int): Normalization constant for coordinate differences.
        sin_embedding (nn.Module): Sinusoid embedding for distances.
        normalization_factor (int): Normalization factor for aggregation.
        aggregation_method (str): Aggregation method for aggregating edge information.
    """

    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
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
        edge_attr=None,
    ):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    """# noqa
    Equivariant Graph Neural Network (EGNN) module.

    Args:
        in_node_nf (int): Number of input node features.
        in_edge_nf (int): Number of input edge features.
        hidden_nf (int): Number of hidden node features.
        device (str): Device to run the module on.
        act_fn (nn.Module): Activation function applied to MLP layers.
        n_layers (int): Number of layers in the EGNN.
        attention (bool): Whether to apply attention mechanism.
        norm_diff (bool): Whether to normalize coordinate differences.
        out_node_nf (int): Number of output node features.
        tanh (bool): Whether to apply hyperbolic tangent to coordinates.
        coords_range (float): Range of coordinate values.
        norm_constant (int): Normalization constant for coordinate differences.
        inv_sublayers (int): Number of layers in the EquivariantBlock.
        sin_embedding (bool): Whether to use sinusoid embeddings.
        normalization_factor (int): Normalization factor for aggregation.
        aggregation_method (str): Aggregation method for aggregating edge information.
    """

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
        super(EGNN, self).__init__()
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
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2  # May have to change it here

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

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=distances,
            )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = (
                h * node_mask
            )  # Squeeze as you want everything to be one dimension here# noqa
        return h, x


class SinusoidsEmbeddingNew(nn.Module):
    """# noqa
    Sinusoidal embedding module for EGNN.

    Args:
        max_res (float): Maximum resolution.
        min_res (float): Minimum resolution.
        div_factor (int): Division factor.
    """

    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2
            * math.pi
            * div_factor ** torch.arange(self.n_frequencies)
            / max_res  # noqa
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    """# noqa
    Converts node coordinates to differences and radial distances.

    Args:
        x (Tensor): Node coordinates.
        edge_index (Tensor): Edge indices.
        norm_constant (int, optional): Normalization constant for coordinate differences.

    Returns:
        Tensor: Radial distances.
        Tensor: Coordinate differences.
    """
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(
    data,
    segment_ids,
    num_segments,
    normalization_factor: int,
    aggregation_method: str = "sum",  # noqa
):
    """# noqa
    Performs unsorted segment sum operation with normalization.

    Args:
        data (Tensor): Data to be segmented.
        segment_ids (Tensor): Indices indicating segments. rows
        num_segments (int): Number of segments.
        normalization_factor (int): Normalization factor for aggregation.
        aggregation_method (str): Aggregation method ('sum' or 'mean').

    Returns:
        Tensor: Result of the operation.
    """

    assert aggregation_method in [
        "sum",
        "mean",
    ], "Please use either mean or sum, as they are permutation invariant aggregation functions"  # noqa

    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)

    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


# Added extra to create the edges:
def get_edges(n_nodes):
    """# noqa
    Generates edges for a graph with a given number of nodes.

    Args:
        n_nodes (int): Number of nodes.

    Returns:
        List: List of rows and columns representing edges.
    """

    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    """noqa
    Generates edges for a batch of graphs with a given number of nodes.

    Args:
        n_nodes (int): Number of nodes in each graph.
        batch_size (int): Batch size.

    Returns:
        Tuple: Edge information (edge indices and attributes).
    """
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


_edges_dict = {}


def get_adj_matrix(n_nodes, batch_size, device):
    """# noqa
    Generates adjacency matrix for a batch of graphs with a given number of nodes.

    Args:
        n_nodes (int): Number of nodes in each graph.
        batch_size (int): Batch size.
        device (str): Device to run the operation on.

    Returns:
        Tuple: Edge information (edge indices and attributes).
    """
    if n_nodes in _edges_dict:
        edges_dic_b = _edges_dict[n_nodes]
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
        _edges_dict[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)


if __name__ == "__main__":
    print("Running Script")

    direc = "data/Dataset_W93/data/Clean_Geometries"
    dataset = W93_TS(directory=direc, graph=False)

    # Calculate the sizes for each split
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Perform the train-test split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = 1

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
        shuffle=True,
    )

    device = setup_device()

    # Dummy parameters
    batch_size = 64
    n_nodes = 23
    n_feat = 10
    x_dim = 3

    egnn = EGNN(
        in_node_nf=n_feat,
        hidden_nf=64,
        out_node_nf=3,
        in_edge_nf=1,
        n_layers=3,
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

        for batch, node_mask in tqdm(train_loader):
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

            h = batch[:, :, :10]
            x = batch[:, :, 10:]

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
                n_nodes=n_nodes,
                batch_size=batch_size,
                device=device,
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
            )

            loss = loss_fn(x_out, noise.view(-1, x_dim).to(device))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"At epoch {epoch}:\t train_loss = {train_loss}")

        egnn.eval()  # Switch to evaluation mode for validation
        with torch.no_grad():
            for batch, node_mask in tqdm(val_loader):
                batch_size = batch.shape[0]
                n_nodes = batch.shape[1]

                edge_mask = (node_mask.unsqueeze(1)) * (node_mask.unsqueeze(2))

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
                edge_mask = edge_mask.view(batch_size * 23 * 23, 1)

                h = batch[:, :, :10]
                x = batch[:, :, 10:]

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
                    node_mask.to(device),
                    edge_mask=edge_mask.to(device),
                )

                loss = loss_fn(x_out, noise.view(-1, x_dim).to(device))

                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"At epoch {epoch}:\t val_loss = {val_loss}")
