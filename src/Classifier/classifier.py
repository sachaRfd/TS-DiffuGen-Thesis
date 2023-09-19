import torch
import torch.nn.functional as F
from torch import nn
from src.EGNN.egnn import EGNN, coord2diff, get_adj_matrix
from src.EGNN.utils import setup_device
from data.Dataset_generated_samples_classifier.generated_dataset_class import (
    classifier_dataset,
)  # noqa
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

"""

File for the classifier used for guiding the diffusion process. 

In this simple first proof of concept, an Invariant SchNet model will 
be predicting whether or not a generated TS is within COV < 0.1. It
should take as input the timestep t.


How to handle the data for training: 
- Sample a bunch of samples from different timesteps:
    --> The sample will all be from certain timesteps X_t
        --> Can convert them to X_0 with Equation xX.

    From this prediction, and the time-embedding --> Train model to 
    predict whether the generated sample has Cov < 0.1

"""  # noqa


# Derive the new class from the original EGNN --> Except make the final layer
# a fully connected layer with sigmoid in output so that value stays between
#  0 and 1
class EGNN_graph_prediction(EGNN):
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
        super().__init__(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            device,
            act_fn,
            n_layers,
            attention,
            out_node_nf,
            tanh,
            coords_range,
            norm_constant,
            inv_sublayers,
            sin_embedding,
            normalization_factor,
            aggregation_method,
        )
        self.out_node_nf = out_node_nf
        # Create the linear layer for final output:
        self.final_linear_model = nn.Sequential(
            nn.Linear(
                23 * self.out_node_nf,
                2,  # 2 outputs --> 0 or 1
            ),  # 23 ATOMS here is hard-coded
            nn.Softmax(dim=1),
        )

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

        # Make the final layer take the whole graph and output value:
        output_prediction = self.final_linear_model(
            x.view(
                -1,
                23 * self.out_node_nf,
            )
        )
        return output_prediction


if __name__ == "__main__":
    print("Running Script")
    n_samples = 20
    only_ts = True

    device = setup_device()

    # Dummy parameters
    batch_size = 32
    n_nodes = 23
    if only_ts:
        n_feat = 10
    else:
        n_feat = 13  # 10 + 3 to include the generated TS

    # Setup dataset:
    dataset = classifier_dataset(
        number_of_samples=n_samples,
        only_output_ts=False,
    )

    # In all papers they use 8:1:1 ratio
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=42
    )

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
        shuffle=False,
    )

    sample, node_mask, label = next(iter(train_loader))

    hidden_nf = 16
    n_layers = 1
    classifier_model = EGNN_graph_prediction(
        in_node_nf=n_feat,
        hidden_nf=hidden_nf,
        out_node_nf=3,
        in_edge_nf=1,
        n_layers=n_layers,
    )
    classifier_model.to(device)

    # Define the loss function
    loss_fn = nn.BCELoss()

    # Define the optimizer
    lr = 2e-4
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    epochs = 400

    # model_name = f"{only_ts}_only_ts_{n_samples}_number_of_samples_{batch_size}_bs_{hidden_nf}_hidden_nf_{n_layers}_layers_{epochs}_epoch_{lr}_lr"  # noqa
    # wandb.init(project="Diffusion_classifier", name=model_name)
    # wandb.config.name = model_name
    # wandb.config.batch_size = batch_size
    # wandb.config.epochs = epochs
    # wandb.config.lr = lr
    # wandb.config.hidden_node_features = hidden_nf
    # wandb.config.number_of_layers = n_layers

    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        val_loss = 0
        val_acc = 0
        classifier_model.train()  # Setup the training mode
        for batch, node_mask, label in tqdm(train_loader):
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

            # Get adj matrix:
            edges = get_adj_matrix(
                n_nodes=n_nodes,
                batch_size=batch_size,
                device=device,
            )
            edges = [
                edge.to(device) for edge in edges
            ]  # Convert each tensor in the list to GPU tensor

            h = batch[:, :, :n_feat]
            x = batch[:, :, -3:]

            # Reshapes:
            h = h.view(-1, n_feat)
            x = x.view(-1, 3)

            node_mask = node_mask.view(batch_size * n_nodes, 1)

            print(h.shape)
            print(x.shape)
            print(len(edges))
            print(len(edges[0]))
            print(node_mask.shape)
            print(edge_mask.shape)
            exit()
            #     exit()
            # exit()
            predicted_label_proba = classifier_model(
                h.to(device),
                x.to(device),
                edges,
                node_mask=node_mask.to(device),
                edge_mask=edge_mask.to(device),
            )

            label_ohe = (
                F.one_hot(label.long(), num_classes=2).float().to(device)
            )  # noqa

            loss = loss_fn(predicted_label_proba, label_ohe).to(device)
            acc = (
                (
                    torch.argmax(predicted_label_proba, 1)
                    == torch.argmax(
                        label_ohe,
                        1,
                    )
                )
                .float()
                .mean()
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"At epoch {epoch}:\t train_loss = {train_loss}")
        print(f"At epoch {epoch}:\t train_accuracy = {train_acc}")
        wandb.log({"Train_loss": train_loss})
        wandb.log({"Train_accuracy": train_acc})

        classifier_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch, node_mask, label in tqdm(val_loader):
                batch_size = batch.shape[0]
                n_nodes = batch.shape[1]

                edge_mask = (node_mask.unsqueeze(1)) * (node_mask.unsqueeze(2))

                # Create mask for diagonal, as atoms cannot
                # connect to themselves:
                diag_mask = (
                    ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
                    .unsqueeze(0)
                    .bool()
                )

                # Expand to batch size:
                diag_mask = diag_mask.expand(edge_mask.size())

                # Multiply the edge mask by the diagonal mask to not
                # have connections with itself:
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * 23 * 23, 1)

                # Get adj matrix:
                edges = get_adj_matrix(
                    n_nodes=n_nodes,
                    batch_size=batch_size,
                    device=device,
                )
                edges = [edge.to(device) for edge in edges]

                h = batch[:, :, :n_feat]
                x = batch[:, :, -3:]

                # Reshapes:
                h = h.view(-1, n_feat)
                x = x.view(-1, 3)

                node_mask = node_mask.view(batch_size * n_nodes, 1)

                predicted_label_proba = classifier_model(
                    h.to(device),
                    x.to(device),
                    edges,
                    node_mask=node_mask.to(device),
                    edge_mask=edge_mask.to(device),
                )

                label_ohe = (
                    F.one_hot(
                        label.long(),
                        num_classes=2,
                    )
                    .float()
                    .to(device)
                )

                loss = loss_fn(predicted_label_proba, label_ohe).to(device)
                acc = (
                    (
                        torch.argmax(predicted_label_proba, 1)
                        == torch.argmax(label_ohe, 1)
                    )
                    .float()
                    .mean()
                )

                val_loss += loss.item()
                val_acc += acc.item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"At epoch {epoch}:\t validation_loss = {val_loss}")
            print(f"At epoch {epoch}:\t validation_accuracy = {val_acc}")
            wandb.log({"Val_loss": val_loss})
            wandb.log({"Val_accuracy": val_acc})

    # Save the model:
    path = "trained_classifier/" + model_name + "/Weights"
    os.makedirs(path, exist_ok=True)
    save_path = path + "/weights.pt"
    torch.save(classifier_model.state_dict(), save_path)
