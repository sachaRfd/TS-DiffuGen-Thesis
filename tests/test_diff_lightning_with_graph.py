from src.lightning_setup import LitDiffusionModel_With_graph
from src.Diffusion.equivariant_diffusion import get_node_features
from data.Dataset_W93.setup_dataset_files import process_reactions
import shutil
import pandas as pd


"""
Script to test the Pytorch lightning class with graph

Will only be tested with the W93 Dataset as this was the dataset
we focussed on, as the others require .h5 downloads.
"""

example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]
temp_dir = "data/Dataset_W93/example_data_for_testing"
temp_dir_clean_geo = (
    "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
)


def setup_dataset():
    # Setup the dataset
    process_reactions(example_dataframe, temp_dir)


def delete_dataset():
    # Delete the dataset files created:
    shutil.rmtree(temp_dir_clean_geo)


def test_class_setup():
    # Setup the dataset:
    setup_dataset()

    # Setup Variables:
    learning_rate_schedule = False

    # If we do not include the product in the diffusoin step:
    no_product = False
    in_node_nf = get_node_features(
        no_product=no_product,
    )
    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    try:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=2,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            test_sampling_number=1,
            save_samples=False,
            save_path=None,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=True,
        )
        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))
        sample, sample_node_mask, sample_edge_attr = batch

        # The rest of the shapes are well known
        assert (
            sample.shape[0] == batch_size
        )  # Will only work for batchsize of 1 as the test dataset is so small (only 10 samples) # noqa
        assert sample.shape[1] == 16
        assert sample.shape[2] == 13

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 16

        assert sample_edge_attr.shape[0] == batch_size
        assert sample_edge_attr.shape[1] == 16**2
        assert sample_edge_attr.shape[2] == 2

        # Test every function:
        loss = lit_diff_model.training_step(batch=batch, batch_idx=0)
        # assert that the loss tensor includes requires grad to true:
        assert loss.requires_grad

    finally:
        delete_dataset()


def test_class_setup_no_product():
    # Setup the dataset:
    setup_dataset()

    # Setup Variables:
    learning_rate_schedule = False

    # If we do not include the product in the diffusoin step:
    no_product = True
    in_node_nf = get_node_features(
        no_product=no_product,
    )
    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    try:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=2,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            test_sampling_number=1,
            save_samples=False,
            save_path=None,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=True,
        )
        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        # Test every function:
        loss = lit_diff_model.training_step(batch=batch, batch_idx=0)
        # assert that the loss tensor includes requires grad to true:
        assert loss.requires_grad

    finally:
        delete_dataset()


if __name__ == "__main__":
    print("Running Script")
    test_class_setup()
    test_class_setup_no_product()
