from src.train_test import main
import os
import yaml
from data.Dataset_W93.setup_dataset_files import process_reactions
import pandas as pd
import shutil


"""
Tests for the train_test script using   config Files
"""

# Example Variables:
example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]
temp_dir = "data/Dataset_W93/example_data_for_testing"
temp_dir_clean_geo = (
    "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
)

config_pretrained_graphs = "configs/test_pre_trained_diffusion_with_graphs.yml"
config_train_simple_model = "configs/train_diffusion.yml"


def get_args_from_config(path):
    with open(path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    class ConfigArgs:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config_args = ConfigArgs(config_dict)
    return config_args


def setup_dataset():
    # Setup the dataset
    process_reactions(example_dataframe, temp_dir)


def delete_dataset():
    # Delete the dataset files created:
    shutil.rmtree(temp_dir_clean_geo)


def test_testing_function_with_graphs():
    # Setup the Dataset:
    setup_dataset()
    try:
        # Get arguments for graph model:
        args = get_args_from_config(path=config_pretrained_graphs)
        print(args)

        # Feed arguments into the test function:
        main(args=args, pytest_time=True)

        # Check that the amount of generated samples
        #  are equal to test_sampling_number
        sample_files = [
            file
            for file in os.listdir(
                "trained_models/" + args.folder_name + "Samples/batch_0/mol_0/"
            )
            if file.startswith("sample")
        ]

        assert len(sample_files) == args.test_sampling_number

        # Remove the generated files:
        samples_path = "trained_models/" + args.folder_name + "Samples/batch_0"
        shutil.rmtree(samples_path)

    finally:
        # Remove the Dataset:
        delete_dataset()


# Will be hard to test the training function as it requires access to WandB


if __name__ == "__main__":
    print("Running Script")
    test_testing_function_with_graphs()
