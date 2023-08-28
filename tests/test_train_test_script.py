# from src.train_test import main
# import argparse
# import os
# import pytest
# from unittest.mock import Mock
# from data.Dataset_W93.setup_dataset_files import process_reactions
# import pandas as pd
# import shutil

# """
# Tests for the train_test script using   config Files
# """

# # Example Variables:
# example_dataframe = pd.read_csv(
#     "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
# )[:10]
# temp_dir = "data/Dataset_W93/example_data_for_testing"
# temp_dir_clean_geo = (
#     "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
# )


# @pytest.fixture
# def mock_args():
#     mock_args = Mock(
#         train_test="train",
#         model_name="test_model",
#         folder_name="test_folder",
#         use_graph_in_model=False,
#         learning_rate_scheduler=False,
#         epochs=10,
#     )
#     return mock_args


# def setup_dataset():
#     # Setup the dataset
#     process_reactions(example_dataframe, temp_dir)


# def delete_dataset():
#     # Delete the dataset files created:
#     shutil.rmtree(temp_dir_clean_geo)


# if __name__ == "__main__":
#     print("Running Script")
