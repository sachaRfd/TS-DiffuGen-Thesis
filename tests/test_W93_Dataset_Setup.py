import numpy as np
import os
import pandas as pd
import shutil
from data.Dataset_W93.setup_dataset_files import (
    extract_geometry,
    create_pdb_from_file_path,
    process_reactions,
)

"""
Test the Setup Function
"""


example_log_file = (
    "data/Dataset_W93/example_data_for_testing/TS/wb97xd3/rxn000000/p000000.log"  # noqa
)

ref_atoms = ["C", "C", "N", "O", "N", "N", "H", "H", "H"]
ref_geoms = np.array(
    [
        [-1.0107833, -0.01140629, -0.06095036],
        [0.47797759, 0.01913727, 0.01389809],
        [1.29737856, -0.99295119, 0.46925282],
        [0.6928138, -1.98447212, 0.83374563],
        [1.74560549, 1.97013284, -0.69764202],
        [1.16419833, 1.07626945, -0.37161078],
        [-1.40204563, 0.91336263, -0.4821491],
        [-1.33267002, -0.84990532, -0.68032037],
        [-1.43288383, -0.15535928, 0.93492709],
    ]
)


example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]

temp_dir = "TEST_DIRECTORY_TO_DELETE/"


def test_extract_geometry():
    # Read .log file
    with open(example_log_file, "r") as file:
        logs = file.read().splitlines()

    # Run extract function:
    atoms, geom = extract_geometry(logs=logs)

    assert atoms == ref_atoms
    assert np.allclose(geom, ref_geoms)


def test_pdb_from_log():
    # create temporary folder:
    os.mkdir(temp_dir)

    temp_file = temp_dir + "test.xyz"

    try:
        create_pdb_from_file_path(example_log_file, temp_file)

        # Assert that the file has been created with the corrext name:
        assert os.path.exists(temp_file)

        # read the file that was created
        with open(temp_file, "r") as file:  # noqa
            first_line = file.readline().strip()
            assert first_line == "9"
            # Check that the next line is empty
            assert file.readline().strip() == ""
            # Check that the 3rd line contains 4 strings
            third_line = file.readline().strip().split()
            assert len(third_line) == 4

    finally:
        # Delete temporary directory as it was just used for testing purposes
        shutil.rmtree(temp_dir)


def test_process_reactions():
    temp_dir = "data/Dataset_W93/example_data_for_testing"

    try:
        process_reactions(example_dataframe, temp_dir)

        # Assert that the required clean geometries directory has been created
        clean_geometries_directory = os.path.join(temp_dir, "Clean_Geometries")
        assert os.path.exists(clean_geometries_directory)

        # Assert that the expected number of reaction directories have been created # noqa
        count = sum(
            folder.startswith("Reaction_")
            for folder in os.listdir(clean_geometries_directory)
        )
        assert count == len(example_dataframe)

        # Check if images have been saved for each reaction
        for i in range(count):
            reaction_directory = os.path.join(
                clean_geometries_directory, f"Reaction_{i}"
            )
            reaction_image_path = os.path.join(
                reaction_directory, "Reaction_image.png"
            )  # noqa
            reaction_ts_path = os.path.join(
                reaction_directory, "TS_geometry.xyz"
            )  # noqa
            reaction_reactant_path = os.path.join(
                reaction_directory, "Reactant_geometry.xyz"
            )
            reaction_product_path = os.path.join(
                reaction_directory, "Product_geometry.xyz"
            )

            assert os.path.exists(reaction_image_path)
            assert os.path.exists(reaction_ts_path)
            assert os.path.exists(reaction_reactant_path)
            assert os.path.exists(reaction_product_path)

    finally:
        # Delete the reaction folders created in the Clean_Geometries directory: # noqa
        reaction_directories = os.listdir(clean_geometries_directory)
        for directory in reaction_directories:
            directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
            shutil.rmtree(directory_path)


if __name__ == "__main__":
    test_extract_geometry()
    test_pdb_from_log()
    test_process_reactions()
