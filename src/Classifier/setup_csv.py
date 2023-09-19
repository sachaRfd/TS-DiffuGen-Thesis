import pandas as pd
import os
from src.evaluate_samples import calculate_distance_matrix, calculate_DMAE
from tqdm import tqdm

"""

This script will be to create the CSV file
with the RMSE of the distance matrices and 
the labels to use within the classifier model.


1. Each row in the CSV will be a molecule.
2. First Column is the reaction number
3. Then the next columns will be the samples RMSE of distance matrices
4. Then Classification 0 for the RMSE is not < 0.1, 
   it is 1 if the RMSE is < 0.1 


Things to adapt when used:
1. Make it change the number of generated samples to 40
2. Change fake_path to "data/Dataset_generated_samples/Clean/"
3. Change last string of true_ts_path to "TS_geometry.xyz"
4. Change last string of sample_ts_path to "Sample_{i}.xyz"

"""  # noqa


def read_xyz_file(file_path):
    """
    Reads an XYZ file and returns its data in float format.
    """
    data = []
    with open(file_path, "r") as read_file:
        lines = read_file.readlines()
        mol = []
        for atom in lines[2:]:
            atom = atom.split(" ")
            data = (float(atom[1]), float(atom[2]), float(atom[3]))
            mol.append(data)
    return mol


if __name__ == "__main__":
    print("Running Script ")

    # Change path when done sampling:
    fake_path = "data/Dataset_generated_samples/Clean/"

    # Get the list of directories that start with "mol_"
    items_in_dir = os.listdir(fake_path)
    number_of_samples = 40

    df_list = []

    for mol in tqdm(items_in_dir):
        reaction_number = int(mol.split("_")[-1])
        row = [reaction_number]

        # Setup the True TS:
        true_ts_path = os.path.join(fake_path, mol, "TS_geometry.xyz")
        true_ts = read_xyz_file(true_ts_path)
        true_ts_dm = calculate_distance_matrix(true_ts)

        for i in range(number_of_samples):  # Should be 40 here
            sample_ts_path = os.path.join(fake_path, mol, f"Sample_{i}.xyz")
            sample_ts = read_xyz_file(sample_ts_path)
            sample_ts_dm = calculate_distance_matrix(sample_ts)
            sample_dmae = calculate_DMAE(
                gen_mol=true_ts_dm,
                true_mol=sample_ts_dm,
            )
            if sample_dmae < 0.1:
                row.append(1)
            else:
                row.append(0)
        df_list.append(row)

    # Create a DataFrame with reaction number as the index
    df = pd.DataFrame(
        data=df_list,
        columns=["Reaction Number"]
        + [f"Sample_{i}" for i in range(number_of_samples)],  # noqa
    )
    df.set_index("Reaction Number", inplace=True)
    df.sort_values("Reaction Number", inplace=True)

    path_to_save = "data/Dataset_generated_samples/reaction_dmae_labels.csv"
    df.to_csv(path_to_save)
