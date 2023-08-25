import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


def extract_geometry(logs):
    """
    Extract geometry from DFT .log files
    """
    for index in reversed(range(len(logs))):
        line = logs[index]
        if "Standard Nuclear Orientation" in line:
            atoms, coords = [], []
            for line in logs[(index + 3) :]:  # noqa
                if "----------" not in line:
                    data = line.split()
                    atom = data[1]
                    coordinates = [float(geo) for geo in data[2:]]
                    atoms.append(atom)
                    coords.append(coordinates)
                else:
                    return atoms, np.array(coords)


def create_pdb_from_file_path(path_to_raw_log, path_to_final):
    """
    Write .XYZ file from a .log DFT file
    """
    if not os.path.exists(path_to_raw_log):
        print("Path to Raw Log is incorrect. Please provide a valid path.")
        exit()

    with open(path_to_raw_log, "r") as file:
        logs = file.read().splitlines()

    atoms, coordinates = extract_geometry(logs)

    with open(path_to_final, "w") as file:
        file.write(str(len(atoms)) + "\n\n")
        for atom, coord in zip(atoms, coordinates):
            x, y, z = coord
            file.write(f"{atom} {x} {y} {z}\n")


def process_reactions(dataframe, directory):
    """
    Function to Process .log files and write them in .xyz format
    """
    TS_Directory = directory + "/TS/wb97xd3/"

    # Assert that the above directory exists
    assert os.path.exists(TS_Directory), f"{TS_Directory} does not exist."

    # Assert that there is the same amount of reactions in the dataframe than in the TS_Directory:   # noqa
    start_string = "rxn"
    count = sum(
        folder.startswith(start_string) for folder in os.listdir(TS_Directory)
    )  # noqa
    print(f"There are {count} TS in our data")
    assert dataframe.shape[0] == count

    reactant_count, product_count, ts_count = 0, 0, 0
    for folder in os.listdir(TS_Directory):
        for files in os.listdir(os.path.join(TS_Directory, folder)):
            if files.startswith("ts"):
                ts_count += 1
            elif files.startswith("r"):
                reactant_count += 1
            elif files.startswith("p"):
                product_count += 1

    assert reactant_count == product_count == ts_count == count

    clean_geometries_directory = os.path.join(directory, "Clean_Geometries")
    os.makedirs(clean_geometries_directory, exist_ok=True)
    for i in tqdm(range(count)):
        reaction_directory = os.path.join(
            clean_geometries_directory, f"Reaction_{i}"
        )  # noqa
        os.makedirs(reaction_directory, exist_ok=True)

        product_path = directory + f"/TS/wb97xd3/rxn{i:06d}/p{i:06d}.log"
        reactant_path = directory + f"/TS/wb97xd3/rxn{i:06d}/r{i:06d}.log"
        ts_path = directory + f"/TS/wb97xd3/rxn{i:06d}/ts{i:06d}.log"

        create_pdb_from_file_path(
            product_path,
            os.path.join(reaction_directory, "Product_geometry.xyz"),  # noqa
        )
        create_pdb_from_file_path(
            reactant_path,
            os.path.join(reaction_directory, "Reactant_geometry.xyz"),  # noqa
        )
        create_pdb_from_file_path(
            ts_path, os.path.join(reaction_directory, "TS_geometry.xyz")
        )

        # Add image of reactant and product from each reaction to the folder too:  # noqa
        smiles_reactant = dataframe.iloc[i][0]
        smiles_product = dataframe.iloc[i][1]

        # Get the Molecule Object:
        mol_reactant = Chem.MolFromSmiles(smiles_reactant)
        mol_product = Chem.MolFromSmiles(smiles_product)

        # Set the dimensions for individual molecule images
        image_width = 300
        image_height = 300

        # Create blank image with extra space for reactant and product
        total_width = 2 * image_width + 20  # Add 20 pixels for spacing
        combined_image = Image.new(
            "RGB", (total_width, image_height), (255, 255, 255)
        )  # noqa

        # Draw reactant image on the left half
        reactant_image = Draw.MolToImage(
            mol_reactant, size=(image_width, image_height)
        )  # noqa
        combined_image.paste(reactant_image, (0, 0))

        # Draw product image on the right half
        product_image = Draw.MolToImage(
            mol_product, size=(image_width, image_height)
        )  # noqa
        combined_image.paste(
            product_image, (image_width + 20, 0)
        )  # Add 20 pixels for spacing

        # Save the combined image as a PNG file
        image_path = (
            clean_geometries_directory + f"/Reaction_{i}/Reaction_image.png"
        )  # noqa
        combined_image.save(image_path)
        count += 1
    print("Finished Setting up xyz files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process reactions data.")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to the directory containing reaction data.",  # noqa
    )
    args = parser.parse_args()

    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
    )

    if args.directory:
        directory = args.directory
    else:
        # directory = "data/Dataset_W93/example_data_for_testing"
        directory = "data/Dataset_W93/data"

    process_reactions(dataframe, directory)
