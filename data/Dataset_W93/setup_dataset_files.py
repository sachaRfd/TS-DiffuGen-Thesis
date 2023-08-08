import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image

from rdkit import Chem

from rdkit.Chem import Draw


def extract_geometry(logs):
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
                else:  # Means we have reached the end --> Return
                    return atoms, np.array(coords)


def create_pdb_from_file_path(path_to_raw_log, path_to_final):
    # Check that the raw file exists   --> Possibly make it do error handling
    if not os.path.exists(path_to_raw_log):
        print(
            "Path to Raw Log is incorrect - Please try again with another path"
        )  # noqa
        exit()
    # Check that the output path exsist:  --> Possibly make it do error Handling  # noqa
    if not os.path.exists(path_to_final):
        print(
            "Path to Raw Log is incorrect - Please try again with another path"
        )  # noqa
        exit()

    # Open the file and store all its lines into logs:
    with open(path_to_raw_log, "r") as file:
        logs = file.read().splitlines()

    atoms, coordinates = extract_geometry(logs)

    with open(path_to_final, "w") as file:
        counter = 1
        for atom, coord in zip(atoms, coordinates):
            x, y, z = coord
            file.write(f"{atom} {x} {y} {z}\n")
            counter += 1


if __name__ == "__main__":
    # First get the total number of reactions from the dataframe
    dataframe = pd.read_csv("data/w93_dataset/wb97xd3.csv", index_col=0)
    print(f"There are {dataframe.shape[0]} reactions in our data")

    # Now we can add the 3D coordinates for the 3D Reactants, 3D Product, and most importantly the TS into the data CSV file we created above  # noqa
    # First we check that we have the correct number of folders for Geometries
    directory = "data/TS/wb97xd3"
    start_string = "rxn"
    count = 0
    for folder in os.listdir(directory):
        if folder.startswith(start_string):
            count += 1
        else:
            print("Something else is in the folder")
    print(f"There are {count} TS in our data")

    # Now we can check that all folders contain the Reactant, Product and TS information  # noqa

    reactant_count, product_count, ts_count = 0, 0, 0
    for folder in os.listdir(directory):
        for files in os.listdir(os.path.join(directory, folder)):
            if files.startswith("ts"):
                ts_count += 1
            elif files.startswith("r"):
                reactant_count += 1
            elif files.startswith("p"):
                product_count += 1

            else:
                print("There is an extra something in one of the folders")

    # No we can assert that the counts are all the same --> making sure we have information for all the geometries:  # noqa
    assert (
        reactant_count
        == product_count
        == ts_count
        == count
        == dataframe.shape[0]  # noqa
    )  # noqa

    # Now we can create folders for each reaction - which contains pdb file for reactants. products and TS  # noqa
    directory = "data/"

    # if the clean_greometries fodler is not present then we can create it:
    if "Clean_Geometries" in os.listdir(directory):
        print(
            "Clean_Geometries folder is already present = Loading the PDB files"  # noqa
        )  # noqa
    else:
        print(
            "Creating the Clearn_Geometries folder - and loading the PDB files"
        )  # noqa
        # Create the Directory:
        os.mkdir(os.path.join(directory, "Clean_Geometries"))

    full_directory = os.path.join(directory, "Clean_Geometries")

    # Now we can create a folder for each reaction, and load the PDB files to each folder  # noqa
    for i in tqdm(range(count)):
        new_directory = os.path.join(full_directory, f"Reaction_{i}")
        # if the folder doesnt already exist --> create it
        if os.path.exists(new_directory):
            print("Folders are already present - Loading new geometries")
        else:
            os.makedirs(new_directory)

        # Update the Log directories: --> Made sure to be padded by 6 Zeroes
        product_path = f"data/TS/wb97xd3/rxn{i:06d}/p{i:06d}.log"
        reactant_path = f"data/TS/wb97xd3/rxn{i:06d}/r{i:06d}.log"
        ts_path = f"data/TS/wb97xd3/rxn{i:06d}/ts{i:06d}.log"
        # Now we can load all the clean geometries for the Reactants / Products and TS in their appropriate folders:  # noqa
        create_pdb_from_file_path(
            product_path,
            f"data/Clean_Geometries/Reaction_{i}/Product_geometry.xyz",  # noqa
        )
        create_pdb_from_file_path(
            reactant_path,
            f"data/Clean_Geometries/Reaction_{i}/Reactant_geometry.xyz",  # noqa
        )
        create_pdb_from_file_path(
            ts_path, f"data/Clean_Geometries/Reaction_{i}/TS_geometry.xyz"
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
        image_path = os.path.join(
            f"data/Clean_Geometries/Reaction_{i}/Reaction_image.png"
        )
        combined_image.save(image_path)
