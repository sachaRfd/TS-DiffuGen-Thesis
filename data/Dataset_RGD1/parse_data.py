import h5py
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

""" # noqa
Adapted from the following repository: https://zenodo.org/record/7618731


Script to iterate over the RGD1 Dataset 

- Download the RGD1_CHNO.h5 file and place it in the same directory as this one. 
- Then run following script to create the .XYZ file
"""


def save_xyz_file(path, atoms, coordinates):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, coord in zip(atoms, coordinates):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def main():
    """
    Main Function to save the samples in required format
    """

    hf = h5py.File("data/Dataset_RGD1/RGD1_CHNO.h5", "r")

    # Convert number to symbol
    num2element = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    count = 0
    for Rind, Rxn in hf.items():
        print("Paring Reaction {}".format(Rind))

        # Parse smiles
        # Rsmiles, Psmiles = str(np.array(Rxn.get("Rsmiles"))), str(
        #     np.array(Rxn.get("Psmiles"))
        # )

        # Parse elements
        elements = [num2element[Ei] for Ei in np.array(Rxn.get("elements"))]

        # Parse geometries
        TS_G = np.array(Rxn.get("TSG"))
        R_G = np.array(Rxn.get("RG"))
        P_G = np.array(Rxn.get("PG"))

        save_path = f"data/Dataset_RGD1/data/Clean_Geometries/Reaction_{count}/"  # noqa
        os.makedirs(save_path, exist_ok=True)

        # Save TS, reactant, and product geometries as .xyz files
        save_xyz_file(
            os.path.join(save_path, "TS_geometry.xyz"), elements, TS_G
        )  # noqa
        save_xyz_file(
            os.path.join(save_path, "Reactant_geometry.xyz"), elements, R_G
        )  # noqa
        save_xyz_file(
            os.path.join(save_path, "Product_geometry.xyz"), elements, P_G
        )  # noqa

        print(f"Geometries saved for Reaction {count}")
        count += 1
        exit()


def checking_duplicates():
    """# noqa
    Function to check for duplicate reactants and products based on SMILES strings, 
    and only save the first one to file.
    """
    # Read the CSV file and open the HDF5 file
    # csv_file = pd.read_csv("data/Dataset_RGD1/DFT_reaction_info.csv")
    hf = h5py.File("data/Dataset_RGD1/RGD1_CHNO.h5", "r")

    unique_reactions = {}  # To store unique reactions' data
    duplicate_reaction_count = 0
    sample_count = 0

    for Rind, Rxn in hf.items():
        if sample_count >= 1_000:
            break
        # Parse SMILES strings
        Rsmiles = str(np.array(Rxn.get("Rsmiles")))
        Psmiles = str(np.array(Rxn.get("Psmiles")))

        # Get Forward reaction activation energy from csv_file using the Rind
        # activation_energy = csv_file[csv_file.channel == Rind].DE_F.item()
        # print(activation_energy)
        reaction_smiles = Rsmiles + "." + Psmiles

        if reaction_smiles in unique_reactions:
            duplicate_reaction_count += 1
            if (
                Rind
                not in unique_reactions[reaction_smiles]["duplicate_indices"]  # noqa
            ):  # noqa
                unique_reactions[reaction_smiles]["duplicate_indices"].append(
                    Rind
                )  # noqa
        else:
            unique_reactions[reaction_smiles] = {
                "name": Rind,
                "smiles": reaction_smiles,
                "duplicate_indices": [Rind],
            }

        sample_count += 1

    print(f"There is a total of {sample_count} reactions")
    print(f"Total duplicate reactions: {duplicate_reaction_count}")
    print("Duplicate reaction data:")
    dup_count = 0
    duplicate_counts = {2: 0, 3: 0, 4: 0}

    for reaction_info in unique_reactions.values():
        num_duplicates = len(reaction_info["duplicate_indices"])
        if num_duplicates > 1:
            if num_duplicates in duplicate_counts:
                duplicate_counts[num_duplicates] += 1
            dup_count += 1

    print(f"Total duplicates with 2 duplicates: {duplicate_counts[2]}")
    print(f"Total duplicates with 3 duplicates: {duplicate_counts[3]}")
    print(f"Total duplicates with 4 duplicates: {duplicate_counts[4]}")
    print(
        f"Total duplicates with more than 4 duplicates: {dup_count - sum(duplicate_counts.values())}"  # noqa
    )


def save_reactions_with_multiple_ts(save_even_single=False):
    """# noqa

    - Function that finds reactions with multiple TS conformers (By looking at identical reaction smiles)
    - Then saves each reaction inside the directory:
        data/RDD1_Dataset/data/Multiple_TS/Reaction_{reaction_smiles}
    - Inside that directory, it saves the true_reactant, true_product, TS_1, TS_X_{activation_energy}, TS_x_{Activation_energy}

    """
    # Read the CSV file and open the HDF5 file
    csv_file = pd.read_csv("data/Dataset_RGD1/DFT_reaction_info.csv")
    hf = h5py.File("data/Dataset_RGD1/RGD1_CHNO.h5", "r")

    # Create a directory to save reactions with multiple TS
    save_dir = "data/Dataset_RGD1/data/Single_and_Multiple_TS/"
    os.makedirs(save_dir, exist_ok=True)

    # Counter for processed reactions
    processed_reactions = 0

    # To store unique reactions' data
    unique_reactions = {}

    # Convert number to symbol
    num2element = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    # Iterate through reactions to identify those with multiple TS conformers
    for Rind, Rxn in tqdm(hf.items()):
        # if processed_reactions >= 1000:  # Process only the first 10 reactions    # noqa
        #     break

        # Parse SMILES strings
        Rsmiles = str(np.array(Rxn.get("Rsmiles")))
        Psmiles = str(np.array(Rxn.get("Psmiles")))

        # Parse elements
        elements = [num2element[Ei] for Ei in np.array(Rxn.get("elements"))]

        # Parse geometries
        TS_G = np.array(Rxn.get("TSG"))
        R_G = np.array(Rxn.get("RG"))
        P_G = np.array(Rxn.get("PG"))

        # Get Forward reaction activation energy from csv_file using the Rind
        activation_energy = csv_file[csv_file.channel == Rind].DE_F.item()
        reaction_smiles = Rsmiles + "." + Psmiles

        if reaction_smiles in unique_reactions:
            unique_reactions[reaction_smiles]["duplicate_indices"].append(Rind)
            unique_reactions[reaction_smiles]["TS_geometry"].append(TS_G)
            unique_reactions[reaction_smiles]["activation_energy"].append(
                activation_energy
            )

        else:
            unique_reactions[reaction_smiles] = {
                # "name": Rind,
                "smiles": reaction_smiles,
                "duplicate_indices": [Rind],
                "elements": elements,
                "reactant_geometry": R_G,
                "product_geometry": P_G,
                "TS_geometry": [TS_G],
                "activation_energy": [activation_energy],
            }
        processed_reactions += 1

    # Save reactions with multiple TS conformers
    reaction_count = 0
    for reaction_info in tqdm(unique_reactions.values()):
        duplicate_indices = reaction_info["duplicate_indices"]
        print(duplicate_indices)

        # Check if there are multiple TS conformers for this reaction
        if save_even_single:
            threshold = 0
        else:
            threshold = 1
        if (
            len(duplicate_indices) > threshold
        ):  # Should be 1 here for multiple BUT I CHANGED IT SO WE HAVE AT LEAST 1 SAMPLE FOR EACH REACTION # noqa
            reaction_dir = os.path.join(save_dir, f"Reaction_{reaction_count}")  # noqa
            reaction_count
            os.makedirs(reaction_dir, exist_ok=True)

            # Save the true reactant and true product geometries as .xyz files  # noqa
            save_xyz_file(
                os.path.join(reaction_dir, "Reactant_geometry.xyz"),
                reaction_info["elements"],
                reaction_info["reactant_geometry"],
            )
            save_xyz_file(
                os.path.join(reaction_dir, "Product_geometry.xyz"),
                reaction_info["elements"],
                reaction_info["product_geometry"],
            )

            # Save each TS conformer geometry as .xyz files
            assert len(reaction_info["TS_geometry"]) == len(
                reaction_info["activation_energy"]
            )
            counter_ts = 0
            for ts, ae in zip(
                reaction_info["TS_geometry"], reaction_info["activation_energy"]  # noqa
            ):
                # This if statement is just so we have a sample that has no doubles # noqa
                if counter_ts == 0:
                    save_xyz_file(
                        os.path.join(reaction_dir, "TS_geometry.xyz"),
                        reaction_info["elements"],
                        ts,
                    )
                save_xyz_file(
                    os.path.join(reaction_dir, f"ts_{ae}.xyz"),
                    reaction_info["elements"],
                    ts,
                )
                counter_ts += 1
            reaction_count += 1


if __name__ == "__main__":
    print("Running scripts")
    # main()
    # checking_duplicates()
    save_reactions_with_multiple_ts(save_even_single=True)
