import pandas as pd
from rdkit import Chem
import os
from data.Dataset_W93.setup_dataset_files import create_pdb_from_file_path
from tqdm import tqdm

from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdmolops
from src.Diffusion.saving_sampling_functions import write_xyz_file


"""

This script will read the reactant and product SMILES and 
output the 3D coordinates generated for them.

"""  # noqa


def smiles_to_coord(smile):
    mol = Chem.MolFromSmiles(smile)  # Read the smile
    mol = Chem.AddHs(mol)  # Add hydrogens to the mol
    rdmolops.RemoveStereochemistry(mol)
    Chem.AssignAtomChiralTagsFromStructure(
        mol,
        replaceExistingTags=True,
    )
    rdDistGeom.EmbedMolecule(
        mol,
        maxAttempts=500,
        useRandomCoords=True,
        randomSeed=42,
    )  # Generate the 3D coordinates
    # Iterate over every atom:
    list_atoms = []
    for idx, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(idx)
        coords = [
            str(atom.GetSymbol()),
            str(positions.x),
            str(positions.y),
            str(positions.z),
        ]
        list_atoms.append(coords)
    return list_atoms


if __name__ == "__main__":
    print("Running script")
    df = pd.read_csv("data/Dataset_RDKIT/data/w93_dataset/wb97xd3.csv")
    directory_path = "data/Dataset_RDKIT/data/Clean_Geometries"
    os.makedirs(directory_path, exist_ok=True)

    # Get the samples we can generate with RDKit without too
    # much issue:
    working_rows = []
    failed_rows = []
    for reaction_number in tqdm(range(df.shape[0])):
        # Create the Reaction directory
        current_reaction_path = f"data/Dataset_RDKIT/data/Clean_Geometries/Reaction_{reaction_number}"  # noqa

        # Load the smiles:
        reactant_sm, product_sm = df.iloc[reaction_number][1:3]
        try:
            # Get the Coordinates:
            reactant_coords = smiles_to_coord(reactant_sm)
            product_coords = smiles_to_coord(product_sm)

            # If it works then save all the files:
            os.makedirs(current_reaction_path, exist_ok=True)
            ts_save_path = current_reaction_path + "/TS_geometry.xyz"
            ts_file = f"data/Dataset_W93/data/TS/wb97xd3/rxn{reaction_number:06d}/ts{reaction_number:06d}.log"  # noqa
            create_pdb_from_file_path(
                path_to_raw_log=ts_file,
                path_to_final=ts_save_path,
            )
            # Real Reactant:
            real_reactant_save_path = (
                current_reaction_path + "/real_reactant_geometry.xyz"
            )
            real_reactant_file = f"data/Dataset_W93/data/TS/wb97xd3/rxn{reaction_number:06d}/r{reaction_number:06d}.log"  # noqa
            create_pdb_from_file_path(
                path_to_raw_log=real_reactant_file,
                path_to_final=real_reactant_save_path,
            )
            # Real Product:
            real_product_save_path = (
                current_reaction_path + "/real_product_geometry.xyz"
            )
            real_product_file = f"data/Dataset_W93/data/TS/wb97xd3/rxn{reaction_number:06d}/p{reaction_number:06d}.log"  # noqa
            create_pdb_from_file_path(
                path_to_raw_log=real_product_file,
                path_to_final=real_product_save_path,
            )

            # Setup the Reactant and Product Paths:
            reactant_save_path = (
                current_reaction_path + "/Reactant_geometry.xyz"
            )  # noqa
            product_save_path = current_reaction_path + "/Product_geometry.xyz"

            # # Save the R and P:
            write_xyz_file(data=reactant_coords, filename=reactant_save_path)
            write_xyz_file(data=product_coords, filename=product_save_path)
            working_rows.append(reaction_number)

        except Exception:
            failed_rows.append(reaction_number)

    print("Rows with failed coordinate loading:", len(failed_rows))

    # Create a new DataFrame with only the rows that worked
    working_df = df.iloc[working_rows]

    # Save the filtered DataFrame to a new CSV file
    path = "data/Dataset_RDKIT/data/w93_dataset/working_reactions.csv"
    working_df.to_csv(
        path,
        index=False,
    )
    print("Saved working reactions CSV file")
