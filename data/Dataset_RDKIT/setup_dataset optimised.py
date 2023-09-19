import pandas as pd
from rdkit import Chem
import os
from data.Dataset_W93.setup_dataset_files import (
    create_pdb_from_file_path,
    extract_geometry,
)
from tqdm import tqdm

from src.evaluate_samples import calculate_distance_matrix, calculate_DMAE

import numpy as np

from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from src.Diffusion.saving_sampling_functions import write_xyz_file


"""

This script will read the reactant and product SMILES and 
output the 3D coordinates generated for them.

"""  # noqa


def smiles_to_coord_optimised(smile):
    mol = Chem.MolFromSmiles(smile, sanitize=True)  # Read the smile
    mol = Chem.AddHs(mol)  # Add hydrogens to the mol

    # Get Initial Coordinates
    AllChem.EmbedMolecule(
        mol,
        maxAttempts=2_000,
        useRandomCoords=True,
        randomSeed=42,
    )
    # Optimise the Coordinate Generation:
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2_000)

    # Iterate over every atom:
    list_atoms = []
    list_coords = []
    for idx, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(idx)
        coords = [
            str(atom.GetSymbol()),
            str(positions.x),
            str(positions.y),
            str(positions.z),
        ]
        coords_only = [
            (positions.x),
            (positions.y),
            (positions.z),
        ]
        list_coords.append(coords_only)
        list_atoms.append(coords)
    return list_atoms, np.array(list_coords)


if __name__ == "__main__":
    print("Running script")
    df = pd.read_csv("data/Dataset_RDKIT/data/w93_dataset/wb97xd3.csv")
    directory_path = "data/Dataset_RDKIT/data/Optimised_Geometries"
    os.makedirs(directory_path, exist_ok=True)

    # Get the samples we can generate with RDKit without too
    # much issue:
    working_rows = []
    failed_rows = []
    reactant_dmaes = []
    product_dmaes = []
    for reaction_number in tqdm(range(df.shape[0])):
        # Create the Reaction directory
        current_reaction_path = f"data/Dataset_RDKIT/data/Optimised_Geometries/Reaction_{reaction_number}"  # noqa

        # Load the smiles:
        reactant_sm, product_sm = df.iloc[reaction_number][1:3]
        try:
            # Get the Coordinates:
            reactant_coords, reactant_coords_only = smiles_to_coord_optimised(
                reactant_sm
            )
            product_coords, product_coords_only = smiles_to_coord_optimised(
                product_sm,
            )

            # # Get the real reactant:
            # real_reactant_file = f"data/Dataset_W93/data/TS/wb97xd3/rxn{reaction_number:06d}/r{reaction_number:06d}.log"  # noqa
            # with open(real_reactant_file, "r") as file:
            #     logs = file.read().splitlines()

            # atoms, reac_coordinates = extract_geometry(logs)

            # # Get the distance matrices:
            # optimised_dist_mat = calculate_distance_matrix(
            #     reactant_coords_only,
            # )
            # true_dist_mat = calculate_distance_matrix(reac_coordinates)

            # dmae_reac = calculate_DMAE(optimised_dist_mat, true_dist_mat)
            # reactant_dmaes.append(dmae_reac)

            # # Get the real reactant:
            # prod_reactant_file = f"data/Dataset_W93/data/TS/wb97xd3/rxn{reaction_number:06d}/p{reaction_number:06d}.log"  # noqa
            # with open(prod_reactant_file, "r") as file:
            #     logs = file.read().splitlines()

            # atoms, prod_coordinates = extract_geometry(logs)

            # # Get the distance matrices:
            # optimised_dist_mat_prod = calculate_distance_matrix(
            #     product_coords_only,
            # )
            # true_dist_mat_prod = calculate_distance_matrix(prod_coordinates)

            # dmae_prod = calculate_DMAE(
            #     optimised_dist_mat_prod,
            #     true_dist_mat_prod,
            # )
            # product_dmaes.append(dmae_prod)

        except Exception as e:
            print(reaction_number)
            print(e)
            print()
            print(reactant_sm)
            print()
            print(product_sm)
            print()
            exit()

            failed_rows.append(reaction_number)

    reactant_dmaes = np.array(reactant_dmaes)
    product_dmaes = np.array(product_dmaes)
    print(reactant_dmaes.mean())
    print(product_dmaes.mean())
    print(len(failed_rows))
