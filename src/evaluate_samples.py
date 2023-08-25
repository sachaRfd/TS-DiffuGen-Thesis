import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign as MA
import pandas as pd
import numpy as np
import sys


def get_paths(sample_path):
    """# noqa
    Load molecule files from a sample path and organize them into true and generated samples.

    Args:
        sample_path (str): Path to the sample directory containing molecule files.

    Returns:
        tuple: Two lists containing true and generated sample file paths.
    """

    # Check that the path exists:
    assert os.path.exists(sample_path)

    # Iterate over all the directories and load the files that end with .xyz
    number_of_molecules = []
    for directory in os.listdir(sample_path):
        dir_path = os.path.join(sample_path, directory)
        if os.path.isdir(dir_path):
            number_of_molecules.append(directory)

    # Load the files of molecules
    true_samples = []
    generated_samples = []

    for batch_dir in number_of_molecules:
        batch_dir_path = os.path.join(sample_path, batch_dir)
        for mol_dir in os.listdir(batch_dir_path):
            mol_dir_path = os.path.join(batch_dir_path, mol_dir)
            if os.path.isdir(mol_dir_path):
                files = os.listdir(mol_dir_path)
                true_samples.extend(
                    [
                        os.path.join(mol_dir_path, file)
                        for file in files
                        if "true_sample" in file
                    ]
                )
                generated_samples.append(
                    [
                        os.path.join(mol_dir_path, file)
                        for file in files
                        if "true" not in file
                    ]
                )  # Because True prod and reactant have TRUE  in their names

    return true_samples, generated_samples


def import_xyz_file(molecule_path, RMSD=False):
    """# noqa
    Import an XYZ file as an RDKit molecule object.

    Args:
        molecule_path (str): File path of the XYZ file to be imported.
        RMSD (bool, optional): True if RMSD format, False if standard XYZ format.

    Returns:
        Chem.Mol or None: RDKit molecule object or None if loading fails.
    """

    if RMSD:
        mol = Chem.rdmolfiles.MolFromXYZFile(molecule_path)
        return mol
    else:
        # Read the XYZ File:
        with open(molecule_path, "r") as xyz_file:
            lines = xyz_file.readlines()

        # Create the tuples:
        molecule = []

        # For NEW XYZ format:# Skip the first 2 lines (as it is just Number of atoms and empty line)    # noqa
        for atom in lines[2:]:
            # for atom in lines[0:]:
            # Make sure to check we are not using the hydrogens in our calculations # noqa
            if atom[0] == "H":
                continue
            else:
                # Split the string by spaces:
                atom = atom.split(" ")
                # Convert the strings to floats:
                mol = (float(atom[1]), float(atom[2]), float(atom[3]))
                molecule.append(mol)

        return molecule


def create_lists(original_path, RMSD=False):
    """# noqa
    Create lists of true and generated molecules from the given path.

    Args:
        original_path (str): Path to the original directory containing molecule files.
        RMSD (bool, optional): True if RMSD format, False if standard XYZ format.

    Returns:
        tuple: Two lists containing RDKit molecule objects.
    """
    true_paths, generated_paths = get_paths(original_path)

    # First we deal with getting the list of true samples:
    true_molecules = []  # Dtype are RDKIT molecules
    for true_path in true_paths:
        true_molecules.append(import_xyz_file(true_path, RMSD=RMSD))

    # Next the generated molecules:
    generated_molecules = []
    for sample_mol in generated_paths:
        gen_mols = []
        for generated_path in sample_mol:
            gen_mols.append(import_xyz_file(generated_path, RMSD=RMSD))
        generated_molecules.append(gen_mols)

    return true_molecules, generated_molecules


def calculate_distance_matrix(coordinates):
    """
    Calculate pairwise distance matrix from 3D coordinates.

    Args:
        coordinates (list): List of 3D coordinates for each atom.

    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    num_atoms = len(coordinates)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(
                np.array(coordinates[i]) - np.array(coordinates[j])
            )  # noqa

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def calculate_DMAE(gen_mol, true_mol):
    """
    Calculate D-MAE between inter-atomic distance matrices.

    Args:
        gen_mol (list): Inter-atomic distance matrix of generated molecule.
        true_mol (list): Inter-atomic distance matrix of true molecule.

    Returns:
        float: D-MAE value.
    """

    # Ensure both matrices have the same number of atoms
    assert len(gen_mol) == len(
        true_mol
    ), "Number of atoms in generated and true molecules must be the same."

    # Get the number of atoms
    N_atom = len(gen_mol)

    # Calculate the D-MAE
    dmae_sum = 0.0
    count = 0
    for i in range(N_atom):
        for j in range(i + 1, N_atom):
            dmae_sum += abs(gen_mol[i][j] - true_mol[i][j])
            count += 1

    # The First part of the following formula is basically just the average of all the atomic distances    # noqa

    dmae = 2.0 / (N_atom * (N_atom - 1)) * dmae_sum
    return dmae


def calculate_best_rmse(
    gen_mol,
    ref_mol,
    max_iters=100_000,
    use_hydrogens=False,
):
    """# noqa
    Calculate Best RMSD between RDKit Molecule Objects.

    Args:
        gen_mol (Chem.Mol): RDKit molecule object representing generated molecule.
        ref_mol (Chem.Mol): RDKit molecule object representing reference molecule.
        max_iters (int, optional): Maximum atom matches. Defaults to 100_000.
        use_hydrogens (bool, optional): True to include hydrogens. Defaults to False.

    Returns:
        float: Best RMSD value.
    """
    # Check that the two molecules have the same atoms:
    gen_atoms = list(atom.GetSymbol() for atom in gen_mol.GetAtoms())
    ref_atoms = list(atom.GetSymbol() for atom in ref_mol.GetAtoms())

    assert gen_atoms == ref_atoms, "Molecules are not the same"

    if use_hydrogens:
        pass
    else:
        # Remove the Hydrogens:
        gen_mol = Chem.rdmolops.RemoveAllHs(gen_mol)
        ref_mol = Chem.rdmolops.RemoveAllHs(ref_mol)

    rmsd = MA.GetBestRMS(
        gen_mol, ref_mol, maxMatches=max_iters
    )  # Was previously 10_000
    return rmsd


def create_table(true_mols, gen_mols, max_iters=1, metric="RMSD"):
    """# noqa
    Create comparison table between molecules.

    Args:
        true_mols (list): List of true molecule RDKit objects.
        gen_mols (list): List of lists containing generated molecule RDKit objects.
        max_iters (int): Maximum atom matches for RMSD calculation.
        metric (str): Metric choice, "RMSD" or "DMAE".

    Returns:
        pd.DataFrame: DataFrame of comparison metrics.
    """
    assert metric in [
        "RMSD",
        "DMAE",
    ], "Invalid metric choice. Use 'RMSD' or 'DMAE'."
    rows = []

    for true_mol, gen_sample in zip(true_mols, gen_mols):
        row = []
        if metric == "DMAE":
            true_molecule_dist_matrix = calculate_distance_matrix(true_mol)

        for gen_mol in gen_sample:  # [5:]:
            if metric == "RMSD":
                value = calculate_best_rmse(gen_mol, true_mol, max_iters)
            elif metric == "DMAE":
                generated_molecule_dist_matrix = calculate_distance_matrix(
                    gen_mol
                )  # noqa
                value = calculate_DMAE(
                    true_molecule_dist_matrix, generated_molecule_dist_matrix
                )

            row.append(value)
        rows.append(row)

    columns = [f"Sample {i+1}" for i in range(len(rows[0]))]
    column_headers = [f"{col}_{metric.upper()}" for col in columns]
    data = [[value for value in row] for row in rows]

    df = pd.DataFrame(data, columns=column_headers)
    return df


def calc_cov_mat(results_matrix, cov_threshold=0.1):
    """
    Calculate COV and MAT scores based on D-MAE matrix.

    Args:
        results_matrix (np.ndarray): D-MAE matrix.
        cov_threshold (float, optional): COV threshold. Defaults to 0.1.

    Returns:
        tuple: Calculated MAT-R mean, median, and COV-R scores.
    """

    # First we calcualte the MAT-R score:
    mat_r_mean = results_matrix.min(axis=1).mean()
    mat_r_median = results_matrix.min(axis=1).median()

    print(f"MAT Mean score is\t{mat_r_mean}")
    print(f"MAT Median score is\t{mat_r_median}")

    # Next we calclate the COV score:
    cov_r = (results_matrix.min(axis=1) < cov_threshold).mean()
    print(f"The COV score is\t{cov_r * 100} %.")  # noqa  # noqa


def evaluate(sample_path, evaluation_type, cov_threshold=0.1):
    """
    Evaluate generated samples using RMSE or D-MAE metrics.

    Args:
        sample_path (str): Path to sample directory.
        evaluation_type (str): Metric choice, "RMSD" or "DMAE".
        cov_threshold (float, optional): COV threshold. Defaults to 0.1.
    """
    assert evaluation_type in [
        "RMSD",
        "DMAE",
    ], "Please choose either RMSD or DMAE"  # noqa

    assert os.path.exists(sample_path)

    if evaluation_type == "RMSD":
        true_mols, gen_mols = create_lists(sample_path, RMSD=True)
        max_iter = 100_000  # gets slow very quickly
        table = create_table(
            true_mols=true_mols,
            gen_mols=gen_mols,
            max_iters=max_iter,
            metric="RMSD",
        )
    elif evaluation_type == "DMAE":
        true_molecules, generated_molecules = create_lists(sample_path)
        table = create_table(
            true_mols=true_molecules,
            gen_mols=generated_molecules,
            metric="DMAE",
        )
    else:
        print("Invalid evaluation type. Please choose either RMSD or DMAE.")
        return

    calc_cov_mat(table, cov_threshold=cov_threshold)


def main():
    if len(sys.argv) > 2:
        print("Usage: python script_name.py [sample_path]")
        return

    if len(sys.argv) == 2:
        sample_path = sys.argv[1]
        if not os.path.exists(sample_path):
            print("Error: Specified sample path does not exist.")
            return
    else:
        sample_path = "src/Diffusion/W93_dataset_weights/False_no_productW93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_cosine_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Samples"  # noqa

    # Sample Cov 0.1:
    cov_threshold = 0.1
    print(f"\nRMSE with cov threshold of: {cov_threshold}\n")
    evaluate(sample_path, "RMSD", cov_threshold=cov_threshold)
    print(f"\nD-MAE with cov threshold of: {cov_threshold}\n")
    evaluate(sample_path, "DMAE", cov_threshold=cov_threshold)

    # Sample Cov 0.2:
    cov_threshold = 0.2
    print(f"\nRMSE with cov threshold of: {cov_threshold}\n")
    evaluate(sample_path, "RMSD", cov_threshold=cov_threshold)
    print(f"\nD-MAE with cov threshold of: {cov_threshold}\n")
    evaluate(sample_path, "DMAE", cov_threshold=cov_threshold)


if __name__ == "__main__":
    main()
