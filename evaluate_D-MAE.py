"""
Script for COV and MAT evaluation using INTER-ATOMIC DISTANCES:
---------------------------------------------------------------
"""

import os
import pandas as pd
import numpy as np


def get_paths(sample_path):
    """# noqa

    Load the files of molecules from a given sample path and organize them into true and generated samples.
        1. Iterate over one of the batch directories
        2. Iteraet over one of the molecule directories
        3. Get the number of samples we have
        4. Create a list of the true samples
        5. Iterate over all of the files in the current directory and add the true transition states to the true samples list
        6. Do the same but create a list of lists where in the inner list you have all the generated molecules for the specific true sample

    Args:
        sample_path (str): The path to the sample directory containing molecule files.

    Returns:
        tuple: A tuple containing two lists:
            - true_samples (list): A list of file paths representing true transition states.
            - generated_samples (list): A list of lists, each inner list containing file paths of generated molecules
              corresponding to a specific true sample.

    Raises:
        AssertionError: If the specified sample_path does not exist.
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

    # Print The number of true samples and generated samples
    print(
        f"\tThere are {len(true_samples)} samples and {len(generated_samples[0])} generated samples in total.\n\tThere are {int(len(generated_samples[0])/len(true_samples))} generated samples per true sample.\n"  # noqa
    )

    return true_samples, generated_samples


def import_xyz_file(molecule_path):
    """# noqa

    Imports an XYZ file and loads it into an RDKit molecule object.

    Args:
        molecule_path (str): The file path of the XYZ file to be imported.
        true_samples (bool, optional): Whether the file was saved using a previous function that omits
            the number count on the top. Defaults to True.

    Returns:
        Chem.Mol or None: An RDKit molecule object representing the loaded molecule,
        or None if the molecule could not be loaded.

    Note:
        If true_samples is True, the function reads the file, calculates the number of initial lines,
        and adds this count to the top of the file before loading the molecule (Needed for RDKit).
    """
    # Read the XYZ File:
    with open(molecule_path, "r") as xyz_file:
        lines = xyz_file.readlines()

    # Create the tuples:
    molecule = []

    # For NEW XYZ format:# Skip the first 2 lines (as it is just Number of atoms and empty line)    # noqa
    for atom in lines[2:]:
        # for atom in lines[0:]:
        # Make sure to check we are not using the hydrogens in our calculations
        if atom[0] == "H":
            # print(atom[0])
            continue
        else:
            # Split the string by spaces:
            atom = atom.split(" ")
            # Convert the strings to floats:
            mol = (float(atom[1]), float(atom[2]), float(atom[3]))
            molecule.append(mol)

    return molecule


def create_lists(original_path):
    """# noqa

    Create lists of true and generated molecules from the given original path.

    This function utilizes helper functions to extract true and generated molecule paths,
    and then loads the corresponding XYZ files into RDKit molecule objects.

    Args:
        original_path (str): The path to the original directory containing molecule files.

    Returns:
        tuple: A tuple containing two lists:
            - true_molecules (list): A list of RDKit molecule objects representing true transition states.
            - generated_molecules (list): A list of lists, each inner list containing RDKit molecule objects
              representing generated molecules corresponding to a specific true sample.

    Note:
        The function relies on the 'get_paths' function to obtain the paths of true and generated molecules.
        The 'import_xyz_file' function is used to read and load the XYZ files into RDKit molecules.
        True molecules are assumed to have been saved with the number count on the top, whereas generated
        molecules are assumed to have been saved using the function that omits the number count on the top.
    """
    true_paths, generated_paths = get_paths(original_path)

    # First we deal with getting the list of true samples:
    true_molecules = []  # Dtype are RDKIT molecules
    for true_path in true_paths:
        true_molecules.append(import_xyz_file(true_path))

    # Next the generated molecules:
    generated_molecules = []
    for sample_mol in generated_paths:
        gen_mols = []
        for generated_path in sample_mol:
            gen_mols.append(import_xyz_file(generated_path))
        generated_molecules.append(gen_mols)

    return true_molecules, generated_molecules


def calculate_distance_matrix(coordinates):
    """# noqa

    Calculate the pairwise distance matrix from a list of 3D coordinates.

    This function takes a list of 3D coordinates (x, y, z) for each atom and computes
    the pairwise distances between all atoms. The distance matrix is a symmetric matrix,
    where the element at (i, j) represents the distance between atom i and atom j.

    Parameters:
        coordinates (list): List of 3D coordinates (x, y, z) for each atom.

    Returns:
        np.ndarray: The distance matrix, which is a 2D NumPy array of shape (num_atoms, num_atoms).
            The element at position (i, j) in the array contains the distance between atom i and atom j.
            The matrix is symmetric, so distance_matrix[i, j] == distance_matrix[j, i].
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
    """# noqa

    Calculate the D-MAE (Difference Mean Absolute Error) between the inter-atomic distance matrices
    of the generated molecule and the true molecule, following the approach in the TS-DIFF paper.

    D-MAE is a metric used for comparing the models, specifically for inter-atomic distance predictions.

    Parameters:
        gen_mol (list): The inter-atomic distance matrix of the generated molecule.
            It should be a symmetric 2D list or array representing the pairwise distances between atoms.
        true_mol (list): The inter-atomic distance matrix of the true molecule.
            It should also be a symmetric 2D list or array representing the pairwise distances between atoms.

        Both matrices must have the same number of atoms for a valid comparison.

    Returns:
        float: The D-MAE value, representing the difference mean absolute error between the two distance matrices.

    Properties:
        - The inter-atomic distance matrices are assumed to be symmetric matrices, where the element at (i, j)
          represents the distance between atom i and atom j, and is equal to the distance between atom j and atom i.

    Calculation:
        D-MAE is calculated using the following formula:
        D-MAE = 2 / (N_atom * (N_atom - 1)) * Sum for i < j of the absolute difference in interatomic distance matrix

        Where:
        - N_atom: The total number of atoms in the molecule.
        - Sum for i < j: A summation performed over all unique pairwise combinations of atoms.
        - gen_mol[i][j]: The inter-atomic distance between atom i and atom j in the generated molecule.
        - true_mol[i][j]: The inter-atomic distance between atom i and atom j in the true molecule.

    Note:
        D-MAE measures the average absolute difference between the inter-atomic distances of the two matrices.
        A lower D-MAE value indicates a better match between the generated and true molecules in terms of atomic distances.
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


def create_dmae_table(true_mols, gen_mols):
    """# noqa

    Create a table of Difference Mean Absolute Error (D-MAE) values between reference and generated samples.

    This function calculates the D-MAE for each generated molecule compared to the reference molecule and
    creates a table to summarize the results. The D-MAE measures the average absolute difference in inter-atomic
    distances between two distance matrices, and it is used to assess the similarity of molecules.

    Parameters:
        true_mols (list): A list of reference molecules, each represented by an inter-atomic distance matrix.
            Each distance matrix should be a symmetric 2D list or array, representing the pairwise distances
            between atoms in the corresponding molecule.
        gen_mols (list): A list of lists, where each inner list contains generated molecules to compare against
            the reference molecules. Each generated molecule should also be represented by an inter-atomic distance
            matrix in the same format as the reference molecules.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the D-MAE values for each generated sample with respect to
            the reference molecules. The table will have rows corresponding to each reference molecule and columns
            corresponding to each generated sample. The (i, j)-th element in the DataFrame will represent the D-MAE
            between the i-th reference molecule and the j-th generated sample.

    Note:
        The input molecules are expected to be represented as inter-atomic distance matrices, where the element at
        (i, j) in the matrix represents the distance between atom i and atom j, and is equal to the distance
        between atom j and atom i. The D-MAE is a useful metric for assessing the accuracy of generated molecular
        structures compared to the reference molecules.
    """
    rows = []
    # ZIP the true molecule and a list of generated samples to compare
    for true_mol, gen_sample in zip(true_mols, gen_mols):
        # Calculate the distance matrix of the true_molecules:
        true_molecule_dist_matrix = calculate_distance_matrix(true_mol)

        row = []
        for gen_mol in gen_sample:
            # Calculate the distance matrix of the generated molecule:
            generated_molecule_dist_matrix = calculate_distance_matrix(gen_mol)

            # Calculate the D-MAE between the true sample and the generated samples:    # noqa

            dmae = calculate_DMAE(
                true_mol=true_molecule_dist_matrix,
                gen_mol=generated_molecule_dist_matrix,
            )
            row.append(dmae)
        rows.append(row)

    df = pd.DataFrame(
        rows, columns=[f"Sample {i+1}" for i in range(len(rows[0]))]
    )  # noqa

    return df


def calc_cov_mat(dmae_matrix, cov_threshold=0.1):
    """
    Calculate the COV and MAT scores based on the input D-MAE matrix.    # noqa


    COV (Coverage):
    - The COV is the percentage of each generated molecule that has a D-MAE score smaller than a threshold.
      In this function, the threshold is set to Epsilon (Eps) = 0.1 Å (Angstroms).

    MAT (Minimum Average D-MAE):
    - The MAT score is the sum of the minimum D-MAE for each sample, calculated as the minimum value along each row
      of the D-MAE matrix.

    Parameters:
        dmae_matrix (np.ndarray or pd.DataFrame): The D-MAE matrix containing D-MAE scores for generated molecules.
            Each row represents a generated sample, and each column contains the D-MAE scores for different samples.
        cov_threshold (float, optional): The threshold value to determine the COV. The default value is 0.1 Å.

    Returns:
        tuple: A tuple containing the calculated MAT-R (Minimum Average D-MAE) mean and median scores, and the COV-R score.

    Notes:
        - The D-MAE matrix should have generated samples as rows and different samples as columns.
        - MAT-R Mean: The mean value of the minimum D-MAE scores across all generated samples.
        - MAT-R Median: The median value of the minimum D-MAE scores across all generated samples.
        - COV-R: The percentage of generated samples that have a D-MAE score smaller than the specified threshold.
    """

    # First we calcualte the MAT-R score:
    mat_r_mean = dmae_matrix.min(axis=1).mean()
    mat_r_median = dmae_matrix.min(axis=1).median()

    print(f"MAT-R Mean score is\t{mat_r_mean}")
    print(f"MAT-R Median score is\t{mat_r_median}")

    # Next we calclate the COV score:
    cov_r = (dmae_matrix.min(axis=1) < cov_threshold).mean()
    print(
        f"The COV-R score with a threshold of\t{cov_threshold}\tis\t{cov_r * 100} %."  # noqa
    )  # noqa

    return None


if __name__ == "__main__":
    print("Running Evaluation Script\n")
    sample_path = "src_coords_graphs/Diffusion/weights_and_samples/True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Samples_3"  # noqa

    # Generates the true and generated molcules:
    true_molecules, generated_molecules = create_lists(sample_path)

    # Create table of D-MAE:
    table = create_dmae_table(
        true_mols=true_molecules, gen_mols=generated_molecules
    )  # noqa
    print(table)

    # Get the COVerage and MATching scores:
    calc_cov_mat(table, cov_threshold=0.2)
