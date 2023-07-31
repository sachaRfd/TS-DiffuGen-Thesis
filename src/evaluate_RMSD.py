"""
Script for COV and MAT evaluation using RMSD between point clouds: 
------------------------------------------------------------------

1. Requires RDKIT optimisation algorithm to best align the two point clouds
2. Calculate COVerage Score: How much of the generated samples are actually close to the true reference   # noqa
3. Calculate MATching Score: How good are the samples (Lowest RMSD)
"""


import os
from rdkit import Chem
from rdkit.Chem import rdMolAlign as MA
import pandas as pd


def get_paths(sample_path):
    """
    Load the files of molecules from a given sample path and organize them into true and generated samples.  # noqa
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


def import_xyz_file(molecule_path, true_samples=True):
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
    if (
        true_samples
    ):  # True Samples are if the files were saved with previous function, which does not put number cout on the top of the file  # noqa
        with open(molecule_path, "r") as xyz_file:
            lines = xyz_file.readlines()

        # Get the number of initial lines
        num_initial_lines = int(len(lines))
        lines = "".join([str(num_initial_lines), "\n\n"] + lines)

        # Load the molecule
        mol = Chem.MolFromXYZBlock(lines)
        if mol is None:
            print("Error: Failed to load molecule files.")
            exit()

    else:
        mol = Chem.rdmolfiles.MolFromXYZFile(molecule_path)

    return mol


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
    # Get the paths
    true_paths, generated_paths = get_paths(original_path)

    # List for true samples:
    true_molecules = []
    for true_path in true_paths:
        true_molecules.append(import_xyz_file(true_path, true_samples=True))

    # List for generated samples:
    generated_molecules = []
    for sample_mol in generated_paths:
        gen_mols = []
        for generated_path in sample_mol:
            gen_mols.append(import_xyz_file(generated_path, true_samples=True))
        generated_molecules.append(gen_mols)

    return true_molecules, generated_molecules


def calculate_best_rmse(gen_mol, ref_mol, max_iters=100_000):
    """# noqa
    Calculate the Best RMSD (Root Mean Square Deviation) between two RDKit Molecule Objects.

    The function removes hydrogen atoms from both molecules before computing the RMSD.
    It uses the maximum number of atom matches specified by 'max_iters' to find the best RMSD value.

    Args:
        gen_mol (Chem.Mol): RDKit molecule object representing the generated molecule.
        ref_mol (Chem.Mol): RDKit molecule object representing the reference (target) molecule.
        max_iters (int, optional): The maximum number of atom matches to consider when computing RMSD.
            Defaults to 100_000.

    Returns:
        float: The calculated Best RMSD value between the two input molecules.

    Note:
        The Best RMSD value represents the minimum RMSD among all possible alignments of the two molecules.
        The function assumes that both 'gen_mol' and 'ref_mol' have already removed hydrogen atoms.
    """

    # Remove the Hydrogens:
    gen_mol = Chem.rdmolops.RemoveAllHs(gen_mol)
    ref_mol = Chem.rdmolops.RemoveAllHs(ref_mol)

    rmsd = MA.GetBestRMS(
        gen_mol, ref_mol, maxMatches=max_iters
    )  # Was previously 10_000
    return rmsd


def create_rmse_table(true_mols, gen_mols, max_iters):
    """# noqa
    Create a table of Best RMSD values between the reference and generated samples.

    This function calculates the Best RMSD (Root Mean Square Deviation) for each generated sample
    compared to the corresponding reference (true) molecule. The results are organized in a table format.

    Args:
        true_mols (list): A list of RDKit molecule objects representing the true (reference) samples.
        gen_mols (list): A list of lists, each inner list containing RDKit molecule objects representing
            the generated samples corresponding to a specific true sample.
        max_iters (int): The maximum number of atom matches to consider when computing RMSD.

    Returns:
        pd.DataFrame: A Pandas DataFrame representing the table of Best RMSD values.
            The rows of the DataFrame correspond to true samples, and columns represent each generated sample.

    Note:
        The function assumes that both 'true_mols' and 'gen_mols' have already removed hydrogen atoms.
        The Best RMSD value represents the minimum RMSD among all possible alignments of two molecules.
        The generated samples should be organized in the same order as the true samples for accurate comparison.
    """
    rows = []
    for true_mol, gen_sample in zip(true_mols, gen_mols):
        row = []
        for gen_mol in gen_sample:
            rmsd = calculate_best_rmse(gen_mol, true_mol, max_iters)
            row.append(rmsd)
        rows.append(row)

    df = pd.DataFrame(
        rows, columns=[f"Sample {i+1}" for i in range(len(rows[0]))]
    )  # noqa
    return df


def calc_cov_mat(rmse_matrix, cov_threshold=0.1):
    """# noqa
    Calculate the COV and MAT scores of the inputted RMSE (Root Mean Square Error) matrix.

    COV (Coverage) Score:
    - COV represents the percentage of each generated molecule that has an RMSE score smaller than a threshold.
    - In this case, the threshold is defined as 'cov_threshold' (default is 0.1 Å).

    MAT (Minimum Among The Best) Score:
    - MAT is the sum of the minimum RMSE values for each sample in the input matrix.
    - It represents the total discrepancy between each generated sample and its closest true sample.

    Args:
        rmse_matrix (numpy.ndarray or pandas.DataFrame): The input matrix containing RMSE values.
            Rows represent true samples, and columns represent generated samples.
        cov_threshold (float, optional): The RMSE threshold for calculating the COV score. Defaults to 0.1.

    Returns:
        None: This function prints the calculated MAT-R (MAT-Row) mean and median scores, as well as the COV-R (COV-Row) score.

    """
    # First we calcualte teh MAT-R score:
    mat_r_mean = rmse_matrix.min(axis=1).mean()
    mat_r_median = rmse_matrix.min(axis=1).median()

    print(f"MAT-R Mean score is\t{mat_r_mean}")
    print(f"MAT-R Median score is\t{mat_r_median}")

    # Next we calclate the COV score:
    cov_r = (rmse_matrix.min(axis=1) < cov_threshold).mean()
    print(
        f"The COV-R score with a threshold of\t{cov_threshold}\tis\t{cov_r * 100} %."  # noqa
    )  # noqa

    return None


if __name__ == "__main__":
    print("Running Evaluation Script\n")

    sample_path = "src/Diffusion/Clean_lightning/BEST_MODEL_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_2000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Samples_3"  # noqa

    true_mols, gen_mols = create_lists(sample_path)

    # Get the number of iterations for the best alignmenet algorithm:
    max_iter = 100_000  # gets quite slow very quickly

    table = create_rmse_table(
        true_mols=true_mols, gen_mols=gen_mols, max_iters=max_iter
    )
    print(table)
    calc_cov_mat(table, cov_threshold=0.1)
