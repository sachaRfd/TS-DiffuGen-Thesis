import numpy as np
from src.evaluate_samples import (
    get_paths,
    import_xyz_file,
    create_lists,
    calculate_distance_matrix,
    calculate_DMAE,
    calculate_best_rmse,
    create_table,
    evaluate,
)
from rdkit import Chem


"""
Testing the Evaluation functions
"""

path_to_samples = "src/Diffusion/Weights_and_Samples/W93_Dataset/True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_cosine_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Example_Samples"  # noqa

example_mol_path = "src/Diffusion/Weights_and_Samples/W93_Dataset/True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_cosine_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Example_Samples/batch_0/mol_2/sample_0.xyz"  # noqa
example_mol_path_2 = "src/Diffusion/Weights_and_Samples/W93_Dataset/True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_cosine_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Example_Samples/batch_0/mol_2/true_sample.xyz"  # noqa
example_mol_path_wrong = "src/Diffusion/Weights_and_Samples/W93_Dataset/True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_cosine_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Example_Samples/batch_0/mol_3/true_sample.xyz"  # noqa


example_mol_coordinates = np.array([(1, 2), (2, 1)])
ref_dist_matrix = np.array([[0.0, 1.41421356], [1.41421356, 0.0]])


def test_get_paths():
    # Test if you input correct path it works:
    paths_true, paths_generated = get_paths(path_to_samples)

    assert len(paths_true) == len(paths_generated) == 5  # five examples
    assert len(paths_generated[0]) == 2  # Two examples per mol

    # Test if you input WRONG path it gives error:
    try:
        wrong_path = "src/Incorrect_Path"
        paths_true, paths_generated = get_paths(wrong_path)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass  # This is expected


def test_imports_RMSD():
    mol = import_xyz_file(molecule_path=example_mol_path, RMSD=True)
    assert isinstance(mol, Chem.Mol), "Molecule should be an RDKit Mol object"


def test_imports_DMAE():
    mol = import_xyz_file(molecule_path=example_mol_path, RMSD=False)
    assert len(mol) == 7
    for tup in mol:
        assert len(tup) == 3


def test_create_list_RMSD():
    # Load the paths to check their numbers:
    paths_true, paths_generated = get_paths(path_to_samples)

    true_mols, gen_mols = create_lists(
        original_path=path_to_samples,
        RMSD=True,
    )

    # Assert Same shapes
    assert len(paths_true) == len(true_mols)
    assert len(paths_generated) == len(gen_mols)
    for i in range(len(paths_true)):
        assert len(paths_generated[i]) == len(gen_mols[i])

    # Assert that all true_mols are RdKit Mol Objects
    for mol in true_mols:
        assert isinstance(mol, Chem.Mol)

    # Assert that all gen_mols are RdKit Mol Objects
    for i in range(len(paths_true)):
        for mol in gen_mols[i]:
            assert isinstance(mol, Chem.Mol)


def test_create_list_DMAE():
    # Load the paths to check their numbers:
    paths_true, paths_generated = get_paths(path_to_samples)

    true_mols, gen_mols = create_lists(
        original_path=path_to_samples,
        RMSD=False,
    )

    # Assert Same shapes
    assert len(paths_true) == len(true_mols)
    assert len(paths_generated) == len(gen_mols)
    for i in range(len(paths_true)):
        assert len(paths_generated[i]) == len(gen_mols[i])


def test_Distance_Matrix():
    # Import the file into tuple format:
    mol = import_xyz_file(molecule_path=example_mol_path, RMSD=False)
    dist_matrix = calculate_distance_matrix(mol)

    # assert the diags are zero
    for element in dist_matrix.diagonal():
        assert element == 0

    # assert ypu have a symmetrical matrix;
    assert np.allclose(dist_matrix, dist_matrix.T)

    dist_matrix_test = calculate_distance_matrix(example_mol_coordinates)
    dist_matrix_test_Transpose = calculate_distance_matrix(
        example_mol_coordinates.T
    )  # noqa

    assert np.allclose(dist_matrix_test, ref_dist_matrix)
    assert np.allclose(dist_matrix_test_Transpose, ref_dist_matrix)


def test_DMAE():
    gen_mol = import_xyz_file(molecule_path=example_mol_path, RMSD=False)
    true_mol = import_xyz_file(molecule_path=example_mol_path_2, RMSD=False)

    gen_dist_matrix = calculate_distance_matrix(gen_mol)
    true_dist_matrix = calculate_distance_matrix(true_mol)

    dmae = calculate_DMAE(gen_mol=gen_dist_matrix, true_mol=true_dist_matrix)
    assert np.isclose(0.0861773405015555, dmae)

    # Test wrong molecule
    wrong_mol = import_xyz_file(
        molecule_path=example_mol_path_wrong,
        RMSD=False,
    )
    wrong_dist_matrix = calculate_distance_matrix(wrong_mol)
    try:
        dmae = calculate_DMAE(
            gen_mol=wrong_dist_matrix,
            true_mol=true_dist_matrix,
        )
    except AssertionError:
        pass  # This is expected


def test_Best_RMSD():
    gen_mol = import_xyz_file(molecule_path=example_mol_path, RMSD=True)
    true_mol = import_xyz_file(molecule_path=example_mol_path_2, RMSD=True)

    rmse_without_h = calculate_best_rmse(
        gen_mol=gen_mol,
        ref_mol=true_mol,
        max_iters=1,
    )
    assert np.isclose(rmse_without_h, 0.23228288775636208)
    rmse_with_h = calculate_best_rmse(
        gen_mol=gen_mol,
        ref_mol=true_mol,
        max_iters=1,
        use_hydrogens=True,
    )
    assert np.isclose(rmse_with_h, 0.6805777150802539)
    assert rmse_with_h > rmse_without_h

    # Assert if you give wrong molecule pair it gives error:
    wrong_mol = import_xyz_file(
        molecule_path=example_mol_path_wrong,
        RMSD=True,
    )

    try:
        rmse_without_h = calculate_best_rmse(
            gen_mol=wrong_mol,
            ref_mol=true_mol,
            max_iters=1,
        )
    except AssertionError:
        pass  # This is expected


def test_RMSD_table():
    # Load samples:
    true_mols, gen_mols = create_lists(path_to_samples, RMSD=True)
    max_iter = 1
    table = create_table(
        true_mols=true_mols,
        gen_mols=gen_mols,
        max_iters=max_iter,
        metric="RMSD",
    )

    assert table.shape == (5, 2)
    return None


def test_DMAE_table():
    # Load samples:
    true_mols, gen_mols = create_lists(path_to_samples, RMSD=False)
    max_iter = 1
    table = create_table(
        true_mols=true_mols,
        gen_mols=gen_mols,
        max_iters=max_iter,
        metric="DMAE",
    )
    assert table.shape == (5, 2)


def test_table_wrong_metric():
    # Load samples:
    true_mols, gen_mols = create_lists(path_to_samples, RMSD=False)
    max_iter = 1

    try:
        _ = create_table(
            true_mols=true_mols,
            gen_mols=gen_mols,
            max_iters=max_iter,
            metric="dmae",
        )
    except AssertionError:
        pass  # This is expected


def test_evaluate():
    evaluate(sample_path=path_to_samples, evaluation_type="RMSD")
    evaluate(sample_path=path_to_samples, evaluation_type="DMAE")

    # Check if you give wrong evaluation_type you get error:
    try:
        evaluate(sample_path=path_to_samples, evaluation_type="rmsd")
    except AssertionError:
        pass

    # Check wrong path:
    try:
        evaluate(sample_path=path_to_samples + "x", evaluation_type="RMSD")
    except AssertionError:
        pass


if __name__ == "__main__":
    test_get_paths()
    test_imports_RMSD()
    test_imports_DMAE()
    test_create_list_RMSD()
    test_create_list_DMAE()
    test_Distance_Matrix()
    test_DMAE()
    test_Best_RMSD()
    test_RMSD_table()
    test_DMAE_table()
    test_table_wrong_metric()
    test_evaluate()
