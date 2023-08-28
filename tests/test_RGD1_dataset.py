from data.Dataset_RGD1.RGD1_dataset_class import RGD1_TS
from src.EGNN.utils import assert_mean_zero_with_mask


def test_dataset_class_setup():
    dir = "data/Dataset_RGD1/example_samples_for_testing/Clean_Geometries"
    dataset = RGD1_TS(directory=dir)

    # Test shape of dataset samples
    for sample, node_mask in dataset:
        assert sample.shape == (16, 13)
        assert node_mask.shape[0] == 16

    # Assert that the centre of Gravity of R, TS, and P is 0:
    for sample, node_mask in dataset:
        assert_mean_zero_with_mask(
            sample[:, -3:].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -6:-3].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -9:-6].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )


def test_dataset_class_setup_without_hydrogens():
    dir = "data/Dataset_RGD1/example_samples_for_testing/Clean_Geometries"
    dataset = RGD1_TS(directory=dir, remove_hydrogens=True)

    # Test shape of dataset samples
    for sample, node_mask in dataset:
        assert sample.shape == (6, 12)
        assert node_mask.shape[0] == 6

    # Assert that the centre of Gravity of R, TS, and P is 0:
    for sample, node_mask in dataset:
        assert_mean_zero_with_mask(
            sample[:, -3:].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -6:-3].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -9:-6].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )


if __name__ == "__main__":
    print("Running script")
    test_dataset_class_setup()
    test_dataset_class_setup_without_hydrogens()
