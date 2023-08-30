# Sacha Raffaud sachaRfd and acse-sr1022

from src.Diffusion.saving_sampling_functions import write_xyz_file, return_xyz  # noqa
import torch
import os
import shutil

"""

Test the noising functions

"""


# Dictionary of encoding for testing
ohe_dict_with_hydrogen = {
    "C": [1, 0, 0, 0],
    "N": [0, 1, 0, 0],
    "O": [0, 0, 1, 0],
    "H": [0, 0, 0, 1],
}

input_tensor = torch.tensor(
    [
        [
            [1.0000, 0.0000, 0.0000, 0.0000, -2.3290, -1.6433, 0.3418],
            [1.0000, 0.0000, 0.0000, 0.0000, -1.0032, -1.5711, 0.3895],
            [0.0000, 0.0000, 1.0000, 0.0000, -0.2625, -1.0158, -0.6041],
            [1.0000, 0.0000, 0.0000, 0.0000, 1.0754, -0.6866, -0.2617],
            [1.0000, 0.0000, 0.0000, 0.0000, 1.1923, 0.5518, 0.6177],
            [1.0000, 0.0000, 0.0000, 0.0000, 0.7544, 1.8527, -0.0037],
            [1.0000, 0.0000, 0.0000, 0.0000, 0.1027, 2.0068, -1.1526],
            [0.0000, 0.0000, 0.0000, 1.0000, -2.8693, -2.1425, 1.1377],
            [0.0000, 0.0000, 0.0000, 1.0000, -2.8892, -1.2212, -0.4868],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.6298, 0.3898, 1.5496],
            [0.0000, 0.0000, 0.0000, 1.0000, -0.4302, -1.9925, 1.2199],
            [0.0000, 0.0000, 0.0000, 1.0000, 1.5501, -1.5429, 0.2418],
            [0.0000, 0.0000, 0.0000, 1.0000, 1.5881, -0.5289, -1.2153],
            [0.0000, 0.0000, 0.0000, 1.0000, 2.2451, 0.6428, 0.9208],
            [0.0000, 0.0000, 0.0000, 1.0000, 1.0144, 2.7402, 0.5749],
            [0.0000, 0.0000, 0.0000, 1.0000, -0.2100, 1.1613, -1.7593],
            [0.0000, 0.0000, 0.0000, 1.0000, -0.1590, 2.9994, -1.5102],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ]
    ]
)
expected_tensor = [
    [
        "C",
        "-2.3289999961853027",
        "-1.6433000564575195",
        "0.3418000042438507",
    ],
    [
        "C",
        "-1.0032000541687012",
        "-1.5710999965667725",
        "0.3894999921321869",
    ],
    [
        "O",
        "-0.26249998807907104",
        "-1.0157999992370605",
        "-0.6040999889373779",
    ],
    [
        "C",
        "1.0753999948501587",
        "-0.6866000294685364",
        "-0.26170000433921814",
    ],
    [
        "C",
        "1.1922999620437622",
        "0.551800012588501",
        "0.6176999807357788",
    ],
    [
        "C",
        "0.7544000148773193",
        "1.8526999950408936",
        "-0.003700000001117587",
    ],
    [
        "C",
        "0.10270000249147415",
        "2.0067999362945557",
        "-1.1526000499725342",
    ],
    [
        "H",
        "-2.86929988861084",
        "-2.1424999237060547",
        "1.1376999616622925",
    ],
    [
        "H",
        "-2.88919997215271",
        "-1.2211999893188477",
        "-0.4867999851703644",
    ],
    [
        "H",
        "0.629800021648407",
        "0.3898000121116638",
        "1.5496000051498413",
    ],
    [
        "H",
        "-0.4302000105381012",
        "-1.9924999475479126",
        "1.2199000120162964",
    ],
    [
        "H",
        "1.5500999689102173",
        "-1.5428999662399292",
        "0.241799995303154",
    ],
    [
        "H",
        "1.5880999565124512",
        "-0.5289000272750854",
        "-1.2152999639511108",
    ],
    [
        "H",
        "2.2451000213623047",
        "0.642799973487854",
        "0.920799970626831",
    ],
    [
        "H",
        "1.0144000053405762",
        "2.7402000427246094",
        "0.5748999714851379",
    ],
    [
        "H",
        "-0.20999999344348907",
        "1.1612999439239502",
        "-1.7592999935150146",
    ],
    [
        "H",
        "-0.1589999943971634",
        "2.9993999004364014",
        "-1.510200023651123",
    ],
]


def test_return_function():
    output_tensor = return_xyz(
        input_tensor,
        ohe_dictionary=ohe_dict_with_hydrogen,
        remove_hydrogen=False,
    )

    # Assert that the hydrogens have been removed:
    assert len(output_tensor) == 17
    # Assert the Correct values are given:
    assert output_tensor == expected_tensor


def test_write_file_saved():
    # Create temporary directory
    temp_dir = "TEST_DIRECTORY_TO_DELETE/"
    os.mkdir(temp_dir)

    try:
        # Read in a sample:
        output_tensor = return_xyz(
            input_tensor,
            ohe_dictionary=ohe_dict_with_hydrogen,
            remove_hydrogen=False,
        )

        # Write file to temp directory:
        example_filename_with_xyz = "Example_filename.xyz"
        write_xyz_file(output_tensor, temp_dir + example_filename_with_xyz)  # noqa

        # Check in the temporary direcotry that the file is present:
        assert os.path.exists(os.path.join(temp_dir, example_filename_with_xyz))  # noqa

        # Check that if there is not .xyz at the end of the file it is added to it  # noqa
        example_filename_without_xyz = "Example_filename_witout"
        write_xyz_file(output_tensor, temp_dir + example_filename_without_xyz)  # noqa
        assert os.path.exists(
            os.path.join(temp_dir, example_filename_without_xyz + ".xyz")
        )

    finally:
        # Delete temporary directory as it was just used for testing purposes
        shutil.rmtree(temp_dir)


def test_write_file_format():
    # Create temporary directory
    temp_dir = "TEST_DIRECTORY_TO_DELETE/"
    os.mkdir(temp_dir)

    try:
        # Read in a sample:
        output_tensor = return_xyz(
            input_tensor,
            ohe_dictionary=ohe_dict_with_hydrogen,
            remove_hydrogen=False,
        )

        # Write file to temp directory:
        example_filename_with_xyz = "Example_filename.xyz"
        write_xyz_file(output_tensor, temp_dir + example_filename_with_xyz)  # noqa

        # Check in the temporary direcotry that the file is present:
        assert os.path.exists(os.path.join(temp_dir, example_filename_with_xyz))  # noqa

        # Check that the first line has the number of atoms in the file:
        with open(
            os.path.join(temp_dir, example_filename_with_xyz), "r"
        ) as file:  # noqa
            first_line = file.readline().strip()
            assert first_line == "17"

            # Check that the next line is empty
            assert file.readline().strip() == ""

            # Check that the 3rd line contains 4 strings
            third_line = file.readline().strip().split()
            assert len(third_line) == 4

    finally:
        # Delete temporary directory as it was just used for testing purposes
        shutil.rmtree(temp_dir)
    return None


if __name__ == "__main__":
    print("Running Tests on Savings script")
    # pytest.main()
    # test_return_function()
    # test_write_file_saved()
    # test_write_file_format()
