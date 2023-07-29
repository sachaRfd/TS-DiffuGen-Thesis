from setuptools import setup, find_packages

# CD to roon directory and run: python setup.py install


setup(
    name='DiffSach',
    version='0.1.0',
    author='Sacha Raffaud',
    author_email='sacha.raffaud@outlook.com',
    description='Diffusion Model for Transition State Geometry Prediction',
    url='https://github.com/schwallergroup/DiffSach',
    packages=find_packages(),  # Automatically find all packages in the project directory
)

