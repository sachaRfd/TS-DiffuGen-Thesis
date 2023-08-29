from setuptools import setup, find_packages


"""
CD to root directory and run: python setup.py install
"""


setup(
    name="TS-DiffuGen",
    version="1.0.0",
    author="Sacha Raffaud",
    author_email="sacha.raffaud@outlook.com",
    description="Diffusion Model for Transition State Geometry Prediction",
    url="https://github.com/schwallergroup/TS-DiffuGen",
    packages=find_packages(),
)
