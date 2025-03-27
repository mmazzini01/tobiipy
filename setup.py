from setuptools import setup, find_packages
from setuptools.command.install import install
import os

def read_requirements():
    with open('requirements.txt') as req:
        return [line.strip() for line in req.readlines() if line.strip() and not line.startswith('#')]

setup(
    name="tobiipy",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    author="Mateo Mazzini",
    description="A Python package for processing Tobbi eye tracker data",
)

