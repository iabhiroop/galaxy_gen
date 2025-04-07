# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
import os

setup(
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'galaxy_gen': []
    }
)