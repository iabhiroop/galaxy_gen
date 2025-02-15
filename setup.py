# setup.py
from setuptools import setup, find_packages

setup(
    name='galaxy_gen',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        # add any other dependencies here
    ],
    author='Your Name',
    description='A library to generate random samples from a trained VAEFlow model.',
)
