# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
import urllib.request
import os

def clone_and_copy():
    repo_url = "https://huggingface.co/Abhiroop174/galaxy_gen"
    clone_dir = "/tmp/repo"
    model_dir = "galaxy_gen/model"
    print("Cloning the repository and copying model files...")
    # Clone the repository
    os.system(f"git clone {repo_url} {clone_dir}")

    # Copy files from the cloned repository to the model directory
    for file_name in os.listdir(clone_dir):
        full_file_name = os.path.join(clone_dir, file_name)
        if os.path.isfile(full_file_name):
            os.system(f"cp {full_file_name} {model_dir}")
    # Remove the cloned repository
    os.system(f"rm -rf {clone_dir}")
    print("Model files copied successfully")

class CustomInstallCommand(install):
    def run(self):
        clone_and_copy()
        install.run(self)

setup(
    cmdclass={
        'install': CustomInstallCommand,
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'galaxy_gen': []
    }
)