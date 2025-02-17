# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
import urllib.request
import os

class CustomInstallCommand(install):
    def run(self):
        # Call the standard install process first
        install.run(self)
        
        # URLs of the models to download
        model_urls = [
            'https://huggingface.co/Abhiroop174/galaxy_gen/formationtime_model.pth',
            'https://huggingface.co/Abhiroop174/galaxy_gen/metallicity_model.pth'
        ]
        
        # Directory to save the models
        data_path = 'models'
        model_dir = data_path = os.path.join(os.path.dirname(__file__), data_path)
        
        # Download the models
        for url in model_urls:
            file_name = url.split('/')[-1]
            urllib.request.urlretrieve(url, f'{model_dir}/{file_name}')
        
        print("Models downloaded successfully.")

setup(
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    },
)