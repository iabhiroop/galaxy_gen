# my_vae_sampler/sampler.py
import torch
from .samplemodel import VAEFlow
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from .metallicityModel import AutoregressiveVAE
from .foramtiontimeModel import VAEWithResidualFlow_Gray
from .massesModel import VAEFlow_mass
from .gammaflowModel import VAEWithResidualFlow_galaxy

def get_model_class(model_name, latent_dim, hidden_dim, num_flows):
    """
    Returns the appropriate model class based on the model name.

    Parameters:
        model_name (str): The name of the model.

    Returns:
        model_class (class): The corresponding model class.
    """
    if model_name == 'formation_time':
        if latent_dim is None or hidden_dim is None or num_flows is None:
            latent_dim = 256
            hidden_dim = 128
            num_flows = 5
        return VAEWithResidualFlow_Gray, latent_dim, hidden_dim, num_flows
    elif model_name == 'metallicity':
        if latent_dim is None or hidden_dim is None or num_flows is None:
            latent_dim = 512
            hidden_dim = 256
            num_flows = 4
        return AutoregressiveVAE, latent_dim, hidden_dim, num_flows 
    elif model_name == 'masses':
        if latent_dim is None or hidden_dim is None or num_flows is None:
            latent_dim = 256
            hidden_dim = 128
            num_flows = 5
        return VAEFlow_mass, latent_dim, hidden_dim, num_flows
    elif model_name == 'galaxy':
        if latent_dim is None or hidden_dim is None or num_flows is None:
            latent_dim = 256
            hidden_dim = 128
            num_flows = 5
            return VAEWithResidualFlow_galaxy, latent_dim, hidden_dim, num_flows
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def load_model(model_name, model_path, latent_dim=None, hidden_dim=None, num_flows=None, device='cpu'):
    """
    Loads the specified model from the checkpoint.

    Parameters:
        model_name (str): The name of the model to load.
        model_path (str): Path to the saved model checkpoint (e.g., 'models/trained_model.pth').
        latent_dim (int): Dimension of the latent space.
        hidden_dim (int): Hidden dimension for the flow.
        num_flows (int): Number of flow layers.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model: The loaded model in evaluation mode.
    """
    model_class, latent_dim, hidden_dim, num_flows = get_model_class(model_name,latent_dim=latent_dim, hidden_dim=hidden_dim, num_flows=num_flows)
    model = model_class(latent_dim=latent_dim, hidden_dim=hidden_dim, num_flows=num_flows)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_sample_model(model_path, latent_dim=64, hidden_dim=128, num_flows=4, device='cpu'):
    """
    Loads the VAEFlow model from the checkpoint.

    Parameters:
        model_path (str): Path to the saved model checkpoint (e.g., 'models/trained_model.pth').
        latent_dim (int): Dimension of the latent space.
        hidden_dim (int): Hidden dimension for the flow.
        num_flows (int): Number of flow layers.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model (VAEFlow): The loaded model in evaluation mode.
    """

    model = VAEFlow(latent_dim=latent_dim, hidden_dim=hidden_dim, num_flows=num_flows)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_random_samples(model, num_samples=64):
    """
    Generates random samples using the model's sample method.

    Parameters:
        model (VAEFlow): The trained model.
        num_samples (int): Number of samples to generate.

    Returns:
        samples (Tensor): A batch of generated samples.
    """
    return model.sample(num_samples)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def generate_samples(model, data_path="data/sample_data.pkl", num_batches=1, batch_size=64):
    """
    Generate samples using the trained model from data in a pickle file.
    
    Args:
        model: trained VAE model
        data_path: path to the pickle file containing the data
        num_batches: number of batches to use for generation
        batch_size: size of each batch
    
    Returns:
        torch.Tensor: Generated samples
    """
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create DataLoader
    data = normalize_data(data)
    data = torch.tensor(data).float().unsqueeze(1)
    dataset = TensorDataset(torch.tensor(data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.cpu()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = batch[0].cpu()
            recon_batch, _, _ = model(batch)
            return recon_batch
    
    return None

def generate_metallicity_samples(model, data_path="data/metallicity_data.pkl", num_batches=1, batch_size=64):
    """
    Generate metallicity samples using the trained model from data in a pickle file.
    
    Args:
        model: trained MetallicityVAE model
        data_path: path to the pickle file containing the data
        num_batches: number of batches to use for generation
        batch_size: size of each batch
    
    Returns:
        torch.Tensor: Generated metallicity samples
    """
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create DataLoader
    data = normalize_data(data)
    data = torch.tensor(data).float().unsqueeze(1)
    dataset = TensorDataset(torch.tensor(data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.cpu()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = batch[0].cpu()
            recon_batch, _, _, _ = model(batch)
            return recon_batch
    
    return None

def generate_formationtime_samples(model, data_path="data/formationtime_data.pkl", num_batches=1, batch_size=64):
    """
    Generate formation time samples using the trained model from data in a pickle file.
    
    Args:
        model: trained VAE model
        data_path: path to the pickle file containing the data
        num_batches: number of batches to use for generation
        batch_size: size of each batch
    
    Returns:
        torch.Tensor: Generated formation time samples
    """
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create DataLoader
    data = normalize_data(data)
    data = torch.tensor(data).float().unsqueeze(1)
    dataset = TensorDataset(torch.tensor(data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.cpu()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = batch[0].cpu()
            recon_batch, _, _ = model(batch)
            return recon_batch
    
    return None


def generate_masses_samples(model, data_path="data/mass_data.pkl", num_batches=1, batch_size=64):
    """
    Generate formation time samples using the trained model from data in a pickle file.
    
    Args:
        model: trained VAE model
        data_path: path to the pickle file containing the data
        num_batches: number of batches to use for generation
        batch_size: size of each batch
    
    Returns:
        torch.Tensor: Generated formation time samples
    """
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create DataLoader
    # data = normalize_data(data)
    print(data.shape)
    # data = torch.tensor(data).float().unsqueeze(1)
    dataset = TensorDataset(torch.tensor(data))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.cpu()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = batch[0].cpu()
            recon_batch, _, _, _ = model(batch)
            return recon_batch
    
    return None

def generate_galaxy_samples(model, data_path="data/galaxy_data.pkl", num_batches=1, batch_size=64):
    """
    Generate formation time samples using the trained model from data in a pickle file.
    
    Args:
        model: trained VAE model
        data_path: path to the pickle file containing the data
        num_batches: number of batches to use for generation
        batch_size: size of each batch
    
    Returns:
        torch.Tensor: Generated formation time samples
    """
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)
    # Load data from pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = TensorDataset(data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.cpu()
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            batch = batch[0].cpu()
            recon_batch, _, _ = model(batch)
            recon_batch = recon_batch.cpu().numpy()
            return recon_batch
    
    return None