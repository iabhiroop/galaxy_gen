# my_vae_sampler/sampler.py
import torch
from .model import VAEFlow
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def load_model(model_path, latent_dim=64, flow_hidden_dim=128, num_flows=4, device='cpu'):
    """
    Loads the VAEFlow model from the checkpoint.

    Parameters:
        model_path (str): Path to the saved model checkpoint (e.g., 'models/trained_model.pth').
        latent_dim (int): Dimension of the latent space.
        flow_hidden_dim (int): Hidden dimension for the flow.
        num_flows (int): Number of flow layers.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model (VAEFlow): The loaded model in evaluation mode.
    """
    model = VAEFlow(latent_dim=latent_dim, flow_hidden_dim=flow_hidden_dim, num_flows=num_flows)
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

def generate_samples(model, data_path="data/vae_flow_data.pkl", num_batches=1, batch_size=64):
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