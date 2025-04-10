from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch

def load_model(path):
    # Initialize the model
    model = Unet(
        dim=32,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    )

    # Initialize the diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,           # number of steps
        sampling_timesteps=250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    # Load the saved model weights
    model.load_state_dict(torch.load(path))
    # Set the model to evaluation mode
    model.eval()

    return diffusion