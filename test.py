# example_usage.py
import torch
import matplotlib.pyplot as plt
from galaxy_gen.sampler import load_model, generate_samples

# Path to your saved model checkpoint.
model_path = 'models/vae_flow_latent64_hidden128_flows4_2Best.pth'
device = 'cpu'  # or 'cuda' if you have a GPU

# Load the model.
model = load_model(model_path, device=device)

# Generate random samples.
samples = generate_samples(model)

# (Optional) Visualize the samples.
samples = samples.cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i][0], cmap='gray')
    ax.axis('off')
plt.show()
