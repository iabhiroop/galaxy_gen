# example_usage.py
import torch
import matplotlib.pyplot as plt
from galaxy_gen.sampler import load_model, generate_metallicity_samples, generate_formationtime_samples

# Path to your saved model checkpoint.
model_path = 'models/formationtime_model.pth'
device = 'cpu'  # or 'cuda' if you have a GPU

# Load the model.
model = load_model("formation_time",model_path, device=device)

# Generate random samples.
samples = generate_formationtime_samples(model)

# (Optional) Visualize the samples.
samples = samples.cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i][0])
    ax.axis('off')
plt.show()
