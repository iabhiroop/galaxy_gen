# galaxy_gen

`galaxy_gen` is a library to generate galaxy data/distributions. This package implements the Flow-Augmented VAE architecture tailored for generating and reconstructing galaxy spectra. It provides a clean, modular interface for training, evaluating, and sampling from models, and supports configuration for different flow types (RealNVP, Autoregressive, Residual).

## Installation

You can install the package using pip:

```sh
pip install galaxy_gen
```

## Usage
Here is an example of how to use the galaxy_gen library:

```python
# example_usage.py
import torch
import matplotlib.pyplot as plt
import galaxy_gen
from galaxy_gen.sampler import load_model, generate_samples
import os

# Path to your saved model checkpoint.
model_path = os.path.join(os.path.dirname(galaxy_gen.__file__), 'models/sample_model')
device = 'cpu'  # or 'cuda' if you have a GPU

# Load the model.
model = load_sample_model(model_path, device=device)

# Generate random samples.
samples = generate_samples(model)

# (Optional) Visualize the samples.
samples = samples.cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i][0], cmap='gray')
    ax.axis('off')
plt.show()
```

Another expample to use the pre-trained model
```python
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

```

## License
This project is licensed under the MIT License - see the LICENSE file for details.