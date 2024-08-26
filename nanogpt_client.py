import torch

from nanogptv2 import Model, device

# Initialize the model
model = Model()

# Load the state_dict into the model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True, map_location=torch.device(device=device)))

# Make sure to move the model to the correct device if needed
model.eval()  # Set model to evaluation mode
model.to(device=device)

# Generate text
print(model.generate(max_tokens=10000))
