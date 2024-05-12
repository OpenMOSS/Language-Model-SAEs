# import torch

# # Load the checkpoint
# checkpoint = torch.load('/remote-home/share/research/mechinterp/qwen1.5-1B-dictionary/results/L3-en-2.5e-05/checkpoints/final.pt', map_location=torch.device('cpu'))

# # Check the type of the loaded checkpoint, it's usually a dictionary
# print(type(checkpoint))

# # If it's a dictionary, you can print the keys to understand its structure
# if isinstance(checkpoint, dict):
#     print(checkpoint.keys())

#     # To check the content of a specific key
#     for key in checkpoint.keys():
#         print(f"Content of key '{key}':")
#         print(checkpoint[key])

#         # If the content is a tensor, you can print its size
#         if isinstance(checkpoint[key], torch.Tensor):
#             print(f"Size of the tensor '{key}': {checkpoint[key].size()}")

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process a list of numbers.")

# Add the argument
parser.add_argument('numbers', type=float, nargs='+', help='A list of numbers')

# Parse the arguments
args = parser.parse_args()

# Now args.numbers will be a list of floats
print(args.numbers)
