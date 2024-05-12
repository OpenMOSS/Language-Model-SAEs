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

import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
# List all GPUs and their details
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"GPU {i}: Name - {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPUs are available")
# Check current GPU memory usage
if cuda_available:
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1e9} GB")
        print(f"  Cached memory: {torch.cuda.memory_reserved(i) / 1e9} GB")

