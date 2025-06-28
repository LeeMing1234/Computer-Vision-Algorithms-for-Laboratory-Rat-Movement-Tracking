import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

import paddle

print("Paddle version:", paddle.__version__)
print("Is GPU available?", paddle.is_compiled_with_cuda())
print("GPU devices available:", paddle.device.cuda.device_count())

# Check name of the current CUDA device
if paddle.device.cuda.device_count() > 0:
    print("Current CUDA device:", paddle.device.cuda.get_device_name(0))

# Check CUDA version used by PaddlePaddle
print("CUDA version compiled with:", paddle.version.cuda())

print(paddle.is_compiled_with_cuda())  # Should return True
print(paddle.device.get_device())      # Should return 'gpu:0'
