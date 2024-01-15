import torch.nn as nn
from deepinfra.convblock.layers.entiites import *

class PoolLayer(nn.Module):
    """
    Pooling layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """
    def __init__(self, pool_stride: int =2, pool_kernel: int=2, pool_type: LayersTypes=LayersTypes.MAX,
                 tensor_type: TensorType=TensorType.D2):
        super(PoolLayer, self).__init__()

        if tensor_type == TensorType.D2:
            if pool_type == LayersTypes.AVG:
                self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride= pool_stride)
            elif pool_type == LayersTypes.MAX:
                self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
            else:
                raise ValueError(f"pool_type {pool_type} is not supported")
        else:
            raise ValueError(f"tensor_type {tensor_type} is not supported")

    def forward(self, input):
        output = self.pool(input)
        return output
