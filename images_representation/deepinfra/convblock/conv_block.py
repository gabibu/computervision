from typing import List, Optional

import torch.nn as nn

from deepinfra.convblock.layers.activations import ActivationLayer
from deepinfra.convblock.layers.convolutions import ConvLayer
from deepinfra.convblock.layers.pooling import PoolLayer
from functools import partial
from deepinfra.convblock.layers.entiites import *

POOL_LAYER_TYPES = {LayersTypes.AVG, LayersTypes.MAX}


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 op_list: List[LayersTypes],
                 kernel_size: int = 3,
                 padding_type: PaddingType = PaddingType.SAME,
                 stride: int = 1,
                 dilation: int = 1,
                 tensor_type: TensorType = TensorType.D2,
                 causal: bool = False,
                 groups: int = 1,
                 activation: str = 'ReLU',
                 separable: bool = False,
                 pool_stride: int = 2,  #
                 pool_kernel: int = 2,
                 dropout: float = 0.0,
                 upsample: Optional[int] = None,
                 conv_init: str = 'kaiming',
                 bn_init: str = 'default'):

        super(ConvBlock, self).__init__()

        if tensor_type != TensorType.D2:
            raise ValueError(f"tensor_type {tensor_type} is not supported")

        dim = tensor_type.value
        kernel_size = (kernel_size,) * dim if type(kernel_size) is not tuple else kernel_size
        dilation = (dilation,) * dim if type(dilation) is not tuple else dilation
        stride = (stride,) * dim if type(stride) is not tuple else stride

        collected_layers = []
        had_conv_layer = False
        for layer_type in op_list:

            if layer_type == LayersTypes.CONV:

                had_conv_layer = True
                collected_layers.append(ConvLayer(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding_type=padding_type, stride=stride,
                                             dilation=dilation, tensor_type=tensor_type,
                                             causal=causal, groups=groups,
                                             separable=separable))  # assumes the stride and kernel are the same size

            elif layer_type in POOL_LAYER_TYPES:
                collected_layers.append(
                    PoolLayer(pool_stride, pool_kernel, pool_type=layer_type, tensor_type=tensor_type))

            elif layer_type == LayersTypes.BN:
                bn_channels = out_channels if had_conv_layer else in_channels
                collected_layers.append(nn.BatchNorm2d(bn_channels))
            elif layer_type == LayersTypes.ACTIV:
                collected_layers.append(ActivationLayer(activation))
            elif layer_type == LayersTypes.DROP:
                collected_layers.append(nn.Dropout2d(p=dropout))
            elif layer_type == LayersTypes.UP:

                if upsample is None:
                    raise ValueError(f"upsample cant be null when there is {layer_type} layer")
                collected_layers.append(nn.Upsample(scale_factor=upsample))

        self.layers = nn.Sequential(*collected_layers)

        weights_init_callable = partial(self.init_weights,
                                        conv_init = conv_init, activation = activation, bn_init = bn_init,
                                        op_list = op_list)
        self.apply(weights_init_callable)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, m, conv_init: str, activation: str, bn_init: str, op_list: List[LayersTypes]):

        if isinstance(m, nn.Conv2d):
            if conv_init == 'kaiming':
                if activation in {'relu', "elu", "lrelu"} and LayersTypes.ACTIV in op_list:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight)
            elif conv_init == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif conv_init == 'zeros':
                nn.init.constant_(m.weight, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if bn_init == 'ones':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif bn_init == 'zeros':
                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        elif (isinstance(m, ConvLayer)):
            weights_init_callable = partial(self.init_weights,
                                            conv_init=conv_init, activation=activation, bn_init=bn_init,
                                            op_list=op_list)
            m.conv_layer.apply(weights_init_callable)



if __name__ == "__main__":
    ConvBlock(in_channels=10, out_channels=5,
              op_list=[LayersTypes.CONV, LayersTypes.BN, LayersTypes.ACTIV])
