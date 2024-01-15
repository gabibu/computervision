import torch.nn as nn
from typing import Union, Tuple
from deepinfra.convblock.layers.entiites import PaddingType, TensorType


class ConvLayer(nn.Module):
    """
    Conv layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    Causal convolution implemented accoarding to:
    from https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3, padding_type: PaddingType = PaddingType.SAME, stride: int=1,
                 dilation: Union[int, Tuple[int, int]]=1, tensor_type: TensorType = TensorType.D2, causal: bool=False,
                 groups: int=1, separable: bool=False):
        super(ConvLayer, self).__init__()

        dilation = self._make_pair_tuple_if_not_already(dilation)
        kernel_size = self._make_pair_tuple_if_not_already(kernel_size)
        stride = self._make_pair_tuple_if_not_already(stride)

        self._dilation = dilation
        self._causal = causal
        self._kernel_size = kernel_size

        if tensor_type != TensorType.D2:
            raise ValueError(f"tensor_type {tensor_type} is not supported")

        padding = self._get_padding(causal, padding_type, kernel_size, dilation)

        if not separable:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=padding, dilation=dilation, stride=stride, groups=groups)

        else:
            conv_layer = []
            conv_layer.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                            padding=padding, dilation=dilation, stride=stride, groups=in_channels))

            conv_layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1))

            self.conv_layer = nn.Sequential(*conv_layer)

    def _make_pair_tuple_if_not_already(self, value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:

        if type(value) != Tuple:
            value = (value, value)

        return value

    def _get_padding(self, causal: bool, padding_type: PaddingType, kernel_size: Tuple[int, int],
                      dilation: Tuple[int, int]) -> int:

        if causal or  padding_type == PaddingType.SAME:
            padding =  tuple(dilation[i] * ((kernel_size[i] - 1) // 2) for i in range(len(kernel_size)))
        elif padding_type == PaddingType.VALID:
            padding = (0,) * len(kernel_size)
        else:
            raise ValueError(f"padding {padding_type} is not supported")

        return padding

    def forward(self, input):

        output = self.conv_layer(input)
        if self._causal:
            output = output[..., self._dilation[-1] * (self._kernel_size[1] - 1):]

        return output
