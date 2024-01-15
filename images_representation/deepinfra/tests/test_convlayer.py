import unittest

import torch
import torch.nn as nn
import torch.testing

from deepinfra.convblock.layers.convolutions import ConvLayer
from deepinfra.convblock.layers.entiites import *


class TestConvLayer(unittest.TestCase):

    def _weights_init_conv(self, conv_module):
        nn.init.constant_(conv_module.weight.data, 1)
        conv_module.bias.data.zero_()

    def test_non_2d(self):

        with self.assertRaises(ValueError) as context:
            ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                      padding_type=PaddingType.SAME,
                      tensor_type=TensorType.D3,
                      stride=4, separable=False)

    def test_unseparable_non_casual_same_paddings(self):

        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.SAME,
                         stride=4,
                         causal=False, separable=False)

        self._weights_init_conv(conv.conv_layer)
        tensor = torch.arange(0, 600, dtype=torch.float32).reshape(-1, 10, 10, 3)
        conv_tensor = conv(tensor)
        expected_tensor = torch.load("expected_tensors/tensor1.pt")
        assert torch.equal(conv_tensor, expected_tensor) == True

    def test_unseparable_non_casual_valid_paddings(self):
        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.VALID,
                         stride=4,
                         causal=False, separable=False)

        self._weights_init_conv(conv.conv_layer)
        tensor = torch.arange(0, 1505280, dtype=torch.float32).reshape(3, 10, 224, 224)
        conv_tensor = conv(tensor)
        expected_tensor = torch.load("expected_tensors/tensor4.pt")
        assert torch.equal(conv_tensor, expected_tensor) == True

    def test_unseparable_casual_valid_padding(self):

        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.VALID, stride=4, causal=True,
                         separable=False)
        self._weights_init_conv(conv.conv_layer)
        input = torch.arange(0, 600, dtype=torch.float32).reshape(-1, 10, 10, 3)
        conv_tensor = conv(input)
        expected_tensor = torch.load("expected_tensors/tensor2.pt")

        assert torch.equal(conv_tensor, expected_tensor) == True

    def test_unseparable_casual_same_padding(self):

        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.SAME, stride=4, causal=True,
                         separable=False)
        self._weights_init_conv(conv.conv_layer)
        input = torch.arange(0, 600, dtype=torch.float32).reshape(-1, 10, 10, 3)
        conv_tensor = conv(input)
        expected_tensor = torch.load("expected_tensors/tensor3.pt")

        assert torch.equal(conv_tensor, expected_tensor) == True

    def test_separable_casual_same_padding(self):
        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.SAME, stride=4, causal=True,
                         separable=True)

        for conv_layer in list(conv.conv_layer.children()):
            self._weights_init_conv(conv_layer)

        input = torch.arange(0, 600, dtype=torch.float32).reshape(-1, 10, 10, 3)
        conv_tensor = conv(input)
        expected_tensor = torch.load("expected_tensors/tensor5.pt")
        assert torch.equal(conv_tensor, expected_tensor) == True

    def test_separable_casual_valid_padding(self):
        conv = ConvLayer(in_channels=10, out_channels=5, kernel_size=5,
                         padding_type=PaddingType.VALID, stride=4, causal=True,
                         separable=True)

        for conv_layer in list(conv.conv_layer.children()):
            self._weights_init_conv(conv_layer)

        input = torch.arange(0, 600, dtype=torch.float32).reshape(-1, 10, 10, 3)
        conv_tensor = conv(input)
        expected_tensor = torch.load("expected_tensors/tensor6.pt")
        assert torch.equal(conv_tensor, expected_tensor) == True


if __name__ == '__main__':
    unittest.main()
