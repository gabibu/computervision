import importlib

import torch.nn as nn


def ActivationLayer(actionvation_cls_name: str) -> nn.Module:
    """
    Get the activation layer that is requested

    :param act_type: activation type
    :return: Activation layer
    """
    module = importlib.import_module("torch.nn")
    activation_class = getattr(module, actionvation_cls_name)
    return activation_class()
