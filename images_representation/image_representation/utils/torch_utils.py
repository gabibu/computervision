import importlib
from enum import Enum
from typing import Dict, List

import torch
from torchvision.transforms import Compose


class ModelType(Enum):
    Siren = 1


def get_optimizer(optimizer_name: str, model, optimizer_parameters: Dict) -> torch.optim.Optimizer:
    if optimizer_name == "Adam":
        return torch.optim.Adam(**optimizer_parameters, params=model.parameters())
    else:
        raise ValueError(f"optimizer {optimizer_name} is not supported")


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_transformations(transformations_configs: List[Dict]):
    transformations = [create_transformation(transformation_config) for transformation_config in
                       transformations_configs]

    return Compose(transformations)

    # transformations = Compose([
    #     Resize(parameters.side_length),
    #     ToTensor(),
    #     Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    # ])


def create_transformation(transformation_config: Dict):
    module = importlib.import_module('torchvision.transforms')
    transformation_cls = getattr(module, transformation_config["type"])
    params = transformation_config["params"] if "params" in transformation_config else {}
    return transformation_cls(**params)

    # transformations = Compose([
    #     Resize(parameters.side_length),
    #     ToTensor(),
    #     Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    # ])
