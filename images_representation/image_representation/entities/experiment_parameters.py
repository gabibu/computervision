from dataclasses import dataclass
from typing import Dict, List, Callable

from image_representation.actions.action import Action
from image_representation.entities.model_type import ModelType


@dataclass
class ExperimentParameters:
    number_of_epocs: int
    plot_images_epocs_interval: int
    model_type: ModelType
    optim_name: str
    images_dir: str
    side_length: int
    batch_size: int
    shuffle: bool
    model_parameters: Dict
    optimizer_parameters: Dict
    metrics: List[Action]
    transformations: Callable
    exp_dir: str
