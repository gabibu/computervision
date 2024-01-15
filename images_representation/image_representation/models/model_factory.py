from typing import Dict

from image_representation.entities.model_type import ModelType
from image_representation.models.siren.model import Siren


def create_model(model_type: ModelType, model_parametsr: Dict):
    if model_type == ModelType.Siren:

        return Siren(**model_parametsr)
    else:
        raise ValueError(f"{model_type} is not supported")
