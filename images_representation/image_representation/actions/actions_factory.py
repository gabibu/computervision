from typing import List, Dict

from image_representation.actions.action import Action
from image_representation.actions.action_type import ActionType
from image_representation.actions.plotting import PlotMetric
from image_representation.utils.torch_utils import get_device


def create_actions(actions_configs: List[Dict]) -> List[Action]:
    return [create_action(action_config) for action_config in actions_configs]


def create_action(action_config: Dict) -> Action:
    metric_type = ActionType[action_config["type"]]

    if metric_type == ActionType.PLOT:
        return PlotMetric(**action_config["params"], device=get_device())
    else:
        raise ValueError(f"action {metric_type} is not supported")
