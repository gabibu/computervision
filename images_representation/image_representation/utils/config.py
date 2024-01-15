import yaml

from image_representation.actions.actions_factory import create_actions
from image_representation.entities.experiment_parameters import ExperimentParameters
from image_representation.entities.model_type import ModelType
from image_representation.utils.torch_utils import create_transformations


def get_experiment_train_configuration(config_file_path: str, exp_dir: str) -> ExperimentParameters:
    with open(config_file_path, 'r') as file:
        experiment_config = yaml.safe_load(file)

        metrics = create_actions(experiment_config["actions"])

        transformations_configs = experiment_config["transformations"]

        transformations = create_transformations(transformations_configs)

        return ExperimentParameters(number_of_epocs=experiment_config["number_of_epocs"],
                                    plot_images_epocs_interval=experiment_config["plot_images_epocs_interval"],
                                    model_type=ModelType[experiment_config["model"]],
                                    optim_name=experiment_config["optim"],
                                    images_dir=experiment_config["images_dir"],
                                    side_length=experiment_config["side_length"],
                                    batch_size=experiment_config["batch_size"],
                                    shuffle=experiment_config["shuffle"],
                                    model_parameters=experiment_config["model_parameters"],
                                    optimizer_parameters=experiment_config["optimizer_parameters"],
                                    metrics=metrics,
                                    transformations=transformations,
                                    exp_dir=exp_dir)
