import argparse
from functools import partial

import numpy as np
from torch.utils.data import DataLoader

from image_representation.data.datasets.image_coordinates_dataset import ImageCordinatesDataset
from image_representation.entities.experiment_parameters import ExperimentParameters
from image_representation.models.model_factory import create_model
from image_representation.train import train
from image_representation.utils.config import get_experiment_train_configuration
from image_representation.utils.experiment_utils import validate_config_file_exist, get_config_file_path, \
    get_weights_dir
from image_representation.utils.image_loader import ImageLoader
from image_representation.utils.image_utils import get_image_coordinates
from image_representation.utils.os_utils import get_files_paths_under_dir
from image_representation.utils.torch_utils import get_optimizer, get_device


def _read_parameters() -> ExperimentParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    args = parser.parse_args()
    print(args)

    exp_dir = args.exp_dir

    validate_config_file_exist(exp_dir)

    config_file_path = get_config_file_path(exp_dir)

    return get_experiment_train_configuration(config_file_path, exp_dir)


def main():
    parameters = _read_parameters()
    images_paths = get_files_paths_under_dir(parameters.images_dir)
    np.random.shuffle(images_paths)
    image_coordinates_creator = partial(get_image_coordinates, sidelen=parameters.side_length)

    # todo: remive 0:10
    dataset = ImageCordinatesDataset(paths=images_paths,
                                     image_loader=ImageLoader(parameters.transformations),
                                     image_coordinates_creator=image_coordinates_creator)

    dataloader = DataLoader(dataset, batch_size=parameters.batch_size,
                            pin_memory=True, shuffle=parameters.shuffle)

    model = create_model(parameters.model_type, parameters.model_parameters)

    optim = get_optimizer(parameters.optim_name, model, parameters.optimizer_parameters)

    weights_dir = get_weights_dir(parameters.exp_dir, create_dir=True)

    train(dataloader=dataloader, model=model, optim=optim,
          device=get_device(), number_of_epocs=parameters.number_of_epocs,
          metrics=parameters.metrics,
          weights_dir=weights_dir)


if __name__ == "__main__":
    main()
