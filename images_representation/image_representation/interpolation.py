import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import more_itertools
import torch
from torch.utils.data import DataLoader

from image_representation.data.datasets.image_coordinates_interpolation_dataset import ImageCordinatesDataset
from image_representation.entities.experiment_parameters import ExperimentParameters
from image_representation.models.model_factory import create_model
from image_representation.utils.config import get_experiment_train_configuration
from image_representation.utils.experiment_utils import get_config_file_path, get_weights_dir, \
    validate_config_file_exist
from image_representation.utils.image_loader import ImageLoader
from image_representation.utils.image_utils import get_image_coordinates
from image_representation.utils.os_utils import get_files_paths_under_dir
from image_representation.utils.torch_utils import get_device


def _read_parameters() -> ExperimentParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--weights_file_name", type=str)
    parser.add_argument("--pairs_csv", type=str)

    args = parser.parse_args()

    exp_dir = args.exp_dir
    weights_file_name = args.weights_file_name
    pairs_csv = args.pairs_csv
    images_ids_strs = pairs_csv.split(",")

    images_ids = map(int, images_ids_strs)
    pairs = more_itertools.chunked(images_ids, 2)

    validate_config_file_exist(exp_dir)

    config_file_path = get_config_file_path(exp_dir)

    return pairs, weights_file_name, get_experiment_train_configuration(config_file_path, exp_dir)


def main():
    pairs, weights_file_name, experiment_parameters = _read_parameters()

    images_paths = get_files_paths_under_dir(experiment_parameters.images_dir)
    image_coordinates_creator = partial(get_image_coordinates, sidelen=experiment_parameters.side_length)

    dataset = ImageCordinatesDataset(paths=images_paths,
                                     pairs=pairs,
                                     image_loader=ImageLoader(experiment_parameters.transformations),
                                     image_coordinates_creator=image_coordinates_creator,
                                     )

    dataloader = DataLoader(dataset, batch_size=1,
                            pin_memory=True, shuffle=False)

    model = create_model(experiment_parameters.model_type, experiment_parameters.model_parameters)
    weights_dir = get_weights_dir(experiment_parameters.exp_dir, create_dir=True)
    model_weights_file = os.path.join(weights_dir, weights_file_name)
    model.load_state_dict(torch.load(model_weights_file))
    model.eval()
    device = get_device()
    model.to(device)

    with torch.no_grad():
        for (image1, image2, coordinates) in dataloader:
            coordinates = coordinates.float().to(device)
            model_output, _ = model(coordinates)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(image1[0].cpu().view(experiment_parameters.side_length, experiment_parameters.side_length,
                                                3).detach().numpy())
            axes[1].imshow(image2[0].cpu().view(experiment_parameters.side_length, experiment_parameters.side_length,
                                                3).detach().numpy())

            axes[2].imshow(
                model_output[0].cpu().view(experiment_parameters.side_length, experiment_parameters.side_length,
                                           3).detach().numpy())

            axes[0].set_title("Image - 1")
            axes[1].set_title("Image - 2")
            axes[2].set_title("Interpolation Signal")

    plt.show()


if __name__ == "__main__":
    main()
