import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from image_representation.data.datasets.image_coordinates_dataset_with_upsampling import ImageCordinatesDataset
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
    parser.add_argument("--upsample_size", type=int)
    parser.add_argument("--weights_file_name", type=str)

    args = parser.parse_args()

    exp_dir = args.exp_dir
    validate_config_file_exist(exp_dir)

    config_file_path = get_config_file_path(exp_dir)

    return args.weights_file_name, args.upsample_size, get_experiment_train_configuration(config_file_path, exp_dir)


def main():
    model_weighs_file_name, upsample_size, experiment_parameters = _read_parameters()

    sidelength = experiment_parameters.side_length

    images_paths = get_files_paths_under_dir(experiment_parameters.images_dir)
    image_coordinates_creator = partial(get_image_coordinates, sidelen=experiment_parameters.side_length)
    upsample_image_coordinates_creator = partial(get_image_coordinates, sidelen=upsample_size)

    dataset = ImageCordinatesDataset(paths=images_paths,
                                     image_loader=ImageLoader(experiment_parameters.transformations),
                                     image_coordinates_creator=image_coordinates_creator,
                                     upsample_image_coordinates_creator=upsample_image_coordinates_creator)

    dataloader = DataLoader(dataset, batch_size=1,
                            pin_memory=True, shuffle=False)

    model = create_model(experiment_parameters.model_type, experiment_parameters.model_parameters)

    weights_dir = get_weights_dir(experiment_parameters.exp_dir, create_dir=True)

    model_weights_file = os.path.join(weights_dir, model_weighs_file_name)

    model.load_state_dict(torch.load(model_weights_file))
    model.eval()

    device = get_device()

    model.to(device)

    with torch.no_grad():
        for (original_coordinates, upsampled_coordinates, ground_truth) in dataloader:
            original_coordinates = original_coordinates.to(device)
            upsampled_coordinates = upsampled_coordinates.to(device)

            model_output_original, _ = model(original_coordinates)
            model_output_upsampled, _ = model(upsampled_coordinates)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(ground_truth[0].cpu().view(sidelength, sidelength, 3).detach().numpy())
            axes[1].imshow(model_output_original[0].cpu().view(sidelength, sidelength, 3).detach().numpy())
            axes[2].imshow(model_output_upsampled[0].cpu().view(upsample_size, upsample_size, 3).detach().numpy())
            axes[0].set_title("Image")
            axes[1].set_title("Original Size Signal")
            axes[2].set_title("Upsample Signal")
            plt.show()


if __name__ == "__main__":
    main()
