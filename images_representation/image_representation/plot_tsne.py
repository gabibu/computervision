import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from image_representation.data.datasets.image_coordinates_dataset import ImageCordinatesDataset
from image_representation.entities.experiment_parameters import ExperimentParameters
from image_representation.models.model_factory import create_model
from image_representation.utils.config import get_experiment_train_configuration
from image_representation.utils.experiment_utils import get_config_file_path, get_weights_dir, \
    validate_config_file_exist
from image_representation.utils.image_loader import ImageLoader
from image_representation.utils.image_utils import get_image_coordinates
from image_representation.utils.os_utils import get_files_paths_under_dir
from image_representation.utils.torch_utils import get_device
from tqdm import tqdm

def _read_parameters() -> ExperimentParameters:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--weights_file_name", type=str)

    args = parser.parse_args()

    exp_dir = args.exp_dir
    weights_file_name = args.weights_file_name
    validate_config_file_exist(exp_dir)

    config_file_path = get_config_file_path(exp_dir)

    return weights_file_name, get_experiment_train_configuration(config_file_path, exp_dir)


def main():
    weights_file_name, experiment_parameters = _read_parameters()

    images_paths = get_files_paths_under_dir(experiment_parameters.images_dir)
    image_coordinates_creator = partial(get_image_coordinates, sidelen=experiment_parameters.side_length)

    dataset = ImageCordinatesDataset(paths=images_paths,
                                     image_loader=ImageLoader(experiment_parameters.transformations),
                                     image_coordinates_creator=image_coordinates_creator)

    dataloader = DataLoader(dataset, batch_size=experiment_parameters.batch_size,
                            pin_memory=True, shuffle=False)

    model = create_model(experiment_parameters.model_type, experiment_parameters.model_parameters)

    weights_dir = get_weights_dir(experiment_parameters.exp_dir, create_dir=True)

    model_weights_file = os.path.join(weights_dir, weights_file_name)

    model.load_state_dict(torch.load(model_weights_file))
    model.eval()

    device = get_device()

    model_outputs = []
    model.to(device)

    with torch.no_grad():

        for (coordinates, _) in tqdm(dataloader, total=len(dataloader)):

            model_input = coordinates.to(device)
            model_output, _ = model(model_input)
            outs = model_output.detach().cpu().numpy()

            for i in range(outs.shape[0]):
                model_outputs.append(outs[i].flatten())

    model_outputs = np.array(model_outputs)

    signals_embedded = TSNE(n_components=2, learning_rate='auto',
                            init='random', perplexity=3).fit_transform(model_outputs)

    tsne_df = pd.DataFrame(signals_embedded, columns=["comp1", "comp2"])

    sns.scatterplot(x="comp1", y="comp2",
                    data=tsne_df).set(title="Signals")

    plt.show()


if __name__ == "__main__":
    main()
