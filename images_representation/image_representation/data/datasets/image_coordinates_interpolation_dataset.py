from typing import List, Callable, Tuple

import numpy as np
from torch.utils.data import Dataset


class ImageCordinatesDataset(Dataset):

    def __init__(self, paths: List[str],
                 pairs: List[Tuple[int, int]],
                 image_loader: Callable, image_coordinates_creator: Callable):
        super().__init__()
        self._paths = paths
        self._image_loader = image_loader
        self._image_coordinates_creator = image_coordinates_creator
        self._pairs = pairs

        self._dataset = [(self._image_loader(self._paths[image_id1]),
                          self._image_loader(self._paths[image_id2]),
                          self._get_interpolated_coordinates(image_id1, image_id2)) for (image_id1, image_id2) in
                         self._pairs]

    def _get_interpolated_coordinates(self, image1, image2):
        image1_coordinates = self._image_coordinates_creator(image_id=image1)
        image2_coordinates = self._image_coordinates_creator(image_id=image2)

        image1_indexes = np.random.choice([False, True], size=image1_coordinates.shape[0])

        final_coordinates = np.zeros(shape=image1_coordinates.shape)
        final_coordinates[image1_indexes, :] = image1_coordinates[image1_indexes, :]
        image2_indexes = np.invert(image1_indexes)
        final_coordinates[image2_indexes, :] = image2_coordinates[image2_indexes, :]

        return final_coordinates

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        pixels1, pixels2, coords = self._dataset[idx]
        return pixels1, pixels2, coords
