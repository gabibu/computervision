from typing import List, Callable

from torch.utils.data import Dataset


class ImageCordinatesDataset(Dataset):

    def __init__(self, paths: List[str],
                 image_loader: Callable, image_coordinates_creator: Callable, preload_images: bool = True):
        super().__init__()
        self._paths = paths

        self._image_loader = image_loader
        self._image_coordinates_creator = image_coordinates_creator

        if preload_images:
            self._dataset = [(self._image_loader(path), self._image_coordinates_creator(image_id=image_id)) for
                             (image_id, path) in enumerate(self._paths)]
        else:
            self._dataset = None

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):

        if self._dataset is not None:
            pixels, coords = self._dataset[idx]
        else:
            path = self._paths[idx]
            pixels = self._image_loader(path)
            coords = self._image_coordinates_creator(image_id=idx)

        return coords, pixels
