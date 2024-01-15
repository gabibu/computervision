from typing import Callable

from PIL import Image


class ImageLoader:

    def __init__(self, transformations: Callable):
        self._transformations = transformations

    def __call__(self, image_path: str, convert_mode: str = "RGB"):
        image = Image.open(image_path).convert(convert_mode)
        return self._transformations(image).permute(1, 2, 0).reshape(-1, 1)
