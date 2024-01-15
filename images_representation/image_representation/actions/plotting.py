import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from image_representation.actions.action import Action


class PlotMetric(Action):

    def __init__(self, plot_epoc_interval: int, device: str):
        self._plot_epoc_interval = plot_epoc_interval
        self._device = device

    def calc(self, dataloader: DataLoader, model, epoc: int):
        model.eval()

        (coordinates, pixels) = next(iter(dataloader))

        model_input, ground_truth = coordinates.to(self._device), pixels.to(self._device)

        with torch.no_grad():
            model_output, coords = model(model_input)

        if epoc % self._plot_epoc_interval == 0:
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            axes[0].imshow(ground_truth[0].cpu().view(48, 48, 3).detach().numpy())
            axes[1].imshow(model_output[0].cpu().view(48, 48, 3).detach().numpy())
            plt.show()
