import logging
import os
from typing import List

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from image_representation.actions.action import Action
from image_representation.utils.experiment_utils import get_weights_file_name


def train(dataloader: DataLoader, model, optim: Optimizer, device: str,
          weights_dir: str,
          number_of_epocs: int = 1500,
          metrics: List[Action] = None):
    for epoc in range(number_of_epocs):
        model.train()
        epoc_loss = 0.
        for (coordinates, pixels) in dataloader:
            model_input, ground_truth = coordinates.to(device), pixels.to(device)
            model_output, coords = model(model_input)

            loss = ((model_output - ground_truth) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoc_loss += loss.detach().cpu().item()

        if metrics:
            for metric in metrics:
                metric.calc(dataloader, model, epoc)

        logging.info(f"loss: epoc: {epoc} {epoc_loss}")

        model_weights_file_name = get_weights_file_name(loss, epoc)
        weight_file_name = os.path.join(weights_dir, model_weights_file_name)

        torch.save(model.state_dict(), weight_file_name)
