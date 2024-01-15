from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class Action(ABC):

    @abstractmethod
    def calc(self, dataloader: DataLoader, model, epoc: int):
        pass
