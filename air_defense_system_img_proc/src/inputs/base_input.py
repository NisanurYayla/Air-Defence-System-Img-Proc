from abc import ABC, abstractmethod
import numpy as np


class BaseInput(ABC):
    """
    Tüm input kaynakları için soyut sınıf.
    """

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def read(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def resize(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
