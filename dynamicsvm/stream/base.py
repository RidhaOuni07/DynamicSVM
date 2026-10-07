"""
Base class and interface for Streaming and Incremental/Decremental SVMs.
"""

from abc import abstractmethod
from typing import Optional, Union
import numpy as np

from dynamicsvm.base import BaseSVM
from dynamicsvm.kernels import Kernel


class StreamingSVM(BaseSVM):
    """
    Abstract base class for streaming SVM models that support
    incremental learning (learn_one / learn_many) and decremental unlearning (forget_one / forget_many).
    """

    @abstractmethod
    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "StreamingSVM":
        """
        Incrementally update the SVM model with a single new incoming instance (x, y).
        """
        pass

    def learn_many(self, X: np.ndarray, y: np.ndarray) -> "StreamingSVM":
        """
        Incrementally update the SVM model with a batch of new incoming instances.
        """
        for xi, yi in zip(X, y):
            self.learn_one(xi, yi)
        return self

    @abstractmethod
    def forget_one(self, index: int) -> "StreamingSVM":
        """
        Decrementally remove/unlearn an instance from the model's active memory.
        """
        pass

    def forget_many(self, indices: np.ndarray) -> "StreamingSVM":
        """
        Decrementally remove/unlearn multiple instances from the model's active memory.
        """
        # Sort indices in reverse order to maintain index consistency if sequential
        sorted_indices = sorted(indices, reverse=True)
        for idx in sorted_indices:
            self.forget_one(idx)
        return self
