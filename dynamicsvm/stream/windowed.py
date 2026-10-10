"""
Sliding Window Support Vector Machine for streaming data.
"""

from collections import deque
from typing import Callable, Optional, Union
import numpy as np

from dynamicsvm.kernels import Kernel
from dynamicsvm.models.svc import SVC
from dynamicsvm.stream.base import StreamingSVM
from dynamicsvm.utils.validation import check_array


class SlidingWindowSVM(StreamingSVM):
    """
    Sliding Window SVM for online data streams.
    Maintains a maximum window size W. Automatically handles incremental insertion
    of new instances and decremental eviction of outdated instances.
    
    Parameters
    ----------
    window_size : int, default=200
        Maximum number of samples in the active window.
    C : float, default=1.0
        Regularization parameter.
    kernel : str, Kernel or callable, default='rbf'
        Kernel type.
    solver : str, default='smo'
        Dual solver ('smo' or 'qp').
    """

    def __init__(
        self,
        window_size: int = 200,
        C: float = 1.0,
        kernel: Union[str, Kernel, Callable] = "rbf",
        solver: str = "smo",
        **kwargs,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.C = C
        self.kernel = kernel
        self.solver = solver
        self.kwargs = kwargs

        self._buffer_X = deque(maxlen=self.window_size)
        self._buffer_y = deque(maxlen=self.window_size)
        self._model: Optional[SVC] = None
        self._last_known_class = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SlidingWindowSVM":
        """Fit model with an initial batch of data."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X_arr.ndim == 2:
            self.n_features_in_ = X_arr.shape[1]

        self._buffer_X.clear()
        self._buffer_y.clear()

        # Load at most window_size latest samples
        start_idx = max(0, len(X_arr) - self.window_size)
        for i in range(start_idx, len(X_arr)):
            self._buffer_X.append(X_arr[i])
            self._buffer_y.append(y_arr[i])

        self._refit()
        return self

    def learn_one(self, x: np.ndarray, y: Union[int, float, str]) -> "SlidingWindowSVM":
        """
        Incrementally add a new sample to the window, evicting the oldest if full.
        """
        x_vec = np.asarray(x, dtype=float).ravel()
        if getattr(self, "n_features_in_", None) is None:
            self.n_features_in_ = len(x_vec)
        self._buffer_X.append(x_vec)
        self._buffer_y.append(y)
        self._last_known_class = y

        # Refit with updated window buffer
        self._refit()
        return self

    def forget_one(self, index: int = 0) -> "SlidingWindowSVM":
        """
        Explicitly remove an instance from the window buffer at the specified index.
        """
        if len(self._buffer_X) > 0 and 0 <= index < len(self._buffer_X):
            del self._buffer_X[index]
            del self._buffer_y[index]
            self._refit()
        return self

    def _refit(self) -> None:
        """Internal helper to re-train the underlying SVC on current window buffer."""
        if len(self._buffer_X) < 2:
            return

        unique_labels = np.unique(list(self._buffer_y))
        if len(unique_labels) < 2:
            return

        X_train = np.array(self._buffer_X)
        y_train = np.array(self._buffer_y)

        self._model = SVC(
            C=self.C,
            kernel=self.kernel,
            solver=self.solver,
            **self.kwargs,
        )
        self._model.fit(X_train, y_train)

        # Sync base properties
        self.support_vectors_ = self._model.support_vectors_
        self.dual_coef_ = self._model.dual_coef_
        self.support_ = self._model.support_
        self.intercept_ = self._model.intercept_
        self.classes_ = self._model.classes_
        self.n_support_ = self._model.n_support_

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        if self._model is not None and self._model.is_fitted:
            return self._model.decision_function(X_arr)
        return np.zeros(X_arr.shape[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = check_array(X, n_features=getattr(self, "n_features_in_", None))
        if self._model is not None and self._model.is_fitted:
            return self._model.predict(X_arr)
        
        # Fallback: predict majority label in buffer or last known class
        if len(self._buffer_y) > 0:
            labels, counts = np.unique(list(self._buffer_y), return_counts=True)
            majority_label = labels[np.argmax(counts)]
            return np.full(X_arr.shape[0], majority_label)

        return np.full(X_arr.shape[0], self._last_known_class)
