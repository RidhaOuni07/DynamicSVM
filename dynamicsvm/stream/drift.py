"""Binary drift detectors used by one-class streaming models."""

from abc import ABC, abstractmethod
import numpy as np


class DriftDetector(ABC):
    """Abstract base class for streaming concept drift detectors."""

    @abstractmethod
    def update(self, error: bool) -> bool:
        """
        Update the detector with a binary error (True if prediction error, False otherwise).
        Returns True if a concept drift is detected.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the internal statistics of the detector."""
        pass


class DDMDriftDetector(DriftDetector):
    r"""
    DDM: Drift Detection Method (Gama et al., 2004).
    Monitors online error rate $p_i$ and $s_i = \sqrt{p_i (1 - p_i) / i}$.
    
    Warning level: $p_i + s_i > p_{min} + \alpha \cdot s_{min}$
    Drift level:   $p_i + s_i > p_{min} + \beta \cdot s_{min}$

    $(p_{min}, s_{min})$ is the pair at the minimum of $p_i + s_i$. With no
    errors yet, $p_i = s_i = 0$, so the minimum would be $p_{min} = s_{min} = 0$
    and the first error would satisfy $p_i + s_i > 0$, a spurious drift. Warm
    start: the minimum is only tracked, and the levels only tested, once
    $i \ge$ ``min_instances``, $s_i > 0$ and at least ``min_errors`` errors were
    seen since the last reset (the usual $n p \ge 5$ condition for the normal
    approximation behind the $\alpha, \beta$ levels). Minima recorded with
    fewer errors are dominated by the discreteness of the count and cause
    false alarms once the error rate settles.

    Parameters
    ----------
    min_instances : int, default=30
        Minimum number of observations before minima are tracked and tested.
    warning_level : float, default=2.0
        $\alpha$ in the warning rule.
    drift_level : float, default=3.0
        $\beta$ in the drift rule.
    min_errors : int, default=5
        Minimum number of errors since reset before warning/drift is tested.
    """

    def __init__(
        self,
        min_instances: int = 30,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_errors: int = 5,
    ):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_errors = min_errors
        self.reset()

    def reset(self) -> None:
        self.n_samples = 0
        self.n_errors = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.p_plus_s_min = float("inf")
        self.in_warning_zone = False
        self.in_drift_zone = False

    def update(self, error: bool) -> bool:
        self.n_samples += 1
        x = 1.0 if error else 0.0
        if error:
            self.n_errors += 1

        # Incremental mean and standard deviation of Bernoulli trial
        self.p += (x - self.p) / self.n_samples
        self.s = np.sqrt(self.p * (1.0 - self.p) / self.n_samples) if self.n_samples > 1 else 0.0

        if self.n_samples < self.min_instances:
            return False

        p_plus_s = self.p + self.s

        # Warm start: no minimum and no test until the normal approximation of
        # the error count is usable (n * p >= min_errors errors, s > 0).
        if self.n_errors < self.min_errors or self.s <= 0.0:
            self.in_warning_zone = False
            self.in_drift_zone = False
            return False

        if p_plus_s < self.p_plus_s_min:
            self.p_min = self.p
            self.s_min = self.s
            self.p_plus_s_min = p_plus_s

        if p_plus_s > self.p_min + self.drift_level * self.s_min:
            self.in_drift_zone = True
            self.in_warning_zone = False
            return True
        elif p_plus_s > self.p_min + self.warning_level * self.s_min:
            self.in_warning_zone = True
            self.in_drift_zone = False
            return False
        else:
            self.in_warning_zone = False
            self.in_drift_zone = False
            return False
