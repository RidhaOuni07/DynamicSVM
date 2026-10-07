"""
Streaming, incremental, decremental, and concept drift modules for dynamic SVM research.
"""

from dynamicsvm.stream.base import StreamingSVM
from dynamicsvm.stream.windowed import SlidingWindowSVM
from dynamicsvm.stream.drift import DriftDetector, DDMDriftDetector, DriftAwareSVM
from dynamicsvm.stream.covariance import (
    OnlineCovarianceTracker,
    LatentCovarianceDriftDetector,
    ObsolescenceScorer,
)
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM, IDCOSVM
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM, IncrementalDecrementalSVC, ExactIncrementalDecrementalSVM
from dynamicsvm.solvers.lasvm import LASVM, LASVMClassifier

__all__ = [
    "StreamingSVM",
    "SlidingWindowSVM",
    "DriftDetector",
    "DDMDriftDetector",
    "DriftAwareSVM",
    "OnlineCovarianceTracker",
    "LatentCovarianceDriftDetector",
    "ObsolescenceScorer",
    "DriftAwareCovarianceSVM",
    "ICOSVM",
    "IncrementalDecrementalCOSVM",
    "IDCOSVM",
    "AdiabaticSVM",
    "IncrementalDecrementalSVC",
    "ExactIncrementalDecrementalSVM",
    "LASVM",
    "LASVMClassifier",
]
