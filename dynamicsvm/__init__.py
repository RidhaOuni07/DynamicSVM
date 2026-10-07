"""
DynamicSVM: A Python library for Standard, Incremental, Decremental, and Drift-Aware SVMs.
"""

from dynamicsvm.kernels import (
    Kernel,
    LinearKernel,
    RBFKernel,
    PolynomialKernel,
    SigmoidKernel,
    get_kernel,
)
from dynamicsvm.base import BaseSVM, BaseClassifier
from dynamicsvm.models.svc import SVC
from dynamicsvm.models.multiclass import MultiClassSVC
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.encoders import (
    BaseEncoder,
    AutoEncoder,
    MLPAutoEncoder,
    PCAEncoder,
    IdentityEncoder,
)
from dynamicsvm.solvers import (
    SMOSolver,
    QPSolver,
    PegasosSolver,
    LASVM,
    LASVMClassifier,
)
from dynamicsvm.stream import (
    StreamingSVM,
    SlidingWindowSVM,
    DriftDetector,
    DDMDriftDetector,
    DriftAwareSVM,
    OnlineCovarianceTracker,
    LatentCovarianceDriftDetector,
    ObsolescenceScorer,
    DriftAwareCovarianceSVM,
    ICOSVM,
    IncrementalDecrementalCOSVM,
    IDCOSVM,
    AdiabaticSVM,
    IncrementalDecrementalSVC,
    ExactIncrementalDecrementalSVM,
)
from dynamicsvm.benchmarks import (
    make_sea_stream,
    make_rotating_hyperplane_stream,
    make_gaussian_drift_stream,
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles,
    PrequentialEvaluator,
    BatchBenchmarkEvaluator,
    AnomalyBenchmarkEvaluator,
    benchmark_streaming_suite,
    benchmark_anomaly_suite,
    benchmark_solvers_suite,
)

__version__ = "0.3.0"
__all__ = [
    "Kernel",
    "LinearKernel",
    "RBFKernel",
    "PolynomialKernel",
    "SigmoidKernel",
    "get_kernel",
    "BaseSVM",
    "BaseClassifier",
    "SVC",
    "MultiClassSVC",
    "CovarianceAwareSVC",
    "COSVM",
    "BaseEncoder",
    "AutoEncoder",
    "MLPAutoEncoder",
    "PCAEncoder",
    "IdentityEncoder",
    "SMOSolver",
    "QPSolver",
    "PegasosSolver",
    "LASVM",
    "LASVMClassifier",
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
    "make_sea_stream",
    "make_rotating_hyperplane_stream",
    "make_gaussian_drift_stream",
    "make_anisotropic_anomaly_dataset",
    "make_benchmark_moons",
    "make_benchmark_circles",
    "PrequentialEvaluator",
    "BatchBenchmarkEvaluator",
    "AnomalyBenchmarkEvaluator",
    "benchmark_streaming_suite",
    "benchmark_anomaly_suite",
    "benchmark_solvers_suite",
]
