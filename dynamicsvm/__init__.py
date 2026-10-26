"""COSVM, its streaming variants, and drift-aware one-class adaptation."""

from dynamicsvm.kernels import (
    Kernel,
    LinearKernel,
    RBFKernel,
    PolynomialKernel,
    SigmoidKernel,
    get_kernel,
)
from dynamicsvm.base import BaseSVM
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.encoders import (
    BaseEncoder,
    AutoEncoder,
    MLPAutoEncoder,
    PCAEncoder,
    IdentityEncoder,
)
from dynamicsvm.stream import (
    DriftDetector,
    DDMDriftDetector,
    DriftAwareOneClassSVM,
    ICOSVM,
    IncrementalDecrementalCOSVM,
    IDCOSVM,
)
from dynamicsvm.benchmarks import (
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles,
    AnomalyBenchmarkEvaluator,
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
    "COSVM",
    "BaseEncoder",
    "AutoEncoder",
    "MLPAutoEncoder",
    "PCAEncoder",
    "IdentityEncoder",
    "DriftDetector",
    "DDMDriftDetector",
    "DriftAwareOneClassSVM",
    "ICOSVM",
    "IncrementalDecrementalCOSVM",
    "IDCOSVM",
    "make_anisotropic_anomaly_dataset",
    "make_benchmark_moons",
    "make_benchmark_circles",
    "AnomalyBenchmarkEvaluator",
]
