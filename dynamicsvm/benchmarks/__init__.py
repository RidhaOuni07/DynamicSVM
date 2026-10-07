"""
DynamicSVM Benchmarking Suite: datasets, prequential streaming evaluators, and comparative suites.
"""

from dynamicsvm.benchmarks.datasets import (
    make_sea_stream,
    make_rotating_hyperplane_stream,
    make_gaussian_drift_stream,
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles,
)
from dynamicsvm.benchmarks.evaluator import (
    PrequentialEvaluator,
    BatchBenchmarkEvaluator,
    AnomalyBenchmarkEvaluator,
)
from dynamicsvm.benchmarks.runner import (
    benchmark_streaming_suite,
    benchmark_anomaly_suite,
    benchmark_solvers_suite,
)

__all__ = [
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
