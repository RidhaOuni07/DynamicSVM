"""Dataset generators and evaluation tools for COSVM-family experiments."""

from dynamicsvm.benchmarks.datasets import (
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles,
)
from dynamicsvm.benchmarks.evaluator import AnomalyBenchmarkEvaluator

__all__ = [
    "make_anisotropic_anomaly_dataset",
    "make_benchmark_moons",
    "make_benchmark_circles",
    "AnomalyBenchmarkEvaluator",
]
