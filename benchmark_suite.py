"""
DynamicSVM Comprehensive Benchmark Suite.

Executes and displays comparative performance benchmarks across:
1. Optimization Solvers (SMO, QP, Pegasos, LASVM) on Non-linear Classification
2. Online & Streaming Classifiers (SlidingWindow, LASVM, AdiabaticSVM, DriftAwareCovarianceSVM) on SEA Drift Stream
3. Rotating Hyperplane Gradual Concept Drift Tracking
4. Anisotropic Low-Variance Direction Anomaly Detection (COSVM, ICOSVM, IDCOSVM)
"""

from typing import Dict
import numpy as np

from dynamicsvm.models.svc import SVC
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.solvers.lasvm import LASVM
from dynamicsvm.solvers.pegasos import PegasosSolver
from dynamicsvm.stream.windowed import SlidingWindowSVM
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM
from dynamicsvm.encoders.autoencoder import PCAEncoder

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


def run_all_benchmarks():
    print("=" * 78)
    print("         DynamicSVM Benchmarking Suite (Comprehensive Evaluation)")
    print("=" * 78)

    # -----------------------------------------------------------------
    # Benchmark 1: Solver Comparison on Non-linear Two Moons
    # -----------------------------------------------------------------
    print("\n[Benchmark 1] Optimization Solvers on Non-linear Two Moons (N=200)")
    solvers = {
        "SMO (Platt / Keerthi)": SVC(C=2.0, kernel="rbf", gamma=1.0, solver="smo", random_state=42),
        "QP (Exact Dual Solver)": SVC(C=2.0, kernel="rbf", gamma=1.0, solver="qp"),
        "LASVM (Active Online SMO)": LASVM(C=2.0, kernel="rbf", gamma=1.0, epochs=2, selection_strategy="active", random_state=42),
        "Covariance-Aware SVC": CovarianceAwareSVC(C=2.0, eta=0.7, kernel="rbf", gamma=1.0),
    }

    solvers_res = benchmark_solvers_suite(solvers, dataset_name="moons", n_samples=200, random_state=42)
    print(f"{'Solver':<28} | {'Train Acc':<10} | {'Test Acc':<10} | {'Fit (ms)':<10} | {'SVs':<6} | {'SV Ratio':<8}")
    print("-" * 82)
    for row in solvers_res["summary"]:
        print(f"{row['solver']:<28} | {row['train_accuracy']*100:>8.2f}% | {row['test_accuracy']*100:>8.2f}% | {row['fit_time_ms']:>8.2f}ms | {row['n_support_vectors']:>4} | {row['sv_ratio']*100:>7.1f}%")

    # -----------------------------------------------------------------
    # Benchmark 2: Prequential Streaming Benchmark on SEA Drift Stream
    # -----------------------------------------------------------------
    print("\n[Benchmark 2] Prequential Streaming Benchmark on SEA Concept Drift Stream (N=600)")
    streaming_models = {
        "SlidingWindow (W=40)": SlidingWindowSVM(window_size=40, kernel="linear", C=1.0),
        "AdiabaticSVM (CP 2001)": AdiabaticSVM(C=1.0, kernel="linear", max_samples=40),
        "LASVM (Online Stream)": LASVM(C=1.0, kernel="linear", epochs=1, selection_strategy="random", random_state=42),
        "DriftAwareCovarianceSVM": DriftAwareCovarianceSVM(
            encoder=PCAEncoder(n_components=2),
            C=1.0,
            eta=0.7,
            max_buffer_size=50,
            min_buffer_size=15,
            drift_threshold=0.35,
            random_state=42,
        ),
    }

    sea_res = benchmark_streaming_suite(
        models=streaming_models,
        dataset_name="sea",
        n_samples=600,
        window_size=50,
        initial_fit_samples=30,
        random_state=42,
    )
    print(f"Abrupt drift points located at sample indices: {sea_res['drift_points']}")
    print(f"{'Model':<26} | {'Cum. Acc':<10} | {'Final Acc':<10} | {'Avg Latency':<12} | {'Throughput':<15} | {'Final SVs':<8}")
    print("-" * 92)
    for row in sea_res["summary"]:
        print(f"{row['model']:<26} | {row['cumulative_accuracy']*100:>8.2f}% | {row['final_rolling_accuracy']*100:>8.2f}% | {row['mean_step_latency_ms']:>9.3f} ms | {row['throughput_samples_per_sec']:>11.1f} s/s | {row['final_sv_count']:>8}")

    # -----------------------------------------------------------------
    # Benchmark 3: Gradual Rotating Hyperplane Streaming Benchmark
    # -----------------------------------------------------------------
    print("\n[Benchmark 3] Rotating Hyperplane Gradual Drift Stream (N=500)")
    rot_models = {
        "SlidingWindow (W=50)": SlidingWindowSVM(window_size=50, kernel="linear", C=1.0),
        "AdiabaticSVM": AdiabaticSVM(C=1.0, kernel="linear", max_samples=50),
        "LASVM": LASVM(C=1.0, kernel="linear", epochs=1, random_state=42),
    }

    rot_res = benchmark_streaming_suite(
        models=rot_models,
        dataset_name="hyperplane",
        n_samples=500,
        window_size=50,
        initial_fit_samples=30,
        random_state=42,
    )
    print(f"{'Model':<24} | {'Cum. Acc':<10} | {'Final Acc':<10} | {'Avg Latency':<12} | {'Throughput':<15} | {'Final SVs':<8}")
    print("-" * 90)
    for row in rot_res["summary"]:
        print(f"{row['model']:<24} | {row['cumulative_accuracy']*100:>8.2f}% | {row['final_rolling_accuracy']*100:>8.2f}% | {row['mean_step_latency_ms']:>9.3f} ms | {row['throughput_samples_per_sec']:>11.1f} s/s | {row['final_sv_count']:>8}")

    # -----------------------------------------------------------------
    # Benchmark 4: Low-Variance Anomaly Detection (COSVM vs ICOSVM vs IDCOSVM)
    # -----------------------------------------------------------------
    print("\n[Benchmark 4] Low-Variance Direction Anomaly Detection Benchmark (Ratio=10.0)")
    anomaly_models = {
        "Standard Baseline OCSVM (eta=1.0)": COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma="scale"),
        "Covariance-Guided COSVM (eta=0.4)": COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
        "Incremental ICOSVM (KKT R matrix)": ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale", max_samples=150),
        "Incremental/Decremental IDCOSVM": IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
    }

    anom_res = benchmark_anomaly_suite(
        models=anomaly_models,
        n_normal=150,
        n_anomalies=30,
        ratio=10.0,
        dimension=2,
        random_state=42,
    )
    print(f"{'Detector':<35} | {'TPR (Normal)':<13} | {'TNR (Anomaly)':<14} | {'Balanced Acc':<13} | {'Fit (ms)':<9}")
    print("-" * 92)
    for row in anom_res["summary"]:
        print(f"{row['model']:<35} | {row['target_detection_rate']*100:>11.2f}% | {row['anomaly_detection_rate']*100:>12.2f}% | {row['balanced_accuracy']*100:>11.2f}% | {row['fit_time_ms']:>7.2f}ms")

    print("\n" + "=" * 78)
    print("                 All benchmark evaluations completed successfully!")
    print("=" * 78)


if __name__ == "__main__":
    run_all_benchmarks()
