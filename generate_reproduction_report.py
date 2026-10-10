"""
Script to run empirical replications of Kefi-Fatteh et al. (2019) and DynamicSVM benchmarks
and generate a comprehensive markdown report in reproduction_results.md.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any

from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM
from dynamicsvm.models.svc import SVC
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.solvers.lasvm import LASVM
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

# Randomized trials per configuration in Sections 1 and 2 (also quoted in the report text).
N_TRIALS = 3


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC) from continuous decision scores.
    y_true: +1 for normal/target, -1 for anomaly.
    scores: higher score => more likely target (+1).
    """
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == -1]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Mann-Whitney U test formula for AUC
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    # count pairs where pos_score > neg_score
    # vectorized
    diffs = pos_scores[:, None] - neg_scores[None, :]
    auc = (np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / (n_pos * n_neg)
    return float(auc)


def compute_fpr_at_fixed_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    Compute False Positive Rate (FPR) when True Positive Rate (TPR) is at least target_tpr.
    y_true: +1 for normal/target, -1 for anomaly.
    Normal points are considered Positive class in anomaly detection.
    """
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == -1]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    # Determine threshold so that fraction of pos_scores >= threshold is target_tpr
    sorted_pos = np.sort(pos_scores)
    threshold_idx = int(np.floor((1.0 - target_tpr) * len(sorted_pos)))
    threshold_idx = max(0, min(len(sorted_pos) - 1, threshold_idx))
    threshold = sorted_pos[threshold_idx]

    # FPR = False Positives (anomalies predicted as normal, score >= threshold) / Total Negatives
    fp = np.sum(neg_scores >= threshold)
    fpr = fp / len(neg_scores)
    return float(fpr)


def generate_synthetic_2d_datasets(dataset_type: str, overlap: str = "low", n_samples: int = 500, random_state: int = 42):
    """
    Generate synthetic 2D datasets mirroring the geometries in Kefi-Fatteh et al. (2019):
    - Gaussian
    - Banana-shaped
    - Spherical
    """
    rng = np.random.RandomState(random_state)
    n_target = n_samples
    n_anomaly = int(n_samples * 0.4)

    if dataset_type == "gaussian":
        # Anisotropic Gaussian
        sep = 1.8 if overlap == "high" else 3.5
        cov_target = np.array([[3.0, 1.8], [1.8, 1.5]])
        cov_anom = np.array([[1.5, -1.0], [-1.0, 2.5]])
        X_target = rng.multivariate_normal([0, 0], cov_target, size=n_target)
        X_anomaly = rng.multivariate_normal([sep, sep], cov_anom, size=n_anomaly)

    elif dataset_type == "banana":
        # Banana-shaped distribution
        noise_level = 0.35 if overlap == "high" else 0.15
        t_target = rng.uniform(-np.pi * 0.8, np.pi * 0.8, size=n_target)
        r_target = 3.0 + rng.randn(n_target) * noise_level
        x1_t = r_target * np.cos(t_target)
        x2_t = r_target * np.sin(t_target) + 0.3 * (x1_t ** 2) / 3.0
        X_target = np.column_stack([x1_t, x2_t])

        sep = 0.5 if overlap == "high" else 2.5
        t_anom = rng.uniform(-np.pi * 0.8, np.pi * 0.8, size=n_anomaly)
        r_anom = 3.0 + rng.randn(n_anomaly) * noise_level
        x1_a = r_anom * np.cos(t_anom)
        x2_a = - (r_anom * np.sin(t_anom) + 0.3 * (x1_a ** 2) / 3.0) + sep
        X_anomaly = np.column_stack([x1_a, x2_a])

    elif dataset_type == "spherical":
        # Isotropic spherical distribution
        sep = 2.0 if overlap == "high" else 4.5
        cov_target = np.eye(2) * 1.5
        cov_anom = np.eye(2) * 1.5
        X_target = rng.multivariate_normal([0, 0], cov_target, size=n_target)
        X_anomaly = rng.multivariate_normal([sep, 0], cov_anom, size=n_anomaly)

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Train / Test split
    n_train = int(n_target * 0.7)
    X_train_target = X_target[:n_train]
    X_test_target = X_target[n_train:]
    X_test_anomaly = X_anomaly

    X_test = np.vstack([X_test_target, X_test_anomaly])
    y_test = np.array([1] * len(X_test_target) + [-1] * len(X_test_anomaly))

    return X_train_target, X_test, y_test


def _fmt_range(values: List[float], fmt: str = "{:+.2f}") -> str:
    """Format the [min, max] range of the finite entries of ``values``."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return "n/a"
    return f"{fmt.format(min(finite))} to {fmt.format(max(finite))}"


def _stream_icosvm(icosvm: ICOSVM, X_stream: np.ndarray, counters: Dict[str, int]) -> None:
    """Stream ``X_stream`` into ``icosvm`` and accumulate its update-path counters."""
    for pt in X_stream:
        icosvm.learn_one(pt)
    counters["incremental"] += int(icosvm.n_incremental_updates_)
    counters["qp"] += int(icosvm.n_qp_solves_)


def run_experiment_tables():
    print("Executing Empirical Benchmarks for Reproduction Report...")
    t_start = time.perf_counter()

    # Update-path counters of every streamed ICOSVM in sections 1-3
    icosvm_counters = {"incremental": 0, "qp": 0}

    header_lines = []
    body_lines = []

    # Title & Metadata
    header_lines.append("# Empirical Reproduction Results: Kefi-Fatteh et al. (2019) & DynamicSVM Suite")
    header_lines.append("")
    header_lines.append("> **Reference Paper**: Takoua Kefi-Fatteh, Riadh Ksantini, Mohamed-Becha Kaaniche, Adel Bouhoula.")
    header_lines.append("> *A novel incremental one-class support vector machine based on low variance direction.*")
    header_lines.append("> **Pattern Recognition 91 (2019): 308–321**.")
    header_lines.append("")

    # Section 1: Synthetic 2D Benchmark
    body_lines.append("---")
    body_lines.append("## 1. Synthetic 2D Manifold Benchmark (Replication of Table 3 & 5)")
    body_lines.append("")
    body_lines.append(
        f"We evaluated target vs. anomaly separation (test AUC) across anisotropic Gaussian, non-linear Banana, "
        f"and isotropic Spherical distributions under both low-overlap and high-overlap regimes over "
        f"{N_TRIALS} randomized trials (mean ± std over trials)."
    )
    body_lines.append("")

    dataset_configs = [
        ("Gaussian", "low"),
        ("Gaussian", "high"),
        ("Banana", "low"),
        ("Banana", "high"),
        ("Spherical", "low"),
        ("Spherical", "high"),
    ]

    table_rows = []
    for d_type, overlap in dataset_configs:
        osvm_aucs = []
        cosvm_aucs = []
        icosvm_aucs = []
        idcosvm_aucs = []

        for seed in range(N_TRIALS):
            X_train, X_test, y_test = generate_synthetic_2d_datasets(d_type.lower(), overlap=overlap, n_samples=160, random_state=100 + seed)

            # Baseline Standard OSVM (eta=1.0)
            osvm = COSVM(nu=0.15, eta=1.0, kernel="rbf", gamma="scale")
            osvm.fit(X_train)
            scores_osvm = osvm.score_samples(X_test)
            osvm_aucs.append(compute_auc(y_test, scores_osvm))

            # Batch COSVM (eta=0.35)
            cosvm = COSVM(nu=0.15, eta=0.35, kernel="rbf", gamma="scale")
            cosvm.fit(X_train)
            scores_cosvm = cosvm.score_samples(X_test)
            cosvm_aucs.append(compute_auc(y_test, scores_cosvm))

            # Streaming ICOSVM
            icosvm = ICOSVM(nu=0.15, eta=0.35, kernel="rbf", gamma="scale", max_samples=160)
            icosvm.fit(X_train[:25])
            _stream_icosvm(icosvm, X_train[25:], icosvm_counters)
            scores_icosvm = icosvm.score_samples(X_test)
            icosvm_aucs.append(compute_auc(y_test, scores_icosvm))

            # IDCOSVM
            idcosvm = IncrementalDecrementalCOSVM(nu=0.15, eta=0.35, kernel="rbf", gamma="scale", max_samples=160)
            idcosvm.fit(X_train[:25])
            for pt in X_train[25:]:
                idcosvm.learn_one(pt)
            scores_idcosvm = idcosvm.score_samples(X_test)
            idcosvm_aucs.append(compute_auc(y_test, scores_idcosvm))

        table_rows.append({
            "dataset": f"{d_type} ({overlap.capitalize()} Overlap)",
            "osvm": (np.mean(osvm_aucs) * 100, np.std(osvm_aucs) * 100),
            "cosvm": (np.mean(cosvm_aucs) * 100, np.std(cosvm_aucs) * 100),
            "icosvm": (np.mean(icosvm_aucs) * 100, np.std(icosvm_aucs) * 100),
            "idcosvm": (np.mean(idcosvm_aucs) * 100, np.std(idcosvm_aucs) * 100),
        })

    body_lines.append("| Dataset & Overlap Regime | Baseline OSVM ($\\eta=1.0$) | Batch COSVM ($\\eta=0.35$) | Streaming iCOSVM | IDCOSVM | iCOSVM AUC Gain over Baseline |")
    body_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    auc_gains = []
    for r in table_rows:
        gain = r["icosvm"][0] - r["osvm"][0]
        auc_gains.append(gain)
        body_lines.append(
            f"| **{r['dataset']}** | {r['osvm'][0]:.2f}% ± {r['osvm'][1]:.2f} | "
            f"{r['cosvm'][0]:.2f}% ± {r['cosvm'][1]:.2f} | "
            f"**{r['icosvm'][0]:.2f}% ± {r['icosvm'][1]:.2f}** | "
            f"{r['idcosvm'][0]:.2f}% ± {r['idcosvm'][1]:.2f} | "
            f"`{gain:+.2f}%` |"
        )
    body_lines.append("")
    n_pos_gain = sum(1 for g in auc_gains if g > 0)
    max_stream_vs_batch = max(abs(r["icosvm"][0] - r["cosvm"][0]) for r in table_rows)
    body_lines.append(
        f"*Observation (computed in this run)*: the mean iCOSVM AUC minus the baseline OSVM AUC ranges from "
        f"{_fmt_range(auc_gains)} percentage points across the {len(auc_gains)} configurations "
        f"(positive in {n_pos_gain} of {len(auc_gains)}). The largest |streaming iCOSVM - batch COSVM| mean AUC "
        f"difference is {max_stream_vs_batch:.2f} percentage points; it comes from `gamma='scale'`, which the "
        f"streaming models resolve once from their initial 25-sample fit while batch COSVM uses all training samples."
    )
    body_lines.append("")

    # Section 2: False Positive Rate at High TPR
    body_lines.append("---")
    body_lines.append("## 2. False Positive Rate (FPR) at Fixed High Detection Rates (Replication of Tables 8 & 9)")
    body_lines.append("")
    body_lines.append(
        f"FPR of anomalies accepted as targets when the score threshold keeps the target TPR at 95% and 99% "
        f"(mean over {N_TRIALS} randomized trials). The last column is the relative change of the FPR at TPR = 95% "
        f"(negative = fewer false alarms than the baseline)."
    )
    body_lines.append("")

    fpr_rows = []
    for d_type, overlap in [("Gaussian", "high"), ("Banana", "high"), ("Spherical", "high")]:
        osvm_fpr95, icosvm_fpr95 = [], []
        osvm_fpr99, icosvm_fpr99 = [], []

        for seed in range(N_TRIALS):
            X_train, X_test, y_test = generate_synthetic_2d_datasets(d_type.lower(), overlap=overlap, n_samples=150, random_state=200 + seed)

            osvm = COSVM(nu=0.15, eta=1.0, kernel="rbf", gamma="scale").fit(X_train)
            sc_osvm = osvm.score_samples(X_test)
            osvm_fpr95.append(compute_fpr_at_fixed_tpr(y_test, sc_osvm, 0.95))
            osvm_fpr99.append(compute_fpr_at_fixed_tpr(y_test, sc_osvm, 0.99))

            icosvm = ICOSVM(nu=0.15, eta=0.35, kernel="rbf", gamma="scale", max_samples=150)
            icosvm.fit(X_train[:25])
            _stream_icosvm(icosvm, X_train[25:], icosvm_counters)
            sc_icosvm = icosvm.score_samples(X_test)
            icosvm_fpr95.append(compute_fpr_at_fixed_tpr(y_test, sc_icosvm, 0.95))
            icosvm_fpr99.append(compute_fpr_at_fixed_tpr(y_test, sc_icosvm, 0.99))

        fpr_rows.append({
            "dataset": f"{d_type} ({overlap.capitalize()} Overlap)",
            "osvm_95": np.mean(osvm_fpr95) * 100,
            "icosvm_95": np.mean(icosvm_fpr95) * 100,
            "osvm_99": np.mean(osvm_fpr99) * 100,
            "icosvm_99": np.mean(icosvm_fpr99) * 100,
        })

    body_lines.append("| Dataset | Baseline OSVM FPR @ TPR=95% | iCOSVM FPR @ TPR=95% | Baseline OSVM FPR @ TPR=99% | iCOSVM FPR @ TPR=99% | Relative FPR Change @ 95% |")
    body_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    fpr_changes = []
    for r in fpr_rows:
        if r["osvm_95"] > 0:
            change = (r["icosvm_95"] / r["osvm_95"] - 1.0) * 100.0
            change_txt = f"`{change:+.1f}%`"
        else:
            change = float("nan")
            change_txt = "n/a (baseline FPR = 0)"
        fpr_changes.append(change)
        body_lines.append(
            f"| **{r['dataset']}** | {r['osvm_95']:.2f}% | **{r['icosvm_95']:.2f}%** | "
            f"{r['osvm_99']:.2f}% | **{r['icosvm_99']:.2f}%** | {change_txt} |"
        )
    body_lines.append("")
    body_lines.append(
        f"*Observation (computed in this run)*: the relative change of the FPR at TPR = 95% ranges from "
        f"{_fmt_range(fpr_changes, '{:+.1f}')}% across the {len(fpr_rows)} high-overlap datasets."
    )
    body_lines.append("")

    # Section 3: Computational Scaling & Latency
    body_lines.append("---")
    body_lines.append("## 3. Computational Scaling & Latency Benchmark (Replication of Table 10)")
    body_lines.append("")
    body_lines.append(
        "Wall-clock time of one batch COSVM fit on $N$ samples, compared with the mean time of one ICOSVM "
        "`learn_one` (30 insertions starting from 20 samples) and one ICOSVM `unlearn_one` (5 removals). "
        "Every ICOSVM step that is not resolved by the KKT predictor re-solves the current COSVM dual "
        "with warm-started SLSQP, so a step costs $\\mathcal{O}(N^3)$ in the worst case (Cholesky of "
        "$A = \\eta K + \\frac{1-\\eta}{N} K H K$ and the QP), on the buffer size at that moment."
    )
    body_lines.append("")

    sample_sizes = [80, 160, 240, 320]
    time_rows = []
    unlearn_rel_diffs = []
    for N in sample_sizes:
        X_tr, _, _ = generate_synthetic_2d_datasets("gaussian", "low", n_samples=N, random_state=42)

        # Batch COSVM Fit
        t0 = time.perf_counter()
        cosvm = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
        cosvm.fit(X_tr)
        t_batch_ms = (time.perf_counter() - t0) * 1000.0

        # Incremental step latency (buffer grows from 20 to 50 samples)
        icosvm = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale", max_samples=N)
        icosvm.fit(X_tr[:20])
        step_times = []
        for sample in X_tr[20:50]:
            t_s0 = time.perf_counter()
            icosvm.learn_one(sample)
            step_times.append((time.perf_counter() - t_s0) * 1000.0)
        t_inc_step_ms = np.mean(step_times)

        # Decremental step latency (ICOSVM.unlearn_one re-solves the reduced problem)
        unlearn_times = []
        for _ in range(5):
            t_u0 = time.perf_counter()
            icosvm.unlearn_one(0)
            unlearn_times.append((time.perf_counter() - t_u0) * 1000.0)
        t_dec_step_ms = np.mean(unlearn_times)
        icosvm_counters["incremental"] += int(icosvm.n_incremental_updates_)
        icosvm_counters["qp"] += int(icosvm.n_qp_solves_)

        # Exactness of unlearning: batch COSVM refit on the remaining buffer (same frozen gamma)
        X_remaining = np.asarray(icosvm.X_buffer_)
        refit = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=icosvm.kernel_.gamma).fit(X_remaining)
        f_stream = icosvm.score_samples(X_tr)
        f_refit = refit.score_samples(X_tr)
        remaining_size = len(X_remaining)
        unlearn_rel_diffs.append(float(np.max(np.abs(f_stream - f_refit)) / (np.max(np.abs(f_refit)) + 1e-300)))

        time_rows.append({
            "N": N,
            "batch_time_ms": t_batch_ms,
            "inc_step_ms": t_inc_step_ms,
            "dec_step_ms": t_dec_step_ms,
            "ratio": t_batch_ms / (t_inc_step_ms + 1e-8),
        })

    body_lines.append("| Batch Size ($N$) | Batch COSVM Fit on $N$ (ms) | ICOSVM learn_one, buffer 20→50 (ms) | ICOSVM decremental re-solve (ms) | Batch Fit / Step Time |")
    body_lines.append("| :---: | :---: | :---: | :---: | :---: |")
    for r in time_rows:
        body_lines.append(
            f"| **{r['N']}** | {r['batch_time_ms']:>8.2f} ms | **{r['inc_step_ms']:>6.3f} ms** | "
            f"**{r['dec_step_ms']:>6.3f} ms** | `{r['ratio']:>6.1f}x` |"
        )
    body_lines.append("")
    body_lines.append(
        "Note: the step timings are measured on a 20-50 sample buffer for every $N$, so the last column compares "
        "a batch fit on $N$ samples with one re-solve on a much smaller buffer; it is not an asymptotic speed-up."
    )
    body_lines.append("")
    body_lines.append(
        f"Exactness check (computed in this run): after the 5 removals, the ICOSVM scores on the $N$ samples "
        f"differ from a batch COSVM refit on the remaining {remaining_size}-sample buffer (same frozen $\\gamma$) by at most "
        f"{max(unlearn_rel_diffs):.1e} relative to max |score|."
    )
    body_lines.append("")
    body_lines.append("```")
    body_lines.append("Complexity per operation (N = current buffer size):")
    body_lines.append("  - Batch COSVM fit: O(N^3) (Cholesky of A, G = K A^-1 K, SLSQP dual)")
    body_lines.append("  - ICOSVM / IDCOSVM learn_one / unlearn_one: warm-started exact re-solve, O(N^3) worst case")
    body_lines.append("```")
    body_lines.append("")

    # Section 4: Automated Benchmarking Suite Live Results
    body_lines.append("---")
    body_lines.append("## 4. End-to-End Suite Live Benchmark Results")
    body_lines.append("")
    body_lines.append("### 4.1 Solver Comparison on Non-linear Two Moons ($N = 200$)")
    solvers = {
        "SMO (Platt / Keerthi)": SVC(C=2.0, kernel="rbf", gamma=1.0, solver="smo", random_state=42),
        "QP (SLSQP, KKT-checked)": SVC(C=2.0, kernel="rbf", gamma=1.0, solver="qp"),
        "LASVM (Active Online SMO)": LASVM(C=2.0, kernel="rbf", gamma=1.0, epochs=2, selection_strategy="active", random_state=42),
        "Covariance-Aware SVC": CovarianceAwareSVC(C=2.0, eta=0.7, kernel="rbf", gamma=1.0),
    }
    solvers_res = benchmark_solvers_suite(solvers, dataset_name="moons", n_samples=200, random_state=42)

    body_lines.append("| Solver Name | Train Accuracy | Test Accuracy | Fit Time (ms) | SV Count | SV Ratio |")
    body_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for row in solvers_res["summary"]:
        body_lines.append(f"| **{row['solver']}** | {row['train_accuracy']*100:.2f}% | {row['test_accuracy']*100:.2f}% | {row['fit_time_ms']:.2f} ms | {row['n_support_vectors']} | {row['sv_ratio']*100:.1f}% |")
    body_lines.append("")

    body_lines.append("### 4.2 Prequential Streaming Benchmark on SEA Concept Drift Stream ($N = 600$)")
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
            drift_threshold=3.0,  # D_t = -log10 p: drift when p < 0.001
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
    body_lines.append(f"> **True Abrupt Drift Points of the Stream**: Indices {sea_res['drift_points']}")
    body_lines.append("")
    body_lines.append("| Streaming Model | Cumulative Accuracy | Final Rolling Accuracy | Step Latency (ms) | Throughput (samples/s) | Final SVs |")
    body_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    stream_latencies = {}
    for row in sea_res["summary"]:
        stream_latencies[row["model"]] = row["mean_step_latency_ms"]
        body_lines.append(f"| **{row['model']}** | {row['cumulative_accuracy']*100:.2f}% | {row['final_rolling_accuracy']*100:.2f}% | {row['mean_step_latency_ms']:.3f} ms | {row['throughput_samples_per_sec']:.1f} s/s | {row['final_sv_count']} |")
    body_lines.append("")

    body_lines.append("### 4.3 Anisotropic Low-Variance Anomaly Detection ($N = 150, \\text{Ratio} = 10.0$)")
    body_lines.append("")
    body_lines.append("All four detectors are fitted in batch on the same 150 normal samples by the suite runner.")
    body_lines.append("")
    anomaly_models = {
        "Baseline OSVM (eta=1.0)": COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma="scale"),
        "Covariance COSVM (eta=0.4)": COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
        "ICOSVM (eta=0.4, batch fit)": ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale", max_samples=150),
        "IDCOSVM (eta=0.4, batch fit)": IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale"),
    }
    anom_res = benchmark_anomaly_suite(
        models=anomaly_models,
        n_normal=150,
        n_anomalies=30,
        ratio=10.0,
        dimension=2,
        random_state=42,
    )
    body_lines.append("| Detector Model | Target Detection Rate (TPR) | Anomaly Detection Rate (TNR) | Balanced Accuracy | Fit / Init Latency |")
    body_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for row in anom_res["summary"]:
        body_lines.append(f"| **{row['model']}** | {row['target_detection_rate']*100:.2f}% | {row['anomaly_detection_rate']*100:.2f}% | **{row['balanced_accuracy']*100:.2f}%** | {row['fit_time_ms']:.2f} ms |")
    body_lines.append("")

    # Section 5: Reproduction Instructions
    body_lines.append("---")
    body_lines.append("## 5. How to Reproduce Directly via DynamicSVM")
    body_lines.append("")
    body_lines.append("All results and benchmarks can be executed via the command line or Python API:")
    body_lines.append("")
    body_lines.append("```bash")
    body_lines.append("# Regenerate this report")
    body_lines.append("python generate_reproduction_report.py")
    body_lines.append("")
    body_lines.append("# Run comprehensive benchmark suite")
    body_lines.append("python benchmark_suite.py")
    body_lines.append("")
    body_lines.append("# Run unit tests across all SVM algorithms (scikit-learn is needed for the reference tests)")
    body_lines.append("pytest")
    body_lines.append("")
    body_lines.append("# Run interactive multi-model demo")
    body_lines.append("python main.py")
    body_lines.append("```")
    body_lines.append("")
    body_lines.append("```python")
    body_lines.append("from dynamicsvm.models.cosvm import COSVM")
    body_lines.append("from dynamicsvm.stream.icosvm import ICOSVM")
    body_lines.append("from dynamicsvm.benchmarks import make_anisotropic_anomaly_dataset, AnomalyBenchmarkEvaluator")
    body_lines.append("")
    body_lines.append("# 1. Generate anisotropic benchmark dataset")
    body_lines.append("X_train, X_test_norm, X_test_anom = make_anisotropic_anomaly_dataset(")
    body_lines.append("    n_normal=150, n_anomalies=30, ratio=10.0, random_state=42")
    body_lines.append(")")
    body_lines.append("")
    body_lines.append("# 2. Fit and stream with iCOSVM (each learn_one re-solves the COSVM dual from a warm start)")
    body_lines.append("icosvm = ICOSVM(nu=0.1, eta=0.4, kernel='rbf', gamma='scale')")
    body_lines.append("icosvm.fit(X_train[:30])")
    body_lines.append("for sample in X_train[30:]:")
    body_lines.append("    icosvm.learn_one(sample)")
    body_lines.append("")
    body_lines.append("# 3. Evaluate detection metrics (note: evaluate() refits the model in batch)")
    body_lines.append("evaluator = AnomalyBenchmarkEvaluator()")
    body_lines.append("metrics = evaluator.evaluate(icosvm, X_train, X_test_norm, X_test_anom)")
    body_lines.append("print(f'Balanced Accuracy: {metrics[\"balanced_accuracy\"] * 100:.2f}%')")
    body_lines.append("```")
    body_lines.append("")

    # Executive summary: generated from the tables computed above
    run_time_s = time.perf_counter() - t_start
    n_steps = icosvm_counters["incremental"] + icosvm_counters["qp"]
    inc_steps = [r["inc_step_ms"] for r in time_rows]
    dec_steps = [r["dec_step_ms"] for r in time_rows]
    summary_lines = []
    summary_lines.append("## Executive Summary")
    summary_lines.append("")
    summary_lines.append(
        "This document reports the empirical evaluation of the **Covariance-guided One-Class SVM (COSVM)**, "
        "the **Incremental COSVM (iCOSVM / ICOSVM)** and the **Incremental / Decremental COSVM (IDCOSVM)** "
        "implemented in `DynamicSVM`. All three share one formulation: the one-class dual "
        "$\\min_\\alpha \\frac{1}{2}\\alpha^T G \\alpha$ s.t. $\\sum_i \\alpha_i = 1$, $0 \\le \\alpha_i \\le \\frac{1}{\\nu N}$, "
        "with $G = K A^{-1} K$, $A = \\eta K + \\frac{1-\\eta}{N} K H K$, $H = I - \\frac{1}{N}\\mathbf{1}\\mathbf{1}^T$ "
        "($\\eta = 1$ gives the standard OSVM)."
    )
    summary_lines.append("")
    summary_lines.append(f"Every number below is computed in this run ({N_TRIALS} randomized trials per configuration, total run time {run_time_s:.1f} s):")
    summary_lines.append(
        f"1. **AUC (Section 1)**: mean iCOSVM AUC minus baseline OSVM AUC ranges from {_fmt_range(auc_gains)} "
        f"percentage points over {len(auc_gains)} configurations (positive in {n_pos_gain})."
    )
    summary_lines.append(
        f"2. **False alarms (Section 2)**: the relative change of the FPR at TPR = 95% versus the baseline ranges "
        f"from {_fmt_range(fpr_changes, '{:+.1f}')}% over {len(fpr_rows)} high-overlap datasets."
    )
    summary_lines.append(
        f"3. **Streaming cost (Section 3)**: one ICOSVM `learn_one` on a 20-50 sample buffer took "
        f"{_fmt_range(inc_steps, '{:.2f}')} ms on average and one `unlearn_one` {_fmt_range(dec_steps, '{:.2f}')} ms "
        f"(per-$N$ means). Over all {n_steps} streamed ICOSVM updates in Sections 1-3, "
        f"{icosvm_counters['incremental']} were resolved by the KKT predictor (`n_incremental_updates_`) and "
        f"{icosvm_counters['qp']} by a warm-started exact re-solve (`n_qp_solves_`); a re-solve costs "
        f"$\\mathcal{{O}}(N^3)$ in the worst case."
    )
    summary_lines.append(
        f"4. **Unlearning (Section 3)**: `unlearn_one` removes the sample and re-solves the reduced dual from a "
        f"feasible warm start; the scores after 5 removals differ from a batch COSVM refit on the remaining buffer "
        f"by at most {max(unlearn_rel_diffs):.1e} (relative)."
    )
    if stream_latencies:
        lat_txt = ", ".join(f"{name} {lat:.3f} ms" for name, lat in stream_latencies.items())
        summary_lines.append(f"5. **Prequential SEA stream (Section 4.2)**: mean step latency {lat_txt}.")
    summary_lines.append("")

    full_md = "\n".join(header_lines + summary_lines + body_lines)
    with open("reproduction_results.md", "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"Reproduction report generated successfully at reproduction_results.md ({run_time_s:.1f} s)!")


if __name__ == "__main__":
    run_experiment_tables()
