"""
Demonstration script for DynamicSVM.
Shows:
1. Simple SVM with Linear Kernel and SMO optimization.
2. Non-linear SVM with RBF (Gaussian) Kernel on non-linear data.
3. Support vector identification, margin visualization, and evaluation metrics.
4. Preview of the dynamic stream learning architecture (Incremental, Decremental, Concept Drift).
"""

import numpy as np
import matplotlib.pyplot as plt

from dynamicsvm.models.svc import SVC
from dynamicsvm.models.multiclass import MultiClassSVC
from dynamicsvm.models.covariance_svc import CovarianceAwareSVC
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.encoders.autoencoder import AutoEncoder, PCAEncoder
from dynamicsvm.stream.windowed import SlidingWindowSVM
from dynamicsvm.stream.drift import DDMDriftDetector, DriftAwareSVM
from dynamicsvm.stream.drift_covariance_svm import DriftAwareCovarianceSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.solvers.lasvm import LASVM
from dynamicsvm.benchmarks.datasets import make_sea_stream
from dynamicsvm.benchmarks.evaluator import PrequentialEvaluator
from dynamicsvm.benchmarks.runner import benchmark_streaming_suite
from dynamicsvm.utils.metrics import accuracy_score, confusion_matrix
from dynamicsvm.utils.kkt import dual_objective, kkt_report
from dynamicsvm.visualization.plots import plot_decision_boundary


def generate_linear_data(n_samples: int = 60, random_state: int = 42):
    """Generate a clean linearly separable 2D dataset."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2
    
    # Class +1 (top-right) and Class -1 (bottom-left)
    X_pos = rng.randn(n_per_class, 2) * 0.7 + np.array([2.0, 2.0])
    X_neg = rng.randn(n_per_class, 2) * 0.7 + np.array([-1.5, -1.5])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_per_class + [-1] * n_per_class)
    return X, y


def generate_nonlinear_moons(n_samples: int = 100, random_state: int = 42):
    """Generate a 2D two-moons non-linear dataset."""
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2
    
    # Moon 1
    theta1 = np.linspace(0, np.pi, n_per_class)
    x1 = np.cos(theta1) + rng.randn(n_per_class) * 0.08
    y1 = np.sin(theta1) + rng.randn(n_per_class) * 0.08
    X1 = np.column_stack([x1, y1])
    
    # Moon 2 (shifted and inverted)
    theta2 = np.linspace(0, np.pi, n_per_class)
    x2 = 1.0 - np.cos(theta2) + rng.randn(n_per_class) * 0.08
    y2 = 1.0 - np.sin(theta2) - 0.5 + rng.randn(n_per_class) * 0.08
    X2 = np.column_stack([x2, y2])
    
    X = np.vstack([X1, X2])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    return X, y


def main():
    print("=" * 70)
    print("  DynamicSVM: Modular Support Vector Machine Library (Draft v0.1.0)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Simple Linear SVM (SMO Solver)
    # -------------------------------------------------------------
    print("\n--- [Experiment 1] Simple Linear SVM on Separable Data ---")
    X_lin, y_lin = generate_linear_data(n_samples=60, random_state=42)

    linear_svm = SVC(C=1.0, kernel="linear", solver="smo", random_state=42)
    linear_svm.fit(X_lin, y_lin)

    y_pred_lin = linear_svm.predict(X_lin)
    acc_lin = accuracy_score(y_lin, y_pred_lin)

    print(f"Model: {linear_svm}")
    print(f"Training Accuracy: {acc_lin * 100:.2f}%")
    print(f"Number of Support Vectors: {len(linear_svm.support_vectors_)} (Class -1: {linear_svm.n_support_[0]}, Class +1: {linear_svm.n_support_[1]})")
    print(f"Learned Intercept (b): {linear_svm.intercept_:.4f}")
    print(f"Dual Coefficients sum (|alpha * y|): {np.sum(np.abs(linear_svm.dual_coef_)):.4f}")

    # Plot and save
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_decision_boundary(linear_svm, X_lin, y_lin, title="Simple Linear SVM (SMO Solver)", ax=ax)
    plt.savefig("svm_linear_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot -> svm_linear_demo.png")

    # -------------------------------------------------------------
    # 2. Non-linear Kernel SVM (RBF Kernel)
    # -------------------------------------------------------------
    print("\n--- [Experiment 2] Non-Linear Kernel SVM (RBF Kernel) ---")
    X_moon, y_moon = generate_nonlinear_moons(n_samples=100, random_state=42)

    rbf_svm = SVC(C=5.0, kernel="rbf", gamma=1.5, solver="smo", random_state=42)
    rbf_svm.fit(X_moon, y_moon)

    y_pred_moon = rbf_svm.predict(X_moon)
    acc_moon = accuracy_score(y_moon, y_pred_moon)
    cm = confusion_matrix(y_moon, y_pred_moon)

    print(f"Kernel: RBF (gamma=1.5, C=5.0)")
    print(f"Accuracy: {acc_moon * 100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Total Support Vectors: {len(rbf_svm.support_vectors_)}")

    # Plot and save
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_decision_boundary(rbf_svm, X_moon, y_moon, title="Non-Linear RBF Kernel SVM", ax=ax)
    plt.savefig("svm_rbf_nonlinear_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot -> svm_rbf_nonlinear_demo.png")

    # -------------------------------------------------------------
    # 3. Solver Verification: SMO vs QP (Exact Dual)
    # -------------------------------------------------------------
    print("\n--- [Experiment 3] Solver Verification: SMO vs QP ---")
    qp_svm = SVC(C=5.0, kernel="rbf", gamma=1.5, solver="qp")
    qp_svm.fit(X_moon, y_moon)
    acc_qp = accuracy_score(y_moon, qp_svm.predict(X_moon))

    print(f"SMO Accuracy: {acc_moon * 100:.2f}% | Number of SVs: {len(rbf_svm.support_vectors_)} | Intercept b: {rbf_svm.intercept_:.4f}")
    print(f"QP  Accuracy: {acc_qp * 100:.2f}% | Number of SVs: {len(qp_svm.support_vectors_)} | Intercept b: {qp_svm.intercept_:.4f}")

    # -------------------------------------------------------------
    # 4. Research Preview: Sliding Window & Concept Drift Detection
    # -------------------------------------------------------------
    print("\n--- [Experiment 4] Stream / Dynamic SVM Preview (PhD Roadmap) ---")
    sw_svm = SlidingWindowSVM(window_size=30, kernel="rbf", gamma=1.0, C=1.0)
    
    # Initialize with 30 points
    sw_svm.fit(X_moon[:30], y_moon[:30])
    print(f"Sliding Window initialized with {len(sw_svm._buffer_X)} instances.")

    # Incremental update: simulate arrival of 10 new streaming points
    for i in range(30, 40):
        sw_svm.learn_one(X_moon[i], y_moon[i])
    print(f"After streaming 10 new samples, active window size: {len(sw_svm._buffer_X)}")

    # Concept Drift Detector test
    drift_detector = DDMDriftDetector(min_instances=15, warning_level=1.5, drift_level=2.5)
    drift_aware_svm = DriftAwareSVM(base_estimator=sw_svm, drift_detector=drift_detector)
    drift_aware_svm.fit(X_moon[:30], y_moon[:30])

    # Stream the remaining instances
    drifts_found = 0
    for i in range(30, len(X_moon)):
        if drift_aware_svm.learn_one(X_moon[i], y_moon[i]):
            drifts_found += 1
            print(f"  [Drift Event] Concept drift detected and adapted at stream sample index {i}!")

    print(f"Streaming execution completed. Total drifts detected: {drifts_found}")

    # -------------------------------------------------------------
    # 5. PhD Research: Drift-Aware Covariance Incremental SVM
    # -------------------------------------------------------------
    print("\n--- [Experiment 5] Drift-Aware Covariance-Aware Incremental SVM ---")
    print("Pipeline: x -> Frozen AutoEncoder -> Latent z -> Online Covariance (mu_t, Sigma_t)")
    print("          -> Drift Metric D_t -> Obsolescence O_i Pruning -> CovarianceAwareSVC")

    rng = np.random.RandomState(42)
    # 6D feature space: initial regime
    X_pre_pos = rng.randn(40, 6) * 0.8 + np.array([2.0, 2.0, 1.0, 0.5, -0.5, 0.0])
    X_pre_neg = rng.randn(40, 6) * 0.8 + np.array([-2.0, -2.0, -1.0, -0.5, 0.5, 0.0])
    X_initial = np.vstack([X_pre_pos, X_pre_neg])
    y_initial = np.array([1] * 40 + [0] * 40)

    # Train AutoEncoder on representation space
    ae = AutoEncoder(latent_dim=3, hidden_dims=(16,), max_iter=60, learning_rate=0.01, random_state=42)
    ae.fit(X_initial)
    ae.freeze()
    print(f"Trained & Frozen MLP AutoEncoder (6D -> 3D latent bottleneck).")

    # Initialize DriftAwareCovarianceSVM
    cov_svm = DriftAwareCovarianceSVM(
        encoder=ae,
        C=2.0,
        eta=0.7,
        max_buffer_size=100,
        min_buffer_size=15,
        drift_threshold=3.0,  # D_t = -log10 p: drift when p < 0.001
        warning_threshold=2.0,  # warning when p < 0.01
        lambda_base=0.92,
        lambda_min=0.45,
        retire_ratio_on_drift=0.35,
        random_state=42,
    )
    cov_svm.fit(X_initial, y_initial)
    init_acc = accuracy_score(y_initial, cov_svm.predict(X_initial))
    print(f"Initial Covariance SVM fit on {len(X_initial)} samples | Accuracy: {init_acc * 100:.2f}%")

    # Generate streaming drift regime (distribution shift in latent space)
    X_drift_pos = rng.randn(30, 6) * 0.8 + np.array([6.0, 4.0, -2.0, 2.0, 1.0, 1.0])
    X_drift_neg = rng.randn(30, 6) * 0.8 + np.array([1.0, -4.0, 2.0, -2.0, -1.0, -1.0])
    X_drift_stream = np.vstack([X_drift_pos, X_drift_neg])
    y_drift_stream = np.array([1] * 30 + [0] * 30)

    print("\nStreaming 60 drifted observations...")
    for idx, (x_t, y_t) in enumerate(zip(X_drift_stream, y_drift_stream)):
        cov_svm.learn_one(x_t, y_t)

    state = cov_svm.get_model_state()
    print(f"Drift Events Detected: {len(cov_svm.drift_events_)}")
    for ev in cov_svm.drift_events_:
        print(f"  -> Drift at step {ev['time']}: D_t = -log10 p = {ev['D_t']:.4f}, adaptive lambda = {ev['lambda_t']:.4f}, retired {ev['retired_samples']} samples")

    post_drift_acc = accuracy_score(y_drift_stream, cov_svm.predict(X_drift_stream))
    print(f"Adapted Model Accuracy on New Regime: {post_drift_acc * 100:.2f}%")
    print(f"Active Memory Buffer Size: {len(cov_svm._buffer_X)} / {cov_svm.max_buffer_size}")

    # -------------------------------------------------------------
    # 6. Low Variance Direction One-Class SVM (COSVM & ICOSVM)
    #    Based on Kefi-Fatteh et al. (Pattern Recognition 2019)
    # -------------------------------------------------------------
    print("\n--- [Experiment 6] Low-Variance Direction One-Class SVM (COSVM / ICOSVM) ---")
    print("Paper: Kefi-Fatteh et al. (Pattern Recognition 2019)")
    print("Optimization: min 0.5 * alpha^T G alpha, G = K A^-1 K, A = eta*K + ((1-eta)/N) * K H K")
    print("ICOSVM updates: warm-started exact re-solve of the current COSVM dual")

    rng = np.random.RandomState(42)
    # Generate anisotropic target distribution (high variance along x, low variance along y)
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X_target_base = rng.randn(60, 2) * np.array([[4.0, 0.4]])
    X_target = np.dot(X_target_base, rot.T)

    # 1. Batch COSVM with optimal eta
    cosvm = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    cosvm.fit(X_target)
    print(f"Batch COSVM fitted on {len(X_target)} target samples | Found {len(cosvm.support_vectors_)} Support Vectors.")

    # 2. Incremental ICOSVM streaming
    icosvm = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale", max_samples=75)
    icosvm.fit(X_target[:40])
    print(f"Incremental ICOSVM initialized with 40 samples.")

    # Stream in 20 additional normal points
    icosvm.learn_batch(X_target[40:60])
    print(f"Streamed 20 normal samples into ICOSVM. Active buffer size: {len(icosvm.X_buffer_)}.")

    # Test on test normal vs outlier instances
    X_test_normal = np.dot(rng.randn(20, 2) * np.array([[4.0, 0.4]]), rot.T)
    # Outliers aligned along the sensitive low-variance orthogonal axis
    X_test_outliers = np.dot(rng.randn(20, 2) * np.array([[0.5, 3.5]]), rot.T) + np.array([5.0, -5.0])

    preds_normal = icosvm.predict(X_test_normal)
    preds_outlier = icosvm.predict(X_test_outliers)

    normal_accuracy = np.mean(preds_normal == 1)
    outlier_accuracy = np.mean(preds_outlier == -1)

    print(f"ICOSVM Target Detection Rate (True Positive): {normal_accuracy * 100:.2f}%")
    print(f"ICOSVM Anomaly Detection Rate (True Negative): {outlier_accuracy * 100:.2f}%")

    # -------------------------------------------------------------
    # 7. Incremental / Decremental SVM (Adiabatic Path Following)
    #    Based on Cauwenberghs & Poggio (NeurIPS 2000 / 2001)
    # -------------------------------------------------------------
    print("\n--- [Experiment 7] Adiabatic Path Following Incremental / Decremental SVM ---")
    print("Paper: Cauwenberghs & Poggio (NeurIPS 2000 / 2001), Laskov et al. (JMLR 2006)")
    print("Mechanism: KKT path following over the sets (S, E, O); R recomputed at every event")

    X_init_ad = np.array([
        [2.0, 2.0], [2.5, 2.3], [3.0, 2.8],
        [-2.0, -2.0], [-2.3, -2.5], [-2.8, -3.0]
    ])
    y_init_ad = np.array([1, 1, 1, -1, -1, -1])

    adiabatic_svm = AdiabaticSVM(C=1.0, kernel="rbf", gamma=0.5, max_samples=10)
    adiabatic_svm.fit(X_init_ad, y_init_ad)
    print(f"Initialized AdiabaticSVM with 6 samples. Model state: {adiabatic_svm.get_model_state()}")

    # Streaming insertions
    stream_ad_X = np.array([[2.2, 2.2], [-2.1, -2.1], [3.5, 3.5], [-3.5, -3.5]])
    stream_ad_y = np.array([1, -1, 1, -1])
    for xi, yi in zip(stream_ad_X, stream_ad_y):
        adiabatic_svm.learn_one(xi, yi)

    print(f"After streaming 4 new observations: {adiabatic_svm.get_model_state()}")

    # Exact decremental unlearning
    adiabatic_svm.forget_one(0)
    print(f"After unlearning oldest observation (forget_one): {adiabatic_svm.get_model_state()}")
    ad_preds = adiabatic_svm.predict(stream_ad_X)
    print(f"Predictions on test stream: {ad_preds} (Ground Truth: {stream_ad_y})")

    # Verify the decremental step: equality constraint, KKT gap and agreement with a retrain
    y_buf_ad = np.asarray(adiabatic_svm.y_buffer_, dtype=float)
    X_buf_ad = np.asarray(adiabatic_svm.X_buffer_)
    ad_report = kkt_report(adiabatic_svm.alpha_, adiabatic_svm.K_full_, y_buf_ad, adiabatic_svm.C)
    print(f"|sum alpha_i y_i| after forget_one(0): {ad_report['sum_alpha_y']:.2e} | "
          f"KKT gap: {ad_report['kkt_gap']:.2e} | SMO fallbacks: {adiabatic_svm.n_smo_fallbacks_}")
    retrain_ad = SVC(C=1.0, kernel="rbf", gamma=0.5, solver="smo", tol=1e-6, max_iter=10000, random_state=42)
    retrain_ad.fit(X_buf_ad, y_buf_ad)
    K_ad = adiabatic_svm.K_full_
    alpha_rt = np.zeros(len(y_buf_ad))
    alpha_rt[retrain_ad.support_] = retrain_ad.alpha_
    X_eval_ad = np.vstack([X_buf_ad, stream_ad_X])
    f_diff_ad = np.max(np.abs(adiabatic_svm.decision_function(X_eval_ad) - retrain_ad.decision_function(X_eval_ad)))
    print(f"Dual objective: decremental {ad_report['dual_objective']:.6f} vs SVC retrain "
          f"{dual_objective(alpha_rt, K_ad, y_buf_ad):.6f} | max |f_decremental - f_retrain| = {f_diff_ad:.2e}")

    # -------------------------------------------------------------
    # 8. LASVM (Online & Active Learning with SV Filtering)
    #    Based on Bordes, Ertekin, Weston, and Bottou (JMLR 2005)
    # -------------------------------------------------------------
    print("\n--- [Experiment 8] LASVM: Online & Active Learning with SV Filtering ---")
    print("Paper: Bordes et al. (JMLR 2005) 'Fast Kernel Classifiers with Online and Active Learning'")
    print("Mechanism: PROCESS, REPROCESS, FINISH with active support vector filtering")

    X_lasvm, y_lasvm = generate_nonlinear_moons(n_samples=120, random_state=42)
    # Convert labels 0 to -1
    y_lasvm = np.where(y_lasvm == 1, 1, -1)
    # Shuffle for balanced train / test sets
    perm = rng.permutation(len(X_lasvm))
    X_lasvm = X_lasvm[perm]
    y_lasvm = y_lasvm[perm]

    lasvm = LASVM(C=2.0, kernel="rbf", gamma=1.0, epochs=2, selection_strategy="active", random_state=42)
    lasvm.fit(X_lasvm[:80], y_lasvm[:80])
    acc_train = accuracy_score(y_lasvm[:80], lasvm.predict(X_lasvm[:80]))
    acc_test = accuracy_score(y_lasvm[80:], lasvm.predict(X_lasvm[80:]))

    print(f"LASVM Active Learning fit on 80 samples | Train Acc: {acc_train * 100:.2f}%, Test Acc: {acc_test * 100:.2f}%")
    print(f"LASVM State: {lasvm.get_model_state()}")

    # Online streaming update
    for xi, yi in zip(X_lasvm[80:100], y_lasvm[80:100]):
        lasvm.learn_one(xi, yi)
    print(f"After streaming 20 test points: {lasvm.get_model_state()}")

    # -------------------------------------------------------------
    # 9. DynamicSVM Benchmarks & Prequential Evaluation Suite
    # -------------------------------------------------------------
    print("\n--- [Experiment 9] DynamicSVM Benchmarks & Prequential Evaluation ---")
    print("Dataset: SEA Concepts Stream (Abrupt Drift across 4 concepts)")
    benchmark_models = {
        "SlidingWindow (W=40)": SlidingWindowSVM(window_size=40, kernel="linear", C=1.0),
        "AdiabaticSVM (Exact KKT)": AdiabaticSVM(C=1.0, kernel="linear", max_samples=40),
        "LASVM (Stream)": LASVM(C=1.0, kernel="linear", epochs=1, random_state=42),
    }
    sea_benchmark = benchmark_streaming_suite(
        models=benchmark_models,
        dataset_name="sea",
        n_samples=400,
        window_size=40,
        initial_fit_samples=20,
        random_state=42,
    )
    print(f"Drift points at steps: {sea_benchmark['drift_points']}")
    print(f"{'Model':<26} | {'Cum. Acc':<10} | {'Throughput':<15} | {'Latency':<10}")
    print("-" * 68)
    for row in sea_benchmark["summary"]:
        print(f"{row['model']:<26} | {row['cumulative_accuracy']*100:>8.2f}% | {row['throughput_samples_per_sec']:>11.1f} s/s | {row['mean_step_latency_ms']:>7.3f}ms")

    print("\nAll demonstrations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
