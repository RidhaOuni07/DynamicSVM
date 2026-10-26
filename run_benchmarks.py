"""
Comprehensive Benchmarking Suite for standard OCSVM, COSVM, ICOSVM, and IDCOSVM across Multiple Distributions.
Evaluates:
1. Batch Computational Performance & Latency Scaling (N = 50 to 1000) across standard OCSVM, COSVM, ICOSVM, and IDCOSVM.
2. Streaming Latency (Incremental learn_one & Decremental forget_one) for ICOSVM and IDCOSVM vs batch refit.
3. Matrix Updates and Geometry Computations (K vs G, condition numbers, Frobenius norms).
4. Multi-Distribution Anomaly Detection & Decision Boundaries:
   - Moon distribution (non-linear half-moons with outliers)
   - Circle distribution (concentric circles with outliers)
   - Normal / Anisotropic Gaussian distribution (low-variance manifold shifts)
5. Pruning Policy and Active Buffer Dynamics under Streaming Scenarios.

Generates high-resolution visualization figures and JSON results for the Markdown report.
"""

import os
import time
import json
from typing import Dict, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IncrementalDecrementalCOSVM, IDCOSVM
from dynamicsvm.models._cosvm_core import COSVMGeometry
from dynamicsvm.benchmarks.datasets import (
    make_anisotropic_anomaly_dataset,
    make_benchmark_moons,
    make_benchmark_circles
)
from dynamicsvm.utils.kkt import kkt_report, oneclass_kkt_report
from dynamicsvm.utils.metrics import accuracy_score


def compute_roc_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC non-parametrically using Mann-Whitney U statistic / rank sum."""
    pos_scores = scores[y_true_binary == 1]
    neg_scores = scores[y_true_binary == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    diffs = pos_scores[:, None] - neg_scores[None, :]
    u = np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
    return float(u / (len(pos_scores) * len(neg_scores)))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy for binary labels (+1, -1)."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == -1)
    tpr = np.mean(y_pred[pos_mask] == 1) if np.sum(pos_mask) > 0 else 0.0
    tnr = np.mean(y_pred[neg_mask] == -1) if np.sum(neg_mask) > 0 else 0.0
    return float(0.5 * (tpr + tnr))


ETA_CANDIDATES = (1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01)


def split_validation_test(
    X_normal: np.ndarray,
    X_anomaly: np.ndarray,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split held-out normal and anomaly samples into independent validation and test sets."""
    rng = np.random.RandomState(random_state)
    normal_order = rng.permutation(len(X_normal))
    anomaly_order = rng.permutation(len(X_anomaly))
    normal_cut = len(X_normal) // 2
    anomaly_cut = len(X_anomaly) // 2

    X_validation = np.vstack([
        X_normal[normal_order[:normal_cut]],
        X_anomaly[anomaly_order[:anomaly_cut]],
    ])
    y_validation = np.hstack([
        np.ones(normal_cut),
        -np.ones(anomaly_cut),
    ])
    X_test = np.vstack([
        X_normal[normal_order[normal_cut:]],
        X_anomaly[anomaly_order[anomaly_cut:]],
    ])
    y_test = np.hstack([
        np.ones(len(normal_order) - normal_cut),
        -np.ones(len(anomaly_order) - anomaly_cut),
    ])
    return X_validation, y_validation, X_test, y_test


def select_eta(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    gamma: Union[str, float],
) -> Tuple[float, float, Dict[str, float]]:
    """Select eta by validation ROC-AUC, preferring larger eta on ties."""
    y_validation_binary = (y_validation == 1).astype(int)
    candidate_aucs: Dict[str, float] = {}
    best_eta = ETA_CANDIDATES[0]
    best_auc = -np.inf

    for eta in ETA_CANDIDATES:
        model = COSVM(nu=0.1, eta=eta, kernel="rbf", gamma=gamma).fit(X_train)
        auc = compute_roc_auc(
            y_validation_binary,
            model.decision_function(X_validation),
        )
        candidate_aucs[str(eta)] = float(auc)
        if auc > best_auc or (auc == best_auc and eta > best_eta):
            best_eta = eta
            best_auc = auc

    return float(best_eta), float(best_auc), candidate_aucs


# Ensure output directory exists
os.makedirs("benchmark_figures", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

results = {}

# ==============================================================================
# BENCHMARK 1: Batch Fit Time Scaling vs N (standard OCSVM, COSVM, ICOSVM, IDCOSVM)
# ==============================================================================
print("--- Running Benchmark 1: Scaling Analysis (Time & Matrices) vs N ---")
sample_sizes = [50, 100, 175, 250, 350, 500]
dim = 10
std_ocsvm_times = []
cosvm_times = []
icosvm_times = []
idcosvm_times = []

gram_times = []
geom_inv_times = []
g_cond_numbers = []
k_cond_numbers = []

for n in sample_sizes:
    X_train_norm, _, _ = make_anisotropic_anomaly_dataset(
        n_normal=n, n_anomalies=int(0.2 * n), dimension=dim, ratio=8.0, random_state=42
    )
    
    # 1. Standard one-class SVM (COSVM with eta=1)
    t0 = time.perf_counter()
    std_ocsvm = COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma="scale")
    std_ocsvm.fit(X_train_norm)
    t_std_ocsvm = (time.perf_counter() - t0) * 1000.0
    std_ocsvm_times.append(t_std_ocsvm)
    
    # 2. COSVM (Batch Covariance-guided One-Class)
    t0 = time.perf_counter()
    cosvm = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    cosvm.fit(X_train_norm)
    t_cosvm = (time.perf_counter() - t0) * 1000.0
    cosvm_times.append(t_cosvm)
    
    # 3. ICOSVM (Incremental Covariance-guided One-Class batch fit)
    t0 = time.perf_counter()
    icosvm = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    icosvm.fit(X_train_norm)
    t_icosvm = (time.perf_counter() - t0) * 1000.0
    icosvm_times.append(t_icosvm)
    
    # 4. IDCOSVM (Incremental / Decremental batch fit)
    t0 = time.perf_counter()
    idcosvm = IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
    idcosvm.fit(X_train_norm)
    t_idcosvm = (time.perf_counter() - t0) * 1000.0
    idcosvm_times.append(t_idcosvm)
    
    # Detailed matrix update breakdown for COSVMGeometry
    K = cosvm.K_train_
    t0 = time.perf_counter()
    _ = cosvm.kernel_(X_train_norm)
    t_gram = (time.perf_counter() - t0) * 1000.0
    gram_times.append(t_gram)
    
    t0 = time.perf_counter()
    geom = COSVMGeometry(K, eta=0.4)
    t_geom = (time.perf_counter() - t0) * 1000.0
    geom_inv_times.append(t_geom)
    
    # Condition numbers
    cond_K = np.linalg.cond(K)
    cond_G = np.linalg.cond(geom.G)
    k_cond_numbers.append(float(cond_K))
    g_cond_numbers.append(float(cond_G))
    
    print(f"  N={n:4d} | Std OCSVM: {t_std_ocsvm:7.2f}ms | COSVM: {t_cosvm:7.2f}ms | ICOSVM: {t_icosvm:7.2f}ms | IDCOSVM: {t_idcosvm:7.2f}ms | Gram: {t_gram:6.2f}ms | Geom: {t_geom:6.2f}ms")

results["benchmark1"] = {
    "sample_sizes": sample_sizes,
    "standard_ocsvm_times": std_ocsvm_times,
    "cosvm_times": cosvm_times,
    "icosvm_times": icosvm_times,
    "idcosvm_times": idcosvm_times,
    "gram_times": gram_times,
    "geom_inv_times": geom_inv_times,
    "k_cond_numbers": k_cond_numbers,
    "g_cond_numbers": g_cond_numbers
}

# Plot 1: Batch Fit Time Comparison
plt.figure(figsize=(10, 5))
plt.plot(sample_sizes, std_ocsvm_times, 'o-', label='Standard OCSVM (COSVM, eta=1.0)', linewidth=2, color='#1f77b4')
plt.plot(sample_sizes, cosvm_times, 's--', label='COSVM (Batch SLSQP)', linewidth=2, color='#ff7f0e')
plt.plot(sample_sizes, icosvm_times, 'D:', label='ICOSVM (Batch SLSQP)', linewidth=2, color='#9467bd')
plt.plot(sample_sizes, idcosvm_times, '^-.', label='IDCOSVM (Batch SLSQP)', linewidth=2, color='#2ca02c')
plt.xlabel('Training Samples (N)')
plt.ylabel('Fit Execution Time (ms)')
plt.title('Batch Fit Time Scaling: Standard OCSVM vs COSVM vs ICOSVM vs IDCOSVM')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig1_batch_scaling_time.png', dpi=200)
plt.close()

# Plot 2: Matrix Computation Decomposition
plt.figure(figsize=(10, 5))
plt.plot(sample_sizes, gram_times, 'o-', label='Kernel Gram Matrix K Calculation', color='#3b528b', linewidth=2)
plt.plot(sample_sizes, geom_inv_times, 's--', label='Covariance Geometry G = K B^-1 Calculation', color='#d95f02', linewidth=2)
plt.xlabel('Training Samples (N)')
plt.ylabel('Computation Time (ms)')
plt.title('Covariance Geometry Matrix Decomposition vs Sample Size (N)')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig2_matrix_decomposition.png', dpi=200)
plt.close()

# Plot 3: Matrix Condition Numbers Comparison
plt.figure(figsize=(10, 5))
plt.plot(sample_sizes, k_cond_numbers, 'o-', label='Standard Kernel Matrix K Condition Number', color='#7570b3', linewidth=2)
plt.plot(sample_sizes, g_cond_numbers, 's--', label='Covariance Dual Gram G Condition Number (eta=0.4)', color='#e7298a', linewidth=2)
plt.xlabel('Training Samples (N)')
plt.ylabel('Condition Number kappa(M)')
plt.title('Matrix Regularization & Condition Number: K vs Dual Gram G')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig3_matrix_condition_numbers.png', dpi=200)
plt.close()


# ==============================================================================
# BENCHMARK 2: Incremental & Decremental Streaming Latency (ICOSVM & IDCOSVM)
# ==============================================================================
print("\n--- Running Benchmark 2: Streaming Latency & Incremental/Decremental Dynamics ---")
X_train_stream, _, _ = make_anisotropic_anomaly_dataset(n_normal=160, n_anomalies=0, dimension=4, ratio=5.0, random_state=42)

init_n = 20
X_init = X_train_stream[:init_n]
X_inc = X_train_stream[init_n:init_n+80]

icosvm_stream = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
icosvm_stream.fit(X_init)

idcosvm_stream = IncrementalDecrementalCOSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
idcosvm_stream.fit(X_init)

inc_times_icosvm = []
inc_times_idcosvm = []
inc_times_batch_cosvm = []
inc_times_refit_ocsvm = []
buffer_sizes = []

current_buf = list(X_init)

for i, x in enumerate(X_inc):
    current_buf.append(x)
    buf_arr = np.array(current_buf)
    buffer_sizes.append(len(buf_arr))
    
    # ICOSVM incremental update
    t0 = time.perf_counter()
    icosvm_stream.learn_one(x)
    t_inc_i = (time.perf_counter() - t0) * 1000.0
    inc_times_icosvm.append(t_inc_i)
    
    # IDCOSVM incremental update
    t0 = time.perf_counter()
    idcosvm_stream.learn_one(x)
    t_inc_id = (time.perf_counter() - t0) * 1000.0
    inc_times_idcosvm.append(t_inc_id)
    
    # COSVM full batch refit
    t0 = time.perf_counter()
    cosvm_temp = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=idcosvm_stream.kernel_.gamma)
    cosvm_temp.fit(buf_arr)
    t_b_cosvm = (time.perf_counter() - t0) * 1000.0
    inc_times_batch_cosvm.append(t_b_cosvm)
    
    # Standard one-class SVM refit
    t0 = time.perf_counter()
    std_ocsvm_temp = COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma=idcosvm_stream.kernel_.gamma)
    std_ocsvm_temp.fit(buf_arr)
    t_b_std_ocsvm = (time.perf_counter() - t0) * 1000.0
    inc_times_refit_ocsvm.append(t_b_std_ocsvm)

# Decremental unlearning phase
dec_times_icosvm = []
dec_times_idcosvm = []
dec_times_batch_cosvm = []
dec_buffer_sizes = []

for step in range(30):
    buf_size_before = len(idcosvm_stream.X_buffer_)
    dec_buffer_sizes.append(buf_size_before)
    
    # ICOSVM unlearn
    t0 = time.perf_counter()
    icosvm_stream.unlearn_one(0)
    t_dec_i = (time.perf_counter() - t0) * 1000.0
    dec_times_icosvm.append(t_dec_i)
    
    # IDCOSVM forget_one
    t0 = time.perf_counter()
    idcosvm_stream.forget_one(0)
    t_dec_id = (time.perf_counter() - t0) * 1000.0
    dec_times_idcosvm.append(t_dec_id)
    
    # Batch COSVM refit
    rem_buf = np.array(idcosvm_stream.X_buffer_)
    t0 = time.perf_counter()
    cosvm_temp = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma=idcosvm_stream.kernel_.gamma)
    cosvm_temp.fit(rem_buf)
    t_dec_b = (time.perf_counter() - t0) * 1000.0
    dec_times_batch_cosvm.append(t_dec_b)

results["benchmark2"] = {
    "buffer_sizes": buffer_sizes,
    "inc_times_icosvm": inc_times_icosvm,
    "inc_times_idcosvm": inc_times_idcosvm,
    "inc_times_batch_cosvm": inc_times_batch_cosvm,
    "inc_times_refit_ocsvm": inc_times_refit_ocsvm,
    "dec_buffer_sizes": dec_buffer_sizes,
    "dec_times_icosvm": dec_times_icosvm,
    "dec_times_idcosvm": dec_times_idcosvm,
    "dec_times_batch_cosvm": dec_times_batch_cosvm
}

# Plot 4: Incremental Step Latency
plt.figure(figsize=(10, 5))
plt.plot(buffer_sizes, inc_times_icosvm, 'm-.', label='ICOSVM Incremental Step (learn_one)', linewidth=2)
plt.plot(buffer_sizes, inc_times_idcosvm, 'g-', label='IDCOSVM Incremental Step (learn_one)', linewidth=2)
plt.plot(buffer_sizes, inc_times_batch_cosvm, 'r--', label='COSVM Batch Refit (Cold Start)', linewidth=1.5, alpha=0.8)
plt.plot(buffer_sizes, inc_times_refit_ocsvm, 'b:', label='Standard OCSVM Full Re-fit (eta=1.0)', linewidth=1.5, alpha=0.8)
plt.xlabel('Current Active Buffer Size (N)')
plt.ylabel('Step Latency (ms)')
plt.title('Incremental Step Latency: ICOSVM & IDCOSVM vs Batch Refits')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig4_incremental_latency.png', dpi=200)
plt.close()

# Plot 5: Decremental Unlearning Latency
plt.figure(figsize=(10, 5))
plt.plot(dec_buffer_sizes, dec_times_icosvm, '^-', color='#984ea3', label='ICOSVM Unlearning (unlearn_one)', linewidth=2)
plt.plot(dec_buffer_sizes, dec_times_idcosvm, 'o-', color='#1b9e77', label='IDCOSVM Unlearning (forget_one)', linewidth=2)
plt.plot(dec_buffer_sizes, dec_times_batch_cosvm, 's--', color='#d95f02', label='COSVM Full Refit from Scratch', linewidth=1.5)
plt.xlabel('Active Buffer Size Before Deletion')
plt.ylabel('Unlearning Step Latency (ms)')
plt.title('Decremental Unlearning Latency: ICOSVM & IDCOSVM vs Cold Refit')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig5_decremental_unlearning.png', dpi=200)
plt.close()


# ==============================================================================
# BENCHMARK 3: Anomaly Detection Performance & Anisotropy Ratio (Normal Distribution)
# ==============================================================================
print("\n--- Running Benchmark 3: Detection Metrics Across Anisotropy Ratios ---")
anisotropy_ratios = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

std_ocsvm_roc_auc = []
cosvm_roc_auc = []
icosvm_roc_auc = []
idcosvm_roc_auc = []

std_ocsvm_bal_acc = []
cosvm_bal_acc = []
icosvm_bal_acc = []
idcosvm_bal_acc = []
selected_eta_by_ratio = []
eta_validation_auc_by_ratio = []

for ratio_index, ratio in enumerate(anisotropy_ratios):
    X_train_norm, X_test_norm, X_test_anom = make_anisotropic_anomaly_dataset(
        n_normal=150, n_anomalies=40, dimension=2, ratio=ratio, random_state=42
    )

    X_validation, y_validation, X_test, y_test = split_validation_test(
        X_test_norm, X_test_anom, random_state=100 + ratio_index
    )
    best_eta, validation_auc, _ = select_eta(
        X_train_norm, X_validation, y_validation, gamma="scale"
    )
    selected_eta_by_ratio.append(best_eta)
    eta_validation_auc_by_ratio.append(validation_auc)
    
    # 1. Standard One-Class SVM (eta=1.0)
    std_svm = COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma="scale")
    std_svm.fit(X_train_norm)
    scores_svm = std_svm.decision_function(X_test)
    preds_svm = std_svm.predict(X_test)
    
    auc_std_ocsvm = compute_roc_auc((y_test == 1).astype(int), scores_svm)
    bacc_std_ocsvm = compute_balanced_accuracy(y_test, preds_svm)
    std_ocsvm_roc_auc.append(auc_std_ocsvm)
    std_ocsvm_bal_acc.append(bacc_std_ocsvm)
    
    # Use validation-selected eta for batch and streaming covariance-guided models.
    cosvm = COSVM(nu=0.1, eta=best_eta, kernel="rbf", gamma="scale")
    cosvm.fit(X_train_norm)
    scores_cosvm = cosvm.decision_function(X_test)
    preds_cosvm = cosvm.predict(X_test)
    
    auc_cosvm = compute_roc_auc((y_test == 1).astype(int), scores_cosvm)
    bacc_cosvm = compute_balanced_accuracy(y_test, preds_cosvm)
    cosvm_roc_auc.append(auc_cosvm)
    cosvm_bal_acc.append(bacc_cosvm)
    
    icosvm = ICOSVM(nu=0.1, eta=best_eta, kernel="rbf", gamma="scale")
    icosvm.fit(X_train_norm)
    scores_icosvm = icosvm.decision_function(X_test)
    preds_icosvm = icosvm.predict(X_test)
    
    auc_icosvm = compute_roc_auc((y_test == 1).astype(int), scores_icosvm)
    bacc_icosvm = compute_balanced_accuracy(y_test, preds_icosvm)
    icosvm_roc_auc.append(auc_icosvm)
    icosvm_bal_acc.append(bacc_icosvm)
    
    idcosvm = IncrementalDecrementalCOSVM(nu=0.1, eta=best_eta, kernel="rbf", gamma="scale")
    idcosvm.fit(X_train_norm)
    scores_idcosvm = idcosvm.decision_function(X_test)
    preds_idcosvm = idcosvm.predict(X_test)
    
    auc_idcosvm = compute_roc_auc((y_test == 1).astype(int), scores_idcosvm)
    bacc_idcosvm = compute_balanced_accuracy(y_test, preds_idcosvm)
    idcosvm_roc_auc.append(auc_idcosvm)
    idcosvm_bal_acc.append(bacc_idcosvm)
    
    print(f"  Ratio: {ratio:4.1f} | selected eta: {best_eta:.2f} (val AUC {validation_auc:.4f}) | Std OCSVM AUC: {auc_std_ocsvm:.4f} | COSVM AUC: {auc_cosvm:.4f} | ICOSVM AUC: {auc_icosvm:.4f} | IDCOSVM AUC: {auc_idcosvm:.4f}")

results["benchmark3"] = {
    "anisotropy_ratios": anisotropy_ratios,
    "selected_eta": selected_eta_by_ratio,
    "eta_validation_roc_auc": eta_validation_auc_by_ratio,
    "eta_candidates": list(ETA_CANDIDATES),
    "standard_ocsvm_roc_auc": std_ocsvm_roc_auc,
    "cosvm_roc_auc": cosvm_roc_auc,
    "icosvm_roc_auc": icosvm_roc_auc,
    "idcosvm_roc_auc": idcosvm_roc_auc,
    "standard_ocsvm_bal_acc": std_ocsvm_bal_acc,
    "cosvm_bal_acc": cosvm_bal_acc,
    "icosvm_bal_acc": icosvm_bal_acc,
    "idcosvm_bal_acc": idcosvm_bal_acc
}

# Plot 6: ROC-AUC vs Anisotropy Ratio
plt.figure(figsize=(10, 5))
plt.plot(anisotropy_ratios, std_ocsvm_roc_auc, 'o--', label='Standard OCSVM (eta = 1.0)', color='#e41a1c', linewidth=2)
plt.plot(anisotropy_ratios, cosvm_roc_auc, 's-', label='COSVM (validation-selected eta)', color='#377eb8', linewidth=2)
plt.plot(anisotropy_ratios, icosvm_roc_auc, 'D:', label='ICOSVM (validation-selected eta)', color='#984ea3', linewidth=2)
plt.plot(anisotropy_ratios, idcosvm_roc_auc, '^-.', label='IDCOSVM (validation-selected eta)', color='#4daf4a', linewidth=2)
plt.xlabel('Anisotropy Variance Ratio (sigma_1 / sigma_2)')
plt.ylabel('ROC-AUC Score')
plt.title('Anomaly Detection ROC-AUC across Anisotropic Ratios')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig6_anisotropy_roc_auc.png', dpi=200)
plt.close()

# Plot 7: Balanced Accuracy vs Anisotropy Ratio
plt.figure(figsize=(10, 5))
plt.plot(anisotropy_ratios, std_ocsvm_bal_acc, 'o--', label='Standard OCSVM (eta = 1.0)', color='#e41a1c', linewidth=2)
plt.plot(anisotropy_ratios, cosvm_bal_acc, 's-', label='COSVM (validation-selected eta)', color='#377eb8', linewidth=2)
plt.plot(anisotropy_ratios, icosvm_bal_acc, 'D:', label='ICOSVM (validation-selected eta)', color='#984ea3', linewidth=2)
plt.plot(anisotropy_ratios, idcosvm_bal_acc, '^-.', label='IDCOSVM (validation-selected eta)', color='#4daf4a', linewidth=2)
plt.xlabel('Anisotropy Variance Ratio (sigma_1 / sigma_2)')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy across Variance Anisotropy Ratios')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig7_balanced_accuracy.png', dpi=200)
plt.close()


# ==============================================================================
# BENCHMARK 4: Multi-Distribution Evaluation & Decision Boundaries
# (Moon, Circle, Anisotropic Normal Distributions for standard OCSVM, COSVM, ICOSVM, IDCOSVM)
# ==============================================================================
print("\n--- Running Benchmark 4: Multi-Distribution Benchmark (Moon, Circle, Normal) ---")

def generate_distribution_data(dist_type: str, n_train: int = 150, n_outliers: int = 35, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    if dist_type == "moon":
        X_all_m, _ = make_benchmark_moons(n_samples=n_train * 2, noise=0.08, random_state=random_state)
        X_train = X_all_m[:n_train]
        X_test_normal = X_all_m[n_train:n_train + 60]
        # Generate outliers uniformly surrounding moons
        X_outliers = rng.uniform(low=[-1.5, -1.0], high=[2.5, 1.5], size=(n_outliers, 2))
        # Keep outliers that are not right on the moon curves
        dists = np.min(np.linalg.norm(X_outliers[:, None, :] - X_train[None, :, :], axis=2), axis=1)
        X_outliers = X_outliers[dists > 0.35][:n_outliers]
        if len(X_outliers) < n_outliers:
            extra = rng.uniform(low=[-1.5, -1.0], high=[2.5, 1.5], size=(n_outliers * 2, 2))
            X_outliers = np.vstack([X_outliers, extra])[:n_outliers]

    elif dist_type == "moon_classes":
        # Two Moons where every moon is for a given class:
        # Class +1: Upper Moon, Class -1: Lower Moon
        # Upper moon (Class +1)
        theta_upper = np.linspace(0, np.pi, n_train + 60)
        x_upper = np.cos(theta_upper) + rng.randn(n_train + 60) * 0.08
        y_upper = np.sin(theta_upper) + rng.randn(n_train + 60) * 0.08
        moon_pos = np.column_stack([x_upper, y_upper])

        # Lower moon (Class -1)
        theta_lower = np.linspace(0, np.pi, n_outliers * 3)
        x_lower = 1.0 - np.cos(theta_lower) + rng.randn(n_outliers * 3) * 0.08
        y_lower = 0.5 - np.sin(theta_lower) - 0.5 + rng.randn(n_outliers * 3) * 0.08
        moon_neg = np.column_stack([x_lower, y_lower])

        # Train on Class +1 moon (and evaluate on held-out Class +1 and Class -1 moon points)
        X_train = moon_pos[:n_train]
        X_test_normal = moon_pos[n_train:n_train + 60]
        # Moon of the other class (-1) serve as test outliers / opposing class samples
        X_outliers = moon_neg[:n_outliers]
            
    elif dist_type == "circle":
        X_all_c, _ = make_benchmark_circles(n_samples=n_train * 2, noise=0.05, factor=0.5, random_state=random_state)
        X_train = X_all_c[:n_train]
        X_test_normal = X_all_c[n_train:n_train + 60]
        # Outliers outside or in between rings
        X_outliers = rng.uniform(low=[-1.6, -1.6], high=[1.6, 1.6], size=(n_outliers * 2, 2))
        radii = np.linalg.norm(X_outliers, axis=1)
        valid_out = (radii > 1.25) | (radii < 0.25) | ((radii > 0.65) & (radii < 0.85))
        X_outliers = X_outliers[valid_out][:n_outliers]
        if len(X_outliers) < n_outliers:
            X_outliers = rng.uniform(low=[-1.8, -1.8], high=[1.8, 1.8], size=(n_outliers, 2))
            
    elif dist_type == "normal":
        X_train, X_test_normal, X_outliers = make_anisotropic_anomaly_dataset(
            n_normal=n_train, n_anomalies=n_outliers, dimension=2, ratio=8.0, random_state=random_state
        )

    elif dist_type == "gaussian_mixture":
        # Multi-modal / Gaussian Mixture with 3 distinct clusters
        centers = np.array([[-1.8, -1.2], [1.8, -1.0], [0.0, 1.8]])
        covs = [
            np.array([[0.18, 0.08], [0.08, 0.12]]),
            np.array([[0.15, -0.06], [-0.06, 0.20]]),
            np.array([[0.25, 0.0], [0.0, 0.10]])
        ]
        n_per_cluster = n_train // 3
        X_tr_list, X_te_list = [], []
        for i in range(3):
            pts = rng.multivariate_normal(centers[i], covs[i], size=n_per_cluster + 20)
            X_tr_list.append(pts[:n_per_cluster])
            X_te_list.append(pts[n_per_cluster:n_per_cluster + 20])
        X_train = np.vstack(X_tr_list)
        X_test_normal = np.vstack(X_te_list)
        
        # Outliers placed in between cluster gaps and outer envelope
        candidates = rng.uniform(low=[-3.0, -2.5], high=[3.0, 3.0], size=(n_outliers * 4, 2))
        dists = np.min(np.linalg.norm(candidates[:, None, :] - X_train[None, :, :], axis=2), axis=1)
        valid = (dists > 0.65)
        X_outliers = candidates[valid][:n_outliers]
        if len(X_outliers) < n_outliers:
            X_outliers = candidates[:n_outliers]

    elif dist_type == "spiral":
        # Two-arm non-linear Archimedean spiral
        n_arm = (n_train + 60) // 2
        theta1 = np.sqrt(rng.rand(n_arm)) * (2.8 * np.pi) + 0.3
        r1 = theta1
        x1 = r1 * np.cos(theta1) + rng.randn(n_arm) * 0.22
        y1 = r1 * np.sin(theta1) + rng.randn(n_arm) * 0.22
        arm1 = np.column_stack([x1, y1])

        theta2 = np.sqrt(rng.rand(n_arm)) * (2.8 * np.pi) + 0.3
        r2 = theta2
        x2 = -r2 * np.cos(theta2) + rng.randn(n_arm) * 0.22
        y2 = -r2 * np.sin(theta2) + rng.randn(n_arm) * 0.22
        arm2 = np.column_stack([x2, y2])

        all_spiral = np.vstack([arm1, arm2]) / 3.0
        perm = rng.permutation(len(all_spiral))
        all_spiral = all_spiral[perm]
        X_train = all_spiral[:n_train]
        X_test_normal = all_spiral[n_train:n_train + 60]

        # Outliers in the interleaved void and outside
        candidates = rng.uniform(low=[-3.2, -3.2], high=[3.2, 3.2], size=(n_outliers * 4, 2))
        dists = np.min(np.linalg.norm(candidates[:, None, :] - X_train[None, :, :], axis=2), axis=1)
        valid = (dists > 0.45)
        X_outliers = candidates[valid][:n_outliers]
        if len(X_outliers) < n_outliers:
            X_outliers = candidates[:n_outliers]

    elif dist_type == "sinusoidal":
        # Non-linear Sinusoidal Wave Ribbon
        x_tr = rng.uniform(-3.0, 3.0, size=n_train + 60)
        y_tr = np.sin(1.8 * x_tr) + 0.3 * np.cos(0.9 * x_tr) + rng.randn(n_train + 60) * 0.16
        all_wave = np.column_stack([x_tr, y_tr])
        X_train = all_wave[:n_train]
        X_test_normal = all_wave[n_train:n_train + 60]

        # Outliers above and below wave troughs/peaks
        x_out = rng.uniform(-3.2, 3.2, size=n_outliers * 3)
        y_out = rng.uniform(-2.2, 2.2, size=n_outliers * 3)
        cand = np.column_stack([x_out, y_out])
        dists = np.min(np.linalg.norm(cand[:, None, :] - X_train[None, :, :], axis=2), axis=1)
        valid = (dists > 0.45)
        X_outliers = cand[valid][:n_outliers]
        if len(X_outliers) < n_outliers:
            X_outliers = cand[:n_outliers]

    else:
        raise ValueError(f"Unknown distribution {dist_type}")
        
    return X_train, X_test_normal, X_outliers


distributions = ["moon", "moon_classes", "circle", "normal", "gaussian_mixture", "spiral", "sinusoidal"]
dist_metrics = {}
selected_eta_by_distribution = {}
eta_selection_details = {}

for dist_index, dist_name in enumerate(distributions):
    X_tr, X_te_norm, X_te_out = generate_distribution_data(dist_name, n_train=150, n_outliers=35, random_state=42)
    X_validation, y_validation, X_eval, y_eval = split_validation_test(
        X_te_norm, X_te_out, random_state=200 + dist_index
    )
    
    if dist_name == "circle":
        gamma_val = 1.5
    elif dist_name == "normal":
        gamma_val = "scale"
    elif dist_name == "spiral":
        gamma_val = 1.2
    elif dist_name == "gaussian_mixture":
        gamma_val = 0.8
    elif dist_name == "sinusoidal":
        gamma_val = 1.0
    elif dist_name == "moon_classes":
        gamma_val = 1.0
    else:
        gamma_val = "scale"

    eta_val, eta_validation_auc, eta_candidate_aucs = select_eta(
        X_tr, X_validation, y_validation, gamma=gamma_val
    )
    selected_eta_by_distribution[dist_name] = eta_val
    eta_selection_details[dist_name] = {
        "selected_eta": eta_val,
        "validation_roc_auc": eta_validation_auc,
        "candidate_validation_roc_auc": eta_candidate_aucs,
    }

    m_svm = COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_cosvm = COSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_icosvm = ICOSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_idcosvm = IncrementalDecrementalCOSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    
    models_dict = {
        "Standard OCSVM (eta=1.0)": m_svm,
        "COSVM (eta selected)": m_cosvm,
        "ICOSVM (eta selected)": m_icosvm,
        "IDCOSVM (eta selected)": m_idcosvm
    }
    
    dist_metrics[dist_name] = {}
    for m_name, model in models_dict.items():
        scores = model.decision_function(X_eval)
        preds = model.predict(X_eval)
            
        auc = compute_roc_auc((y_eval == 1).astype(int), scores)
        bacc = compute_balanced_accuracy(y_eval, preds)
        acc = accuracy_score(y_eval, preds)
        
        n_sv = len(model.support_)
        sv_ratio = n_sv / len(X_tr)
        
        dist_metrics[dist_name][m_name] = {
            "eta": 1.0 if m_name == "Standard OCSVM (eta=1.0)" else eta_val,
            "roc_auc": float(auc),
            "balanced_accuracy": float(bacc),
            "accuracy": float(acc),
            "n_support_vectors": int(n_sv),
            "sv_ratio": float(sv_ratio)
        }
        print(f"  [{dist_name.upper():<16}] {m_name:<24} eta={model.eta:.2f} | test AUC: {auc:.4f} | BalAcc: {bacc*100:.2f}% | SVs: {n_sv}")

results["benchmark4_distributions"] = dist_metrics
results["benchmark4_eta_selection"] = {
    "method": "validation ROC-AUC; highest eta wins ties",
    "candidates": list(ETA_CANDIDATES),
    "by_distribution": eta_selection_details,
}

# Plot 8: Multi-Distribution Decision Boundaries Grid (7 distributions x 4 models)
fig, axes = plt.subplots(7, 4, figsize=(20, 30))

dist_titles = {
    "moon": "Two Moons (Inlier Envelope)",
    "moon_classes": "Moons Per Class (Upper vs Lower Moon)",
    "circle": "Concentric Circles Distribution",
    "normal": "Anisotropic Normal Distribution",
    "gaussian_mixture": "Gaussian Mixture (3-Cluster)",
    "spiral": "Two-Arm Spiral Manifold",
    "sinusoidal": "Sinusoidal Wave Ribbon"
}

# Also generate standalone absolute demo plots like svm_rbf_nonlinear_demo.png for each distribution and model
demo_figures_dir = "benchmark_figures"

for row_idx, dist_name in enumerate(distributions):
    X_tr, X_te_norm, X_te_out = generate_distribution_data(dist_name, n_train=150, n_outliers=35, random_state=42)
    X_all_pts = np.vstack([X_tr, X_te_norm, X_te_out])
    
    x_min, x_max = X_all_pts[:, 0].min() - 0.8, X_all_pts[:, 0].max() + 0.8
    y_min, y_max = X_all_pts[:, 1].min() - 0.8, X_all_pts[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    eta_val = selected_eta_by_distribution[dist_name]
    if dist_name == "circle":
        gamma_val = 1.5
    elif dist_name == "normal":
        gamma_val = "scale"
    elif dist_name == "spiral":
        gamma_val = 1.2
    elif dist_name == "gaussian_mixture":
        gamma_val = 0.8
    elif dist_name == "sinusoidal":
        gamma_val = 1.0
    elif dist_name == "moon_classes":
        gamma_val = 1.0
    else:
        gamma_val = "scale"

    m_svm = COSVM(nu=0.1, eta=1.0, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_cosvm = COSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_icosvm = ICOSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    m_idcosvm = IncrementalDecrementalCOSVM(nu=0.1, eta=eta_val, kernel="rbf", gamma=gamma_val).fit(X_tr)
    
    row_models = [
        ("Standard OCSVM (eta=1.0)", m_svm, "svm"),
        ("COSVM (eta tuned per dataset)", m_cosvm, "cosvm"),
        ("ICOSVM (eta tuned per dataset)", m_icosvm, "icosvm"),
        ("IDCOSVM (eta tuned per dataset)", m_idcosvm, "idcosvm")
    ]
    
    for col_idx, (m_label, model, model_slug) in enumerate(row_models):
        ax = axes[row_idx, col_idx]
        Z = model.decision_function(grid).reshape(xx.shape)
        
        # Grid plot styling
        ax.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=["#f4a582", "#92c5de"], alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0.0], colors=["black"],
                   linestyles=["-"], linewidths=[2.2])
        
        # Plot data points
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c="#2b83ba", edgecolors="k", s=35, alpha=0.8, label="Target / Inliers (+1)")
        ax.scatter(X_te_out[:, 0], X_te_out[:, 1], c="#d7191c", s=40, marker='x', linewidths=1.8, label="Outliers (-1)")
        
        # Support vectors
        sv_idx = model.support_
        sv_pts = X_tr[sv_idx]
        ax.scatter(sv_pts[:, 0], sv_pts[:, 1], s=90, facecolors='none', edgecolors='gold', linewidths=2.0, label=f"SVs ({len(sv_idx)})")
        
        if row_idx == 0:
            ax.set_title(m_label, fontsize=13, fontweight='bold', pad=10)
        if col_idx == 0:
            ax.set_ylabel(f"{dist_titles[dist_name]}\nFeature 2", fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel("Feature 2", fontsize=10)
        ax.set_xlabel("Feature 1", fontsize=10)
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Standalone Demo Figure (matching svm_rbf_nonlinear_demo.png style)
        # -------------------------------------------------------------
        fig_single, ax_s = plt.subplots(figsize=(8, 6))
        ax_s.contourf(xx, yy, Z > 0, levels=[-0.5, 0.5, 1.5], colors=["#f4a582", "#92c5de"], alpha=0.25)
        c_line = ax_s.contour(
            xx, yy, Z,
            levels=[0.0],
            linestyles=["-"],
            colors=["black"],
            linewidths=[2.4]
        )
        ax_s.scatter(X_tr[:, 0], X_tr[:, 1], c="blue", edgecolors="k", s=50, alpha=0.85, label="Target / Inliers (+1)")
        ax_s.scatter(X_te_out[:, 0], X_te_out[:, 1], c="red", s=55, marker='x', linewidths=1.8, label="Anomalies / Outliers (-1)")
        ax_s.scatter(
            sv_pts[:, 0], sv_pts[:, 1],
            s=130, facecolors="none", edgecolors="gold", linewidths=2.2,
            label=f"Support Vectors ({len(sv_idx)})"
        )
        eta_title = "" if model.eta == 1.0 and model is m_svm else f" (eta={model.eta:g})"
        ax_s.set_title(f"{m_label}{eta_title} Acceptance Boundary ({dist_titles[dist_name]})", fontsize=13, fontweight="bold")
        ax_s.set_xlabel("Feature 1 (X1)", fontsize=11)
        ax_s.set_ylabel("Feature 2 (X2)", fontsize=11)
        ax_s.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax_s.grid(True, linestyle=":", alpha=0.6)
        
        standalone_path = os.path.join(demo_figures_dir, f"demo_{dist_name}_{model_slug}.png")
        fig_single.tight_layout()
        fig_single.savefig(standalone_path, dpi=180, bbox_inches="tight")
        plt.close(fig_single)

fig.tight_layout()
fig.savefig('benchmark_figures/fig8_decision_boundaries_comparison.png', dpi=200)
plt.close(fig)


# Plot 10: Quantitative Performance across Distributions (Bar chart grid: 2 rows x 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
model_names_plot = [
    "Standard OCSVM (eta=1.0)",
    "COSVM (eta selected)",
    "ICOSVM (eta selected)",
    "IDCOSVM (eta selected)",
]
colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']

for idx, dist_name in enumerate(distributions):
    r, c = divmod(idx, 4)
    ax = axes[r, c]
    metric_keys = [
        "Standard OCSVM (eta=1.0)",
        "COSVM (eta selected)",
        "ICOSVM (eta selected)",
        "IDCOSVM (eta selected)",
    ]
    aucs = [dist_metrics[dist_name][m]["roc_auc"] for m in metric_keys]
    baccs = [dist_metrics[dist_name][m]["balanced_accuracy"] * 100 for m in metric_keys]
    
    x = np.arange(len(model_names_plot))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, [a * 100 for a in aucs], width, label='ROC-AUC (%)', color='#3b528b')
    rects2 = ax.bar(x + width/2, baccs, width, label='Balanced Acc (%)', color='#5ec962')
    
    ax.set_title(f"{dist_titles[dist_name]}", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_plot, rotation=15, ha="right")
    ax.set_ylim(50, 105)
    ax.set_ylabel("Metric Score (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    if idx == 0:
        ax.legend(loc="lower right")

# Hide unused 8th subplot
axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig('benchmark_figures/fig10_distribution_performance_bars.png', dpi=200)
plt.close()


# ==============================================================================
# BENCHMARK 5: Pruning Policies Comparison in Buffer Management
# ==============================================================================
print("\n--- Running Benchmark 5: Pruning Strategies ---")
pruning_policies = ['fifo', 'non_sv_first', 'error_first', 'mahalanobis']
buffer_cap = 60
stream_len = 150

policy_results = {}

for policy in pruning_policies:
    idcosvm_pruner = IncrementalDecrementalCOSVM(
        nu=0.1, eta=0.4, kernel="rbf", gamma="scale",
        max_samples=buffer_cap, pruning_strategy=policy
    )
    
    X_p_stream, _, _ = make_anisotropic_anomaly_dataset(n_normal=stream_len, n_anomalies=0, dimension=2, ratio=6.0, random_state=123)
    
    kkt_residuals = []
    sv_counts = []
    
    for x in X_p_stream:
        idcosvm_pruner.learn_one(x)
        kkt = idcosvm_pruner._kkt_residuals()
        kkt_residuals.append(kkt['max_stationarity_error'])
        sv_counts.append(len(idcosvm_pruner.support_))
        
    policy_results[policy] = {
        "kkt_residuals": kkt_residuals,
        "sv_counts": sv_counts,
        "final_sv_count": len(idcosvm_pruner.support_),
        "buffer_size": len(idcosvm_pruner.X_buffer_)
    }

results["benchmark5"] = {
    "pruning_policies": pruning_policies,
    "policy_results": policy_results
}

# Plot 9: SV Count dynamics under buffer retention policies
plt.figure(figsize=(10, 5))
for policy in pruning_policies:
    plt.plot(range(stream_len), policy_results[policy]["sv_counts"], label=f"Policy: {policy}", linewidth=2)
plt.axvline(x=buffer_cap, color='gray', linestyle=':', label=f'Max Buffer Capacity (W={buffer_cap})')
plt.xlabel('Stream Sample Step t')
plt.ylabel('Support Vector Count |S|')
plt.title('Support Vector Dynamics Under Pruning Policies (Max Capacity = 60)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('benchmark_figures/fig9_pruning_policies_svs.png', dpi=200)
plt.close()

# Save all results to JSON
with open('benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAll benchmarks executed and figures generated successfully!")
