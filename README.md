# DynamicSVM: A Modular Python Library for Static & Dynamic SVMs

**DynamicSVM** is an extensible, mathematically rigorous Python library designed for academic research (PhD thesis) and practical machine learning applications. It provides full implementations of standard Support Vector Machines (dual quadratic solvers, Sequential Minimal Optimization, primal SGD, and kernel functions) alongside an architecture for **Incremental Learning**, **Decremental Learning (Unlearning)**, **Online Covariance Tracking**, and **Concept Drift Adaptation** based on latent-space monitoring.

---

## 🚀 Key Features

- **Representation Learning & Frozen Encoders**:
  - `AutoEncoder` / `MLPAutoEncoder`: Multilayer neural autoencoder in pure NumPy with Adam optimization.
  - `PCAEncoder`: Linear orthogonal subspace projection encoder.
  - `IdentityEncoder`: Passthrough representation wrapper.
  - `freeze()` / `unfreeze()` coordinate stabilization for downstream streaming SVMs.
- **Latent Distribution Monitoring & Drift Detection**:
  - `OnlineCovarianceTracker`: Streaming mean $\mu_t$ and covariance $\Sigma_t$ estimation with drift-adaptive exponential forgetting $\lambda(D_t)$.
  - `LatentCovarianceDriftDetector`: Statistical discrepancy monitoring ($D_t$) separating warmup, warning, and confirmed drift regimes.
  - `ObsolescenceScorer`: Multi-criteria obsolescence scoring ($O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$) with support vector protection for controlled decremental unlearning.
- **Covariance-Aware Support Vector Machines**:
  - `CovarianceAwareSVC`: Optimization on covariance-guided Gram matrices $Q = \eta K + (1 - \eta) K H K^T$.
  - `DriftAwareCovarianceSVM`: End-to-end streaming architecture ($x \to \text{Frozen AE} \to z \to \text{Covariance Tracker} \to \text{Obsolescence Pruning} \to \text{Incremental Covariance SVM}$).
  - `COSVM` & `ICOSVM`: Covariance-guided One-Class Support Vector Machine and Incremental One-Class SVM based on low-variance directions (*Kefi-Fatteh et al., Pattern Recognition 2019*).
  - `IncrementalDecrementalCOSVM` / `IDCOSVM`: Complete online incremental learning and exact decremental unlearning with multi-policy obsolescence pruning (`fifo`, `non_sv_first`, `error_first`, `mahalanobis`).
- **Exact Incremental / Decremental & Online Solvers**:
  - `AdiabaticSVM` / `IncrementalDecrementalSVC`: Exact Kuhn-Tucker relations for continuous adiabatic migration of $\alpha_i$ and $b$ upon sample insertion and deletion (*Cauwenberghs & Poggio 2001, Laskov et al. 2006*).
  - `LASVM` / `LASVMClassifier`: Online and active learning support vector classifier with support vector filtering and fast KKT directional optimization (*Bordes et al., JMLR 2005*).
- **Benchmarking Suite & Streaming Evaluation Protocols**:
  - **Prequential (Test-Then-Train) Streaming Evaluator**: `PrequentialEvaluator` tracking rolling/cumulative accuracy, sample step latency, throughput (samples/sec), and support vector dynamics.
  - **Benchmark Dataset Generators**: Synthetic concept drift streams (`make_sea_stream`, `make_rotating_hyperplane_stream`, `make_gaussian_drift_stream`), anisotropic low-variance anomaly datasets (`make_anisotropic_anomaly_dataset`), and non-linear classification benchmarks (`make_benchmark_moons`, `make_benchmark_circles`).
  - **Comparative Benchmark Runners**: `benchmark_streaming_suite`, `benchmark_anomaly_suite`, `benchmark_solvers_suite`.
- **Standard SVM & Kernels**:
  - Binary & Multi-Class Support Vector Classification (`SVC`, `MultiClassSVC` with One-vs-Rest and One-vs-One).
  - Built-in kernels: Linear, Radial Basis Function (RBF / Gaussian), Polynomial, Sigmoid, and Custom User Kernels.
- **Multiple Optimization Solvers**:
  - **Sequential Minimal Optimization (SMO)**: Platt's algorithm with dual variable caching, two-loop heuristic selection, and KKT violation tracking.
  - **Quadratic Programming (QP)**: Exact reference dual solver via SLSQP optimization.
  - **Pegasos SGD**: Fast primal stochastic subgradient solver.
- **Diagnostics & Visualization**:
  - Decision boundary, margin line ($\pm 1$), and support vector visualization (`plot_decision_boundary`).
  - Classification metrics: accuracy, confusion matrix, hinge loss.

---

## 📐 Mathematical Formulation

### 1. Dual SVM & Covariance Geometry
For training samples $(x_i, y_i)_{i=1}^n$ with $y_i \in \{-1, +1\}$:

$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j Q(z_i, z_j) - \sum_{i=1}^n \alpha_i$$

Subject to:
$$0 \le \alpha_i \le C, \quad \forall i=1,\dots,n$$
$$\sum_{i=1}^n \alpha_i y_i = 0$$

Where the covariance-guided Gram matrix is:
$$Q = \eta K + (1 - \eta) \Delta, \quad \Delta = K H K^T, \quad H = \frac{1}{d} Z (\Sigma_t + \epsilon I)^{-1} Z^T$$

### 2. Online Latent Covariance & Drift Metric
Incoming batches or samples are evaluated against current latent statistics:
$$\mu_t = \mathbb{E}[z_t], \quad \Sigma_t = \mathbb{E}[(z_t - \mu_t)(z_t - \mu_t)^T]$$
$$D_t = \lambda_\mu \|\mu_{B_t} - \mu_t\|_2^2 + \lambda_\Sigma \|\Sigma_{B_t} - \Sigma_t\|_F^2$$

Adaptive exponential forgetting:
$$\Sigma_t = \lambda(D_t) \Sigma_{t-1} + (1 - \lambda(D_t)) \Sigma_{B_t}$$
$$\lambda(D_t) = \max\left(\lambda_{\min}, \, \lambda_{\text{base}} \cdot e^{-\gamma_D D_t}\right)$$

### 3. Obsolescence-Guided Controlled Forgetting
Stored instances are assigned an obsolescence score:
$$O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$$
where $A_i$ is normalized age, $T_i$ is temporal decay distance, and $D_i$ is latent Mahalanobis inconsistency $d_M(z_i, \mu_t, \Sigma_t) = \sqrt{(z_i - \mu_t)^T (\Sigma_t + \epsilon I)^{-1} (z_i - \mu_t)}$.

### 4. Covariance-Guided One-Class SVM (COSVM & ICOSVM)
Based on *Kefi-Fatteh et al. (Pattern Recognition 2019)*, low variance directions carry decisive discriminative power for one-class anomaly detection:

$$\min_{\alpha} \frac{1}{2} \alpha^T \tilde{K} \alpha - b \left(1 - \sum_{i=1}^N \alpha_i\right)$$
$$\text{s.t. } 0 \le \alpha_i \le \frac{1}{\nu N}, \quad \sum_{i=1}^N \alpha_i = 1$$
where the covariance-augmented Gram matrix in feature space is:
$$\tilde{K} = \eta K + (1 - \eta) K \left(I - \frac{1}{N} \mathbf{1}\mathbf{1}^T\right) K^T$$

In **ICOSVM**, streaming updates maintain Karush-Kuhn-Tucker equilibrium across partitions $S$ (margin SVs), $E$ (error SVs), and $O$ (non-SVs) using adiabatic sensitivity vectors:
$$R = \begin{bmatrix} 0 & \mathbf{1}_S^T \\ \mathbf{1}_S & \tilde{K}_{S,S} \end{bmatrix}^{-1}, \quad \beta = -R \begin{bmatrix} 1 \\ \tilde{K}_{S,c} \end{bmatrix}, \quad \gamma_i = \tilde{K}_{i,c} + \sum_{j \in S} \tilde{K}_{ij} \beta_j$$

### 5. Exact Adiabatic Incremental & Decremental SVM (Cauwenberghs & Poggio)
Exact Kuhn-Tucker equilibrium migration for binary SVMs upon sample insertion and deletion:
$$\begin{bmatrix} 0 & \mathbf{y}_{\mathcal{S}}^T \\ \mathbf{y}_{\mathcal{S}} & \mathbf{Q}_{\mathcal{S}\mathcal{S}} \end{bmatrix} \begin{bmatrix} \Delta b \\ \Delta \boldsymbol{\alpha}_{\mathcal{S}} \end{bmatrix} = -\begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix} \Delta \alpha_c$$
$$\boldsymbol{\beta} = -R \begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix}, \quad \gamma_i = Q_{ic} + y_i \beta_0 + \sum_{j \in \mathcal{S}} Q_{ij} \beta_j$$

### 6. LASVM Online & Active Learning (Bordes et al. 2005)
Directional gradient updates and support vector filtering:
$$g_i = y_i - \sum_{j \in \mathcal{S}} \alpha_j y_j K(x_i, x_j)$$
$\tau$-violating pair selection: $\max_{i \in I_{\text{up}}} g_i - \min_{j \in I_{\text{low}}} g_j > \tau$.
Non-violating zero-$\alpha$ candidates are pruned during `REPROCESS` and `FINISH` steps.

---

## 📂 Project Structure

```
DynamicDVM/
├── pyproject.toml              # Build metadata and dependencies
├── README.md                   # Documentation and PhD roadmap
├── main.py                     # Demo runner with benchmarks & plots
├── dynamicsvm/                 # Core Python package
│   ├── __init__.py             # Public exports
│   ├── base.py                 # BaseEstimator, BaseClassifier, BaseSVM
│   ├── kernels.py              # LinearKernel, RBFKernel, PolynomialKernel, SigmoidKernel
│   ├── encoders/               # Representation Learning
│   │   ├── __init__.py
│   │   ├── base.py             # BaseEncoder (freeze, fit, encode, decode)
│   │   └── autoencoder.py      # AutoEncoder (MLP), PCAEncoder, IdentityEncoder
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── smo.py              # Sequential Minimal Optimization (Platt)
│   │   ├── qp.py               # Dual Quadratic Programming (scipy SLSQP)
│   │   ├── pegasos.py          # Primal Stochastic Subgradient Solver
│   │   └── lasvm.py            # LASVM Online & Active Solver (Bordes et al. 2005)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svc.py              # SupportVectorClassifier (binary)
│   │   ├── multiclass.py       # MultiClassSVC (OvR and OvO)
│   │   ├── covariance_svc.py   # CovarianceAwareSVC (covariance-guided geometry)
│   │   └── cosvm.py            # Covariance-guided One-Class SVM (Kefi-Fatteh et al. 2019)
│   ├── stream/
│   │   ├── __init__.py
│   │   ├── base.py             # StreamingSVM interface (learn_one / forget_one)
│   │   ├── windowed.py         # SlidingWindowSVM
│   │   ├── drift.py            # DDMDriftDetector & DriftAwareSVM
│   │   ├── covariance.py       # OnlineCovarianceTracker, LatentCovarianceDriftDetector, ObsolescenceScorer
│   │   ├── drift_covariance_svm.py # DriftAwareCovarianceSVM
│   │   ├── icosvm.py           # Incremental Covariance One-Class SVM (ICOSVM)
│   │   ├── idcosvm.py          # Incremental & Decremental Covariance One-Class SVM (IDCOSVM)
│   │   └── adiabatic_svm.py    # Adiabatic Path SVM (Cauwenberghs & Poggio 2001)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py       # Array checks and input sanitization
│   │   └── metrics.py          # Accuracy, Confusion Matrix, Hinge Loss
│   ├── benchmarks/             # Benchmarking Suite & Protocols
│   │   ├── __init__.py
│   │   ├── datasets.py         # SEA, Rotating Hyperplane, Gaussian Drift, Anisotropic datasets
│   │   ├── evaluator.py        # PrequentialEvaluator, BatchBenchmarkEvaluator, AnomalyBenchmarkEvaluator
│   │   └── runner.py           # Comparative benchmark suites
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # 2D decision boundary & support vector plots
├── benchmark_suite.py          # Standalone benchmark execution runner
└── tests/                      # Automated unit test suite
    ├── test_benchmarks.py
    ├── test_adiabatic_svm.py
    ├── test_lasvm.py
    ├── test_cosvm.py
    ├── test_icosvm.py
    ├── test_idcosvm.py
    ├── test_encoders.py
    ├── test_covariance.py
    ├── test_covariance_svc.py
    ├── test_drift_covariance_svm.py
    ├── test_kernels.py
    ├── test_solvers.py
    ├── test_svc.py
    ├── test_multiclass.py
    └── test_stream.py
```

---

## ⚡ Quick Start

### 1. Drift-Aware Covariance Incremental SVM
```python
import numpy as np
from dynamicsvm.encoders import AutoEncoder
from dynamicsvm.stream import DriftAwareCovarianceSVM

# Initial dataset
X_init = np.random.randn(80, 6)
y_init = np.array([1] * 40 + [0] * 40)

# Pretrain & freeze latent representation encoder
encoder = AutoEncoder(latent_dim=3, hidden_dims=(16,), max_iter=50)
encoder.fit(X_init)
encoder.freeze()

# Initialize dynamic drift-aware covariance SVM
model = DriftAwareCovarianceSVM(
    encoder=encoder,
    C=1.0,
    eta=0.7,
    drift_threshold=2.0,
    max_buffer_size=100,
)
model.fit(X_init, y_init)

# Streaming data arrival with online drift adaptation
for x_t, y_t in zip(X_stream, y_stream):
    model.learn_one(x_t, y_t)

print("Predictions:", model.predict(X_test))
print("Model State:", model.get_model_state())
```

### 2. Low-Variance Direction One-Class SVM (COSVM, ICOSVM & IDCOSVM)
```python
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM
from dynamicsvm.stream.idcosvm import IDCOSVM, IncrementalDecrementalCOSVM

# 1. Batch Covariance-guided One-Class SVM
cosvm = COSVM(nu=0.1, eta=0.4, kernel="rbf")
cosvm.fit(X_normal)
outlier_scores = cosvm.decision_function(X_test)  # >= 0 normal, < 0 outlier

# 2. Incremental One-Class SVM (Streaming)
icosvm = ICOSVM(nu=0.1, eta=0.4, kernel="rbf", max_samples=100)
icosvm.fit(X_initial)

# Online stream updates
for x_new in stream:
    icosvm.learn_one(x_new)

# 3. Incremental and Decremental One-Class SVM (IDCOSVM)
idcosvm = IDCOSVM(
    nu=0.1,
    eta=0.4,
    kernel="rbf",
    max_samples=50,
    pruning_strategy="non_sv_first",  # 'fifo', 'non_sv_first', 'error_first', 'mahalanobis'
)
idcosvm.fit(X_initial)

# Incremental learning
idcosvm.learn_batch(X_incoming)

# Exact decremental unlearning by index, sample value, or obsolescence policy
idcosvm.unlearn_one(index=0)
idcosvm.unlearn_sample(specific_vector)
idcosvm.unlearn_obsolete(n_samples=5, strategy="mahalanobis")
```

### 3. Exact Adiabatic Incremental & Decremental SVM
```python
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM

model = AdiabaticSVM(C=1.0, kernel="rbf", gamma="scale", max_samples=100)
model.fit(X_train, y_train)

# Online continuous adiabatic insertion
for x_t, y_t in zip(X_stream, y_stream):
    model.learn_one(x_t, y_t)

# Exact decremental unlearning without retraining
model.forget_one(index=0)
model.unlearn_sample(x_unwanted, y=y_unwanted)
```

### 4. LASVM Online & Active Learning
```python
from dynamicsvm.solvers.lasvm import LASVM

# Active learning with support vector filtering
lasvm = LASVM(C=1.0, kernel="rbf", epochs=2, selection_strategy="active")
lasvm.fit(X_train, y_train)

# Online stream updates
for x_t, y_t in zip(X_stream, y_stream):
    lasvm.learn_one(x_t, y_t)
```

### 5. Standard SVM with RBF Kernel
```python
from dynamicsvm.models.svc import SVC

model = SVC(C=1.0, kernel="rbf", gamma="scale", solver="smo")
model.fit(X, y)
predictions = model.predict(X)
```

### 6. Streaming & Anomaly Benchmarking Suite
```python
from dynamicsvm.benchmarks import (
    make_sea_stream,
    make_anisotropic_anomaly_dataset,
    benchmark_streaming_suite,
    benchmark_anomaly_suite,
)
from dynamicsvm.stream import SlidingWindowSVM, AdiabaticSVM
from dynamicsvm.models.cosvm import COSVM
from dynamicsvm.stream.icosvm import ICOSVM

# 1. Prequential streaming evaluation on concept drift streams
streaming_models = {
    "SlidingWindow": SlidingWindowSVM(window_size=40, kernel="linear"),
    "AdiabaticSVM": AdiabaticSVM(C=1.0, kernel="linear", max_samples=40),
}
results = benchmark_streaming_suite(streaming_models, dataset_name="sea", n_samples=500)
for row in results["summary"]:
    print(f"Model: {row['model']} | Cumulative Acc: {row['cumulative_accuracy']:.2%} | Throughput: {row['throughput_samples_per_sec']:.1f} samples/s")

# 2. Anomaly benchmark on low-variance manifolds
anomaly_models = {
    "COSVM": COSVM(nu=0.1, eta=0.4),
    "ICOSVM": ICOSVM(nu=0.1, eta=0.4),
}
anom_results = benchmark_anomaly_suite(anomaly_models, ratio=10.0)
```

---

## 🧪 Running Tests & Demo

```bash
# Run the complete test suite (57 tests)
pytest -v

# Run the visual demo and interactive experiments
python main.py

# Run the comprehensive comparative benchmark suite
python benchmark_suite.py
```

---

## 🔬 PhD Research Roadmap: Next Steps

1. **Advanced Concept Drift Mechanisms**:
   - EDDM (Early Drift Detection Method) & ADWIN (Adaptive Windowing).
   - Time-decayed kernel functions $K_t(x_i, x_j) = e^{-\lambda (t - t_i)} K(x_i, x_j)$ for continuous drift adaptation.
2. **Large Scale Streaming**:
   - Incremental Budgeted SVM (random Fourier features, Nyström approximation, and budget maintenance strategies).
