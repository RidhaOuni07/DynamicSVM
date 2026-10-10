# DynamicSVM: A Modular Python Library for Static & Dynamic SVMs

**DynamicSVM** is an extensible Python library (NumPy/SciPy) designed for academic research (PhD thesis) and practical machine learning applications. It provides implementations of standard Support Vector Machines (dual quadratic solvers, Sequential Minimal Optimization, primal SGD, and kernel functions) alongside an architecture for **Incremental Learning**, **Decremental Learning (Unlearning)**, **Online Covariance Tracking**, and **Concept Drift Adaptation** based on latent-space monitoring. Optimality of the dual solutions can be checked with `dynamicsvm.utils.kkt_report` / `oneclass_kkt_report`; the test suite compares the solvers with scikit-learn references.

---

## 🚀 Key Features

- **Representation Learning & Frozen Encoders**:
  - `AutoEncoder` / `MLPAutoEncoder`: Multilayer neural autoencoder in pure NumPy with Adam optimization.
  - `PCAEncoder`: Linear orthogonal subspace projection encoder.
  - `IdentityEncoder`: Passthrough representation wrapper.
  - `freeze()` / `unfreeze()` coordinate stabilization for downstream streaming SVMs.
- **Latent Distribution Monitoring & Drift Detection**:
  - `OnlineCovarianceTracker`: Exponentially weighted mean $\mu_t$ and unbiased covariance $\Sigma_t$ with a drift-adaptive forgetting factor $\lambda(D_t)$ (see §2).
  - `LatentCovarianceDriftDetector`: Thresholds on the calibrated surprise score $D_t = -\log_{10} p$ (warning 2.0 = $p < 0.01$, drift 3.0 = $p < 0.001$) with warm-up and a refractory reset after each confirmed drift.
  - `ObsolescenceScorer`: Heuristic obsolescence ranking ($O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$) with support vector protection for controlled decremental unlearning.
- **Covariance-Aware Support Vector Machines**:
  - `CovarianceAwareSVC`: Optimization on covariance-guided Gram matrices $Q = \eta K + (1 - \eta)\, s\, K H K$ (see §1).
  - `DriftAwareCovarianceSVM`: End-to-end streaming architecture ($x \to \text{Frozen AE} \to z \to \text{Covariance Tracker} \to \text{Obsolescence Pruning} \to \text{CovarianceAwareSVC refit}$); every `learn_one` refits `CovarianceAwareSVC` exactly on the current buffer.
  - `COSVM` & `ICOSVM`: Covariance-guided One-Class SVM based on low-variance directions (*Kefi-Fatteh et al., Pattern Recognition 2019*) and its streaming version; ICOSVM updates are warm-started exact re-solves of the current dual (see §4).
  - `IncrementalDecrementalCOSVM` / `IDCOSVM`: Incremental learning and decremental unlearning by warm-started exact re-solve, with multi-policy pruning (`fifo`, `non_sv_first`, `error_first`, `mahalanobis`).
- **Incremental / Decremental & Online Solvers**:
  - `AdiabaticSVM` / `IncrementalDecrementalSVC`: Kuhn-Tucker path following of $\alpha$ and $b$ upon sample insertion and deletion (*Cauwenberghs & Poggio 2001, Laskov et al. 2006*); SMO is only a counted last-resort fallback (`n_smo_fallbacks_`).
  - `LASVM` / `LASVMClassifier`: Online and active learning support vector classifier with the Bordes et al. support vector filtering rule, feasible `forget_one` / `forget_sample_id` (*Bordes et al., JMLR 2005*).
- **Benchmarking Suite & Streaming Evaluation Protocols**:
  - **Prequential (Test-Then-Train) Streaming Evaluator**: `PrequentialEvaluator` tracking rolling/cumulative accuracy, sample step latency, throughput (samples/sec), and support vector dynamics.
  - **Benchmark Dataset Generators**: Synthetic concept drift streams (`make_sea_stream`, `make_rotating_hyperplane_stream`, `make_gaussian_drift_stream`), anisotropic low-variance anomaly datasets (`make_anisotropic_anomaly_dataset`), and non-linear classification benchmarks (`make_benchmark_moons`, `make_benchmark_circles`).
  - **Comparative Benchmark Runners**: `benchmark_streaming_suite`, `benchmark_anomaly_suite`, `benchmark_solvers_suite`.
- **Standard SVM & Kernels**:
  - Binary & Multi-Class Support Vector Classification (`SVC`, `MultiClassSVC` with One-vs-Rest and One-vs-One).
  - `SVC(probability=True)`: Platt sigmoid fitted on out-of-fold decision values (3-fold internal CV, Lin, Lin & Weng 2007). Every other `predict_proba` is an uncalibrated logistic of the margin.
  - Built-in kernels: Linear, Radial Basis Function (RBF / Gaussian), Polynomial, Sigmoid, and Custom User Kernels. A string `gamma` (`'scale'` / `'auto'`, scikit-learn semantics) is resolved once from the training data, so predictions do not depend on how queries are batched.
- **Multiple Optimization Solvers**:
  - **Sequential Minimal Optimization (SMO)**: Platt's algorithm with dual variable caching, two-loop heuristic selection, and KKT violation tracking; warns when `max_iter` stops it with violations left.
  - **Quadratic Programming (QP)**: SLSQP dual solver (`ftol=1e-10`) whose KKT gap is checked against `tol` (warning otherwise).
  - **Pegasos SGD**: Primal stochastic subgradient solver with a regularised bias (augmented feature, the objective of `LinearSVC(loss='hinge', intercept_scaling=s)`) and suffix averaging.
- **Diagnostics & Visualization**:
  - `kkt_report(alpha, K, y, C)` / `oneclass_kkt_report(alpha, G, C, rho)`: equality-constraint residual, maximal-violating-pair KKT gap, dual objective and box violation.
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
$$Q = \eta K + (1 - \eta)\, s\, \Delta, \quad \Delta = K H K, \quad H = \frac{1}{d} Z_c (\Sigma_t + \epsilon I)^{-1} Z_c^T, \quad s = \frac{\|K\|_F}{\|K H K\|_F}$$

$Z_c$ is the latent buffer centred with its own mean, so $H$ is a centred Mahalanobis Gram matrix ($H \succeq 0$, $H\mathbf{1} = 0$) and $Q$ is PSD. The scale $s$ is a constant fitted once on the training buffer (`delta_scale_`) so that $\eta$ weighs two terms of equal Frobenius norm; prediction reuses the same $s$, $H$ and training kernel matrix.

### 2. Online Latent Covariance & Calibrated Drift Statistic
`OnlineCovarianceTracker` keeps exponentially weighted (EW) estimates. Each row $z$ is merged with weight $1 - \lambda$:
$$\delta = z - \mu, \quad \mu \leftarrow \mu + (1 - \lambda)\,\delta, \quad V \leftarrow \lambda \left(V + (1 - \lambda)\,\delta\delta^T\right), \quad n_{\text{eff}} \leftarrow \left(\lambda^2 / n_{\text{eff}} + (1 - \lambda)^2\right)^{-1}$$
and exposes $\Sigma_t = V\, n_{\text{eff}} / (n_{\text{eff}} - 1)$, which is unbiased for i.i.d. rows ($n_{\text{eff}} \to (1+\lambda)/(1-\lambda)$ for constant $\lambda$). `fit` starts from the sample mean, the unbiased sample covariance and $n_{\text{eff}} = n$.

For an incoming batch $B$ ($n_B$ rows, mean $\bar z$), evaluated against the state before the update:
$$T^2 = \frac{(\bar z - \mu)^T (\Sigma + \epsilon I)^{-1} (\bar z - \mu)}{1/n_B + 1/n_{\text{eff}}}$$
- $n_B < d + 1$ (single samples): Hotelling calibration $F = T^2 (n_{\text{eff}} - d) / (d (n_{\text{eff}} - 1)) \sim F(d, n_{\text{eff}} - d)$ (or $T^2 \sim \chi^2_d$ while $n_{\text{eff}} \le d + 1$).
- $n_B \ge d + 1$: add the covariance likelihood-ratio term $\mathrm{LRT} = (n_B - 1)\left[\mathrm{tr}(\Sigma^{-1} S_B) - \log\det(\Sigma^{-1} S_B) - d\right]$ ($S_B$ the unbiased batch covariance) and use $T^2 + \mathrm{LRT} \sim \chi^2_{d + d(d+1)/2}$ (asymptotic).

The drift statistic is the surprise score $D_t = -\log_{10} p \in [0, 300]$, so a threshold $\tau$ means a per-check false-alarm probability $10^{-\tau}$ under stationarity (exact for equally weighted Gaussian rows, approximate for EW weights). Defaults: warning 2.0 ($p < 0.01$), drift 3.0 ($p < 0.001$). After a confirmed drift the detector restarts its warm-up (refractory period). A single-sample test that misses the first shifted sample absorbs it into $\Sigma$, which can mask the following ones.

Adaptive forgetting (computed with the state before the update):
$$\lambda(D_t) = \begin{cases} \lambda_0 & D_t \le \tau_{\text{adapt}} \\ \max\left(\lambda_{\min},\ \lambda_0\, e^{-\gamma_D (D_t - \tau_{\text{adapt}})}\right) & D_t > \tau_{\text{adapt}} \end{cases}, \qquad \tau_{\text{adapt}} = \texttt{adapt\_threshold} = 2.0$$

### 3. Obsolescence-Guided Controlled Forgetting (heuristic)
Stored instances are ranked by
$$O_i = \lambda_A A_i + \lambda_T T_i + \lambda_D D_i$$
where $A_i$ is max-normalized age, $T_i = 1 - e^{-\beta (t - t_i)}$ is a temporal decay distance, and $D_i$ is the max-normalized latent Mahalanobis inconsistency $d_M(z_i, \mu_t, \Sigma_t) = \sqrt{(z_i - \mu_t)^T (\Sigma_t + \epsilon I)^{-1} (z_i - \mu_t)}$; support vectors are protected by `sv_protection_factor`. The weights, normalisation and protection factor are design choices, not derived quantities; only the ordering is used.

### 4. Covariance-Guided One-Class SVM (COSVM, ICOSVM, IDCOSVM)
Based on *Kefi-Fatteh et al. (Pattern Recognition 2019)*. With feature map $\Phi$, centring matrix $H = I - \frac{1}{N}\mathbf{1}\mathbf{1}^T$ and feature-space covariance $\Sigma_\Phi = \frac{1}{N}\Phi H \Phi^T$, the primal
$$\min_{w, \rho, \xi}\ \frac{1}{2} w^T\left(\eta I + (1 - \eta)\Sigma_\Phi\right) w - \rho + C \sum_i \xi_i, \quad w^T\phi(x_i) \ge \rho - \xi_i,\ \xi_i \ge 0,\ C = \frac{1}{\nu N}$$
has the kernelised dual
$$\min_{\alpha} \frac{1}{2} \alpha^T G \alpha \quad \text{s.t. } 0 \le \alpha_i \le \frac{1}{\nu N}, \quad \sum_{i=1}^N \alpha_i = 1, \qquad G = K A^{-1} K, \quad A = \eta K + \frac{1 - \eta}{N} K H K$$
and decision function $f(x) = k(x)^T A^{-1} K \alpha - \rho$, with $\rho$ the mean of $(G\alpha)_i$ over free SVs (midpoint of the admissible interval otherwise). $A$ is factorised by Cholesky with a relative ridge $\epsilon I$ (`reg=1e-10`). $\eta$ must lie in $(0, 1]$: at $\eta = 0$, $A$ is singular for a full-rank kernel and out-of-sample scores blow up like $1/\epsilon$. $\eta = 1$ gives $G = K$, i.e. the standard one-class SVM (scikit-learn's `OneClassSVM` decision equals ours times $\nu N$).

**ICOSVM** and **IDCOSVM** use the same geometry. Adding or removing a sample changes $N$, $C$, $H$ and $A^{-1}$, hence every entry of $G$, so each `learn_one` / `unlearn_one` re-solves the current dual exactly from a feasible warm start (the previous $\alpha$, made feasible); the result equals batch COSVM on the same buffer and kernel to solver precision. ICOSVM first tries a KKT sensitivity predictor ($d\rho = -\beta_0$ from the bordered system over the margin set) and accepts it only if the full KKT check of the new problem passes; `n_incremental_updates_` / `n_qp_solves_` count how often each path ran. Cost per update is $\mathcal{O}(N^3)$ in the worst case.

### 5. Incremental & Decremental Binary SVM (Cauwenberghs & Poggio)
With $Q_{ij} = y_i y_j K_{ij}$, $g_i = \sum_j Q_{ij}\alpha_j + y_i b - 1$ and the sets $\mathcal{S}$ ($0 < \alpha < C$, $g = 0$), $\mathcal{E}$ ($\alpha = C$, $g \le 0$), $\mathcal{O}$ ($\alpha = 0$, $g \ge 0$), a change of one multiplier $\alpha_c$ keeps the KKT conditions if
$$\begin{bmatrix} 0 & \mathbf{y}_{\mathcal{S}}^T \\ \mathbf{y}_{\mathcal{S}} & \mathbf{Q}_{\mathcal{S}\mathcal{S}} \end{bmatrix} \begin{bmatrix} \Delta b \\ \Delta \boldsymbol{\alpha}_{\mathcal{S}} \end{bmatrix} = -\begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix} \Delta \alpha_c, \qquad \boldsymbol{\beta} = -R \begin{bmatrix} y_c \\ \mathbf{Q}_{\mathcal{S}, c} \end{bmatrix}, \quad \gamma_i = Q_{ic} + y_i \beta_0 + \sum_{j \in \mathcal{S}} Q_{ij} \beta_j$$
with $\Delta\alpha_{\mathcal{S}} = \beta_{\mathcal{S}}\Delta\alpha_c$, $\Delta b = \beta_0 \Delta\alpha_c$, $\Delta g_i = \gamma_i \Delta\alpha_c$.
- **Insertion** (`learn_one`): $\alpha_c$ grows from 0 until the first event ($\alpha_c = C$, $g_c = 0$, an $\mathcal{S}$ point hits 0 or $C$, an $\mathcal{E}$/$\mathcal{O}$ point reaches $g = 0$); sets are updated and the path continues.
- **Removal** (`forget_one`): $p$ leaves its set, $\mathcal{S}' = \mathcal{S}\setminus\{p\}$, $\beta = -R'[y_p; Q_{\mathcal{S}',p}]$ and $\alpha_p$ decreases by $\delta$: $\Delta\alpha_s = -\beta_s\delta$, $\Delta b = -\beta_0\delta$, $\Delta g_i = -\gamma_i\delta$. Events: $\alpha_p = 0$, an $\mathcal{S}'$ point hits 0 or $C$, an $\mathcal{E}$ point ($\gamma_i < 0$) or $\mathcal{O}$ point ($\gamma_i > 0$) reaches $g = 0$. $p$ is deleted only once $\alpha_p = 0$, so $\sum_i \alpha_i y_i = 0$ and all other $g_i$ are unchanged.
- **Empty margin set**: a bias-only step $\Delta b = y_c t$ (insertion) or $\Delta b = -y_p t$ (removal) moves $b$ until an eligible $\mathcal{O}$/$\mathcal{E}$ point reaches $g = 0$ and joins $\mathcal{S}$.

$R$ is recomputed at every event ($\mathcal{O}(|\mathcal{S}|^3)$, no rank-one updates) and $(b, \alpha_{\mathcal{S}})$ is re-solved from the margin system after each set change. If the path is exhausted, a feasible SMO warm start repairs the solution and is counted in `n_smo_fallbacks_`.

### 6. LASVM Online & Active Learning (Bordes et al. 2005)
Gradients on the working set $\mathcal{S}$:
$$g_i = y_i - \sum_{j \in \mathcal{S}} \alpha_j y_j K(x_i, x_j)$$
with $I_{\text{up}} = \{i : (y_i = +1, \alpha_i < C) \text{ or } (y_i = -1, \alpha_i > 0)\}$ and $I_{\text{low}} = \{i : (y_i = +1, \alpha_i > 0) \text{ or } (y_i = -1, \alpha_i < C)\}$. A $\tau$-violating pair satisfies $\max_{i \in I_{\text{up}}} g_i - \min_{j \in I_{\text{low}}} g_j > \tau$. Filtering (Bordes rule): an $\alpha_s = 0$ point is removed only if ($y_s = +1$ and $g_s \le \min_{I_{\text{low}}} g$) or ($y_s = -1$ and $g_s \ge \max_{I_{\text{up}}} g$). `fit` ends with full-set passes that re-insert $\tau$-violators from the training set. `forget_one(index)` (position in the working set) first transfers $\alpha_p$ to partners along feasible directions, keeping $\sum_i \alpha_i y_i = 0$ and the cached gradients exact, then deletes $p$ and runs FINISH; `forget_sample_id(sample_id)` does the same by stable id (`SV_ids_`) and is a no-op for samples already filtered out with $\alpha = 0$.

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
│   │   ├── _cosvm_core.py      # Shared COSVM geometry G = K A^-1 K and one-class dual solver
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
│   │   ├── metrics.py          # Accuracy, Confusion Matrix, Hinge Loss
│   │   ├── kkt.py              # kkt_report, oneclass_kkt_report, kkt_gap, compute_intercept
│   │   └── calibration.py      # Platt scaling (fit_platt_sigmoid, platt_proba)
│   ├── benchmarks/             # Benchmarking Suite & Protocols
│   │   ├── __init__.py
│   │   ├── datasets.py         # SEA, Rotating Hyperplane, Gaussian Drift, Anisotropic datasets
│   │   ├── evaluator.py        # PrequentialEvaluator, BatchBenchmarkEvaluator, AnomalyBenchmarkEvaluator
│   │   └── runner.py           # Comparative benchmark suites
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # 2D decision boundary & support vector plots
├── benchmark_suite.py          # Standalone benchmark execution runner
├── generate_reproduction_report.py # Writes reproduction_results.md from computed results
└── tests/                      # Automated unit test suite
    ├── test_math_properties.py # KKT / optimality / calibration checks against references
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
    drift_threshold=3.0,   # D_t = -log10 p: drift when p < 0.001
    warning_threshold=2.0, # warning when p < 0.01
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

# Online stream updates: each step is a warm-started exact re-solve of the current dual
for x_new in stream:
    icosvm.learn_one(x_new)
print(icosvm.n_incremental_updates_, icosvm.n_qp_solves_)  # predictor vs. re-solve counts

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

# Decremental unlearning (exact re-solve on the reduced buffer) by index, sample value, or obsolescence policy
idcosvm.unlearn_one(index=0)
idcosvm.unlearn_sample(specific_vector)
idcosvm.unlearn_obsolete(n_samples=5, strategy="mahalanobis")
```

### 3. Adiabatic Incremental & Decremental SVM
```python
import numpy as np
from dynamicsvm.stream.adiabatic_svm import AdiabaticSVM
from dynamicsvm.utils import kkt_report

model = AdiabaticSVM(C=1.0, kernel="rbf", gamma="scale", max_samples=100)
model.fit(X_train, y_train)

# Online insertion by KKT path following
for x_t, y_t in zip(X_stream, y_stream):
    model.learn_one(x_t, y_t)

# Decremental unlearning by path following (no retraining unless a fallback is counted)
model.forget_one(index=0)
model.unlearn_sample(x_unwanted, y=y_unwanted)

# Check optimality of the current solution
report = kkt_report(model.alpha_, model.K_full_, np.asarray(model.y_buffer_), model.C)
print(report["sum_alpha_y"], report["kkt_gap"], model.n_smo_fallbacks_)
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

# Feasible decremental removal by stable sample id (training index for fit)
lasvm.forget_sample_id(0)
```

### 5. Standard SVM with RBF Kernel
```python
from dynamicsvm.models.svc import SVC

model = SVC(C=1.0, kernel="rbf", gamma="scale", solver="smo", probability=True, random_state=0)
model.fit(X, y)
predictions = model.predict(X)
probabilities = model.predict_proba(X)  # Platt-calibrated because probability=True
print(model.kkt_gap_)                   # KKT gap of the dual solution
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
# Install the dev extra (pytest, scikit-learn for the reference tests)
pip install -e ".[dev]"

# Run the complete test suite (97 tests)
pytest -v

# Regenerate the reproduction report (reproduction_results.md, a few minutes)
python generate_reproduction_report.py

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
