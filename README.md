# DynamicSVM

DynamicSVM implements covariance-guided one-class SVMs and their streaming variants: `COSVM`, `ICOSVM`, and `IDCOSVM`. It also includes an optional drift-aware wrapper, adaptive encoders, kernel implementations, and focused anomaly-detection benchmarks.

## Models

- `COSVM` fits a covariance-guided one-class SVM to a batch of normal samples.
- `ICOSVM` supports streaming insertion with a bounded sample buffer.
- `IDCOSVM` supports insertion and removal, with configurable buffer-pruning policies.
- `DriftAwareOneClassSVM` wraps any of these estimators and uses a binary drift detector to trigger refitting on a recent raw-input window.

`eta=1.0` gives the standard one-class SVM setting. COSVM-family models require `eta` in `(0, 1]`.

## Batch usage

```python
from dynamicsvm import COSVM

model = COSVM(nu=0.1, eta=0.4, kernel="rbf", gamma="scale")
model.fit(X_train_normal)
labels = model.predict(X_test)  # +1: inlier, -1: anomaly
scores = model.decision_function(X_test)
```

## Streaming and drift adaptation

```python
from dynamicsvm import (
    AutoEncoder,
    COSVM,
    DDMDriftDetector,
    DriftAwareOneClassSVM,
)

model = DriftAwareOneClassSVM(
    COSVM(nu=0.1, eta=0.4, kernel="rbf"),
    window_size=100,
    drift_detector=DDMDriftDetector(),
    encoder=AutoEncoder(latent_dim=4, random_state=42),
).fit(X_initial_normal)

drift_detected = model.learn_one(x_stream)
```

With an encoder, each accepted sample triggers an encoder fit on the active raw-input window, re-encoding of that window, and a batch refit of the COSVM-family estimator. This keeps buffered points in a consistent latent coordinate system, but does not use ICOSVM/IDCOSVM's usual incremental update path. The drift signal is based on anomaly decisions and may confuse isolated outliers with a genuine regime change.

Without an encoder, ICOSVM and IDCOSVM retain their streaming update path; COSVM is refit on its active window.

## Benchmarks

Run the COSVM-family comparison and generate benchmark figures/results with:

```console
python run_benchmarks.py
```

The benchmark includes `eta` selection on validation data and reports performance on held-out test data. The selected `eta` is split- and dataset-dependent; it is not a universal setting.

## Development tests

```console
python -m pytest -q
```
