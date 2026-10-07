"""
Unit tests for representation learning encoders in DynamicSVM.
"""

import numpy as np
import pytest

from dynamicsvm.encoders import (
    AutoEncoder,
    IdentityEncoder,
    MLPAutoEncoder,
    PCAEncoder,
)


def test_identity_encoder():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    enc = IdentityEncoder()
    enc.fit(X)
    assert enc.is_frozen_
    assert enc.latent_dim_ == 2
    Z = enc.encode(X)
    np.testing.assert_allclose(Z, X)
    X_rec = enc.decode(Z)
    np.testing.assert_allclose(X_rec, X)


def test_pca_encoder():
    rng = np.random.RandomState(42)
    # 2D data with high variance along first direction
    X = np.dot(rng.randn(100, 2), np.array([[3.0, 0.5], [0.5, 0.2]]))
    
    enc = PCAEncoder(n_components=2)
    enc.fit(X)
    assert enc.latent_dim_ == 2
    
    Z = enc.encode(X)
    assert Z.shape == (100, 2)
    
    X_rec = enc.decode(Z)
    assert X_rec.shape == X.shape
    np.testing.assert_allclose(X_rec, X, atol=1e-5)

    # Test freezing
    enc.freeze()
    assert enc.is_frozen_
    # Refitting frozen encoder does nothing
    X_dummy = np.ones((10, 2))
    enc.fit(X_dummy)
    np.testing.assert_allclose(enc.mean_, np.mean(X, axis=0))


def test_autoencoder_training_and_reconstruction():
    rng = np.random.RandomState(42)
    X = rng.randn(120, 6)
    
    ae = AutoEncoder(
        latent_dim=2,
        hidden_dims=(16,),
        activation="tanh",
        output_activation="linear",
        learning_rate=0.01,
        max_iter=80,
        batch_size=30,
        random_state=42,
    )
    ae.fit(X)
    
    assert ae.latent_dim_ == 2
    assert len(ae.loss_history_) == 80
    # Loss should decrease during training
    assert ae.loss_history_[-1] < ae.loss_history_[0]
    
    Z = ae.encode(X)
    assert Z.shape == (120, 2)
    
    X_rec = ae.decode(Z)
    assert X_rec.shape == (120, 6)
    
    rec_err = ae.reconstruction_error(X)
    assert rec_err.shape == (120,)
    assert np.all(rec_err >= 0.0)
