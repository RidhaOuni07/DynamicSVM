"""
AutoEncoder architectures and dimensionality reduction encoders for DynamicSVM.
Provides representation learning modules to map input data into stable latent spaces.
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from dynamicsvm.encoders.base import BaseEncoder
from dynamicsvm.utils.validation import check_array


class IdentityEncoder(BaseEncoder):
    """
    Identity / Passthrough encoder.
    Returns input features unchanged.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IdentityEncoder":
        X_arr = check_array(X)
        self.latent_dim_ = X_arr.shape[1]
        self.freeze()
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return check_array(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return check_array(Z)


class PCAEncoder(BaseEncoder):
    """
    Principal Component Analysis (PCA) Linear Autoencoder.
    Computes an exact orthogonal linear subspace projection.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of latent dimensions to keep.
    whiten : bool, default=False
        Whether to divide components by singular values to ensure unit variance.
    """

    def __init__(self, n_components: int = 2, whiten: bool = False):
        super().__init__()
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAEncoder":
        if self.is_frozen_:
            return self

        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape
        self.latent_dim_ = min(self.n_components, n_features)

        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        # SVD of centered data: X_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.singular_values_ = S[:self.latent_dim_]
        self.components_ = Vt[:self.latent_dim_]  # shape (latent_dim, n_features)
        self.explained_variance_ = (self.singular_values_ ** 2) / max(1, n_samples - 1)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAEncoder is not fitted.")
        X_arr = check_array(X)
        X_centered = X_arr - self.mean_
        Z = np.dot(X_centered, self.components_.T)
        if self.whiten:
            scale = np.sqrt(self.explained_variance_) + 1e-8
            Z = Z / scale
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAEncoder is not fitted.")
        Z_arr = check_array(Z)
        if self.whiten:
            scale = np.sqrt(self.explained_variance_) + 1e-8
            Z_arr = Z_arr * scale
        X_rec = np.dot(Z_arr, self.components_) + self.mean_
        return X_rec


class AutoEncoder(BaseEncoder):
    """
    Multilayer Perceptron (MLP) Neural AutoEncoder implemented in pure NumPy.
    Maps input data x into a low-dimensional latent space z through nonlinear encoder layers,
    and reconstructs back via decoder layers.
    
    Parameters
    ----------
    latent_dim : int, default=4
        Dimensionality of the bottleneck latent representation.
    hidden_dims : Sequence[int], default=(32,)
        Hidden layer dimensions between input and latent space.
    activation : {'tanh', 'relu', 'sigmoid', 'linear'}, default='tanh'
        Activation function for hidden and latent layers.
    output_activation : {'linear', 'sigmoid', 'tanh'}, default='linear'
        Activation function for the final reconstructed output layer.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    max_iter : int, default=200
        Maximum training epochs.
    batch_size : int, default=32
        Mini-batch size for training.
    weight_decay : float, default=1e-5
        L2 regularization weight penalty.
    random_state : int, optional
        Seed for weight initialization and mini-batch shuffling.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dims: Sequence[int] = (32,),
        activation: str = "tanh",
        output_activation: str = "linear",
        learning_rate: float = 1e-3,
        max_iter: int = 200,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.random_state = random_state

        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []
        self.loss_history_: List[float] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_encoder_layers_: int = 0

    def _act(self, x: np.ndarray, act_name: str) -> np.ndarray:
        if act_name == "tanh":
            return np.tanh(x)
        elif act_name == "relu":
            return np.maximum(0.0, x)
        elif act_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
        elif act_name == "linear":
            return x
        else:
            raise ValueError(f"Unknown activation: '{act_name}'")

    def _act_grad(self, a: np.ndarray, act_name: str) -> np.ndarray:
        """Derivative with respect to pre-activation given activation output `a`."""
        if act_name == "tanh":
            return 1.0 - a ** 2
        elif act_name == "relu":
            return (a > 0.0).astype(float)
        elif act_name == "sigmoid":
            return a * (1.0 - a)
        elif act_name == "linear":
            return np.ones_like(a)
        else:
            raise ValueError(f"Unknown activation: '{act_name}'")

    def _init_layers(self, n_features: int) -> None:
        rng = np.random.RandomState(self.random_state)
        # Encoder structure: input -> hidden -> latent
        encoder_dims = [n_features] + self.hidden_dims + [self.latent_dim]
        # Decoder structure: latent -> reversed hidden -> input
        decoder_dims = [self.latent_dim] + list(reversed(self.hidden_dims)) + [n_features]

        layer_dims = encoder_dims[:-1] + decoder_dims
        self.n_encoder_layers_ = len(encoder_dims) - 1

        self.weights_ = []
        self.biases_ = []

        all_dims = encoder_dims + decoder_dims[1:]
        for i in range(len(all_dims) - 1):
            din = all_dims[i]
            dout = all_dims[i + 1]
            # Xavier/Glorot initialization
            limit = np.sqrt(6.0 / (din + dout))
            W = rng.uniform(-limit, limit, size=(din, dout))
            b = np.zeros(dout)
            self.weights_.append(W)
            self.biases_.append(b)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoEncoder":
        if self.is_frozen_:
            return self

        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape
        self.latent_dim_ = self.latent_dim

        # Standardization for stable gradient descent
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        X_norm = (X_arr - self.mean_) / self.std_

        self._init_layers(n_features)

        # Adam optimizer state
        m_w = [np.zeros_like(w) for w in self.weights_]
        v_w = [np.zeros_like(w) for w in self.weights_]
        m_b = [np.zeros_like(b) for b in self.biases_]
        v_b = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        rng = np.random.RandomState(self.random_state)
        t = 0
        self.loss_history_ = []

        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            epoch_losses = []

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                X_batch = X_norm[batch_idx]
                curr_bsz = X_batch.shape[0]

                # Forward pass
                activations = [X_batch]
                for l_idx, (W, b) in enumerate(zip(self.weights_, self.biases_)):
                    z_val = np.dot(activations[-1], W) + b
                    act_type = (
                        self.output_activation
                        if l_idx == len(self.weights_) - 1
                        else self.activation
                    )
                    a_val = self._act(z_val, act_type)
                    activations.append(a_val)

                X_rec = activations[-1]
                # MSE loss
                diff = X_rec - X_batch
                batch_loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))
                epoch_losses.append(batch_loss)

                # Backward pass
                t += 1
                # Output gradient
                grad_output = diff / curr_bsz
                delta = grad_output * self._act_grad(X_rec, self.output_activation)

                for l_idx in reversed(range(len(self.weights_))):
                    a_prev = activations[l_idx]
                    grad_W = np.dot(a_prev.T, delta) + self.weight_decay * self.weights_[l_idx]
                    grad_b = np.sum(delta, axis=0)

                    if l_idx > 0:
                        delta = np.dot(delta, self.weights_[l_idx].T) * self._act_grad(
                            a_prev, self.activation
                        )

                    # Adam update for W
                    m_w[l_idx] = beta1 * m_w[l_idx] + (1.0 - beta1) * grad_W
                    v_w[l_idx] = beta2 * v_w[l_idx] + (1.0 - beta2) * (grad_W ** 2)
                    m_hat = m_w[l_idx] / (1.0 - beta1 ** t)
                    v_hat = v_w[l_idx] / (1.0 - beta2 ** t)
                    self.weights_[l_idx] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

                    # Adam update for b
                    m_b[l_idx] = beta1 * m_b[l_idx] + (1.0 - beta1) * grad_b
                    v_b[l_idx] = beta2 * v_b[l_idx] + (1.0 - beta2) * (grad_b ** 2)
                    m_b_hat = m_b[l_idx] / (1.0 - beta1 ** t)
                    v_b_hat = v_b[l_idx] / (1.0 - beta2 ** t)
                    self.biases_[l_idx] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            self.loss_history_.append(float(np.mean(epoch_losses)))

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or not self.weights_:
            raise RuntimeError("AutoEncoder is not fitted.")
        X_arr = check_array(X)
        X_norm = (X_arr - self.mean_) / self.std_

        a = X_norm
        for l_idx in range(self.n_encoder_layers_):
            z_val = np.dot(a, self.weights_[l_idx]) + self.biases_[l_idx]
            a = self._act(z_val, self.activation)
        return a

    def decode(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or not self.weights_:
            raise RuntimeError("AutoEncoder is not fitted.")
        Z_arr = check_array(Z)

        a = Z_arr
        for l_idx in range(self.n_encoder_layers_, len(self.weights_)):
            z_val = np.dot(a, self.weights_[l_idx]) + self.biases_[l_idx]
            act_type = (
                self.output_activation
                if l_idx == len(self.weights_) - 1
                else self.activation
            )
            a = self._act(z_val, act_type)

        X_rec = a * self.std_ + self.mean_
        return X_rec


# Convenient alias
MLPAutoEncoder = AutoEncoder
