"""
Fourier Neural Operator: Chunk 4 - Advanced Topics
Complete Implementation: Training, Physics-Informed Learning, and Application

This file implements everything from the Chunk 4 theory guide:
1. Complete Training Pipeline with logging and checkpointing
2. Data Preparation utilities (normalization, augmentation, splitting)
3. Physics-Informed Neural Operator (PINO) losses
4. Advanced FNO architectures (U-FNO, Factorized FNO)
5. Model Interpretation tools (spectral analysis, feature importance)
6. Comprehensive Evaluation Suite
7. Urban Temperature Application Template
8. Debugging and Diagnostic Tools

Author: FNO Tutorial for PhD Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Dict, List, Optional, Callable
import os
import time
import json
from dataclasses import dataclass
from copy import deepcopy

# Create output directory
os.makedirs('figures', exist_ok=True)

print("="*70)
print("CHUNK 4: ADVANCED FNO - TRAINING, PINO, AND APPLICATION")
print("="*70)


#=============================================================================
# SECTION 1: CORE FNO COMPONENTS (from Chunk 3)
#=============================================================================
print("\n" + "="*70)
print("SECTION 1: Core FNO Components (Foundation)")
print("="*70)

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class SpectralConv2d:
    """2D Spectral Convolution Layer."""
    
    def __init__(self, d_in: int, d_out: int, k_max_x: int, k_max_y: int, 
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_in = d_in
        self.d_out = d_out
        self.k_max_x = k_max_x
        self.k_max_y = k_max_y
        
        scale = 1.0 / np.sqrt(d_in * d_out)
        self.R = (np.random.randn(k_max_x, k_max_y, d_in, d_out) + 
                  1j * np.random.randn(k_max_x, k_max_y, d_in, d_out)) * scale
        
        self.n_params = 2 * k_max_x * k_max_y * d_in * d_out
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        Nx, Ny, _ = v.shape
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        w_hat = np.zeros((Nx, Ny//2+1, self.d_out), dtype=complex)
        
        kx_max = min(self.k_max_x, Nx)
        ky_max = min(self.k_max_y, Ny//2+1)
        
        for kx in range(kx_max):
            for ky in range(ky_max):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ self.R[kx, ky, :, :]
        
        for kx in range(1, kx_max):
            for ky in range(ky_max):
                w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(self.R[kx, ky, :, :])
        
        return np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))


class FourierLayer:
    """Complete Fourier Layer with spectral and local paths."""
    
    def __init__(self, d_v: int, k_max: int, dropout: float = 0.0, 
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_v = d_v
        self.k_max = k_max
        self.dropout = dropout
        
        self.spectral_conv = SpectralConv2d(d_v, d_v, k_max, k_max, seed)
        
        scale = 1.0 / np.sqrt(d_v * d_v)
        self.W = np.random.randn(d_v, d_v) * scale
        self.b = np.zeros(d_v)
        
        self.n_params = self.spectral_conv.n_params + d_v * d_v + d_v
    
    def forward(self, v: np.ndarray, training: bool = True) -> np.ndarray:
        spectral_out = self.spectral_conv.forward(v)
        local_out = v @ self.W
        
        # Apply dropout to local path only
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1-self.dropout, local_out.shape)
            local_out = local_out * mask / (1 - self.dropout)
        
        return gelu(spectral_out + local_out + self.b)


class FNO2d:
    """Complete 2D Fourier Neural Operator."""
    
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int,
                 n_layers: int = 4, d_mid: Optional[int] = None,
                 dropout: float = 0.0, seed: Optional[int] = None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.k_max = k_max
        self.n_layers = n_layers
        self.d_mid = d_mid if d_mid is not None else d_v * 2
        self.dropout = dropout
        
        # Lifting
        scale = np.sqrt(2.0 / (d_a + d_v))
        self.P = np.random.randn(d_a, d_v) * scale
        self.b_P = np.zeros(d_v)
        
        # Fourier layers
        self.fourier_layers = []
        for i in range(n_layers):
            layer_seed = seed + i + 1 if seed is not None else None
            self.fourier_layers.append(
                FourierLayer(d_v, k_max, dropout, layer_seed)
            )
        
        # Projection
        scale1 = np.sqrt(2.0 / (d_v + self.d_mid))
        self.Q1 = np.random.randn(d_v, self.d_mid) * scale1
        self.b_Q1 = np.zeros(self.d_mid)
        
        scale2 = np.sqrt(2.0 / (self.d_mid + d_u))
        self.Q2 = np.random.randn(self.d_mid, d_u) * scale2
        self.b_Q2 = np.zeros(d_u)
        
        # Count parameters
        self.n_params = (d_a * d_v + d_v +  # Lifting
                        sum(fl.n_params for fl in self.fourier_layers) +
                        d_v * self.d_mid + self.d_mid + 
                        self.d_mid * d_u + d_u)  # Projection
    
    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        # Lifting
        v = a @ self.P + self.b_P
        
        # Fourier layers
        for fl in self.fourier_layers:
            v = fl.forward(v, training)
        
        # Projection
        h = gelu(v @ self.Q1 + self.b_Q1)
        u = h @ self.Q2 + self.b_Q2
        
        return u
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all weights as dictionary."""
        weights = {
            'P': self.P.copy(),
            'b_P': self.b_P.copy(),
            'Q1': self.Q1.copy(),
            'b_Q1': self.b_Q1.copy(),
            'Q2': self.Q2.copy(),
            'b_Q2': self.b_Q2.copy(),
        }
        for i, fl in enumerate(self.fourier_layers):
            weights[f'FL{i}_R'] = fl.spectral_conv.R.copy()
            weights[f'FL{i}_W'] = fl.W.copy()
            weights[f'FL{i}_b'] = fl.b.copy()
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set all weights from dictionary."""
        self.P = weights['P'].copy()
        self.b_P = weights['b_P'].copy()
        self.Q1 = weights['Q1'].copy()
        self.b_Q1 = weights['b_Q1'].copy()
        self.Q2 = weights['Q2'].copy()
        self.b_Q2 = weights['b_Q2'].copy()
        for i, fl in enumerate(self.fourier_layers):
            fl.spectral_conv.R = weights[f'FL{i}_R'].copy()
            fl.W = weights[f'FL{i}_W'].copy()
            fl.b = weights[f'FL{i}_b'].copy()


print("✓ Core FNO components loaded")
print(f"  FNO2d(d_a=42, d_v=32, d_u=1, k_max=12) → {FNO2d(42, 32, 1, 12).n_params:,} params")


#=============================================================================
# SECTION 2: DATA PREPARATION UTILITIES
#=============================================================================
print("\n" + "="*70)
print("SECTION 2: Data Preparation Utilities")
print("="*70)

class DataNormalizer:
    """
    Comprehensive data normalizer for FNO training.
    Handles per-channel normalization for inputs and global normalization for outputs.
    """
    
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, inputs: np.ndarray, outputs: np.ndarray, 
            feature_names: Optional[List[str]] = None):
        """
        Fit normalizer on training data.
        
        Parameters:
        -----------
        inputs : np.ndarray, shape (N, Nx, Ny, d_a)
        outputs : np.ndarray, shape (N, Nx, Ny, d_u)
        feature_names : optional list of feature names
        """
        # Per-channel statistics for inputs
        self.input_mean = np.mean(inputs, axis=(0, 1, 2), keepdims=True)
        self.input_std = np.std(inputs, axis=(0, 1, 2), keepdims=True)
        self.input_std = np.maximum(self.input_std, 1e-8)
        
        # Global statistics for outputs
        self.output_mean = np.mean(outputs)
        self.output_std = np.std(outputs)
        self.output_std = max(self.output_std, 1e-8)
        
        self.feature_names = feature_names
        self.is_fitted = True
        
        return self
    
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted")
        return (x - self.input_mean) / self.input_std
    
    def normalize_output(self, y: np.ndarray) -> np.ndarray:
        """Normalize output data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted")
        return (y - self.output_mean) / self.output_std
    
    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized output to original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted")
        return y_norm * self.output_std + self.output_mean
    
    def get_stats(self) -> Dict:
        """Get normalization statistics."""
        return {
            'input_mean_range': (float(self.input_mean.min()), 
                                float(self.input_mean.max())),
            'input_std_range': (float(self.input_std.min()), 
                               float(self.input_std.max())),
            'output_mean': float(self.output_mean),
            'output_std': float(self.output_std),
        }
    
    def save(self, filepath: str):
        """Save normalizer to file."""
        np.savez(filepath,
                input_mean=self.input_mean,
                input_std=self.input_std,
                output_mean=self.output_mean,
                output_std=self.output_std)
    
    def load(self, filepath: str):
        """Load normalizer from file."""
        data = np.load(filepath)
        self.input_mean = data['input_mean']
        self.input_std = data['input_std']
        self.output_mean = float(data['output_mean'])
        self.output_std = float(data['output_std'])
        self.is_fitted = True


class DataAugmentation:
    """
    Data augmentation for spatial FNO data.
    Only uses physically valid transformations.
    """
    
    def __init__(self, flip_h: bool = True, flip_v: bool = True,
                 rotate_90: bool = True):
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rotate_90 = rotate_90
    
    def augment(self, x: np.ndarray, y: np.ndarray, 
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentation to input-output pair.
        
        Parameters:
        -----------
        x : np.ndarray, shape (Nx, Ny, d_a)
        y : np.ndarray, shape (Nx, Ny, d_u)
        """
        if seed is not None:
            np.random.seed(seed)
        
        x_aug = x.copy()
        y_aug = y.copy()
        
        # Horizontal flip
        if self.flip_h and np.random.rand() > 0.5:
            x_aug = np.flip(x_aug, axis=1)
            y_aug = np.flip(y_aug, axis=1)
        
        # Vertical flip
        if self.flip_v and np.random.rand() > 0.5:
            x_aug = np.flip(x_aug, axis=0)
            y_aug = np.flip(y_aug, axis=0)
        
        # 90° rotation
        if self.rotate_90:
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                x_aug = np.rot90(x_aug, k, axes=(0, 1))
                y_aug = np.rot90(y_aug, k, axes=(0, 1))
        
        return x_aug.copy(), y_aug.copy()  # Ensure contiguous


def train_val_test_split(n_samples: int, train_frac: float = 0.7,
                         val_frac: float = 0.15, seed: int = 42
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test sets.
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    return train_idx, val_idx, test_idx


# Demonstrate data preparation
print("\n--- Data Preparation Demo ---")

# Create synthetic data
N, Nx, Ny, d_a, d_u = 100, 64, 64, 42, 1
np.random.seed(42)

# Features with different scales (like your data)
synthetic_inputs = np.zeros((N, Nx, Ny, d_a))
synthetic_inputs[:, :, :, 0] = np.random.randn(N, Nx, Ny) * 0.3  # NDVI [-1, 1]
synthetic_inputs[:, :, :, 1] = np.random.randn(N, Nx, Ny) * 0.2 + 0.5  # NDBI
synthetic_inputs[:, :, :, 2] = np.random.rand(N, Nx, Ny) * 100  # Building height [0, 100]
synthetic_inputs[:, :, :, 3] = np.random.rand(N, Nx, Ny) * 50 + 280  # ERA5 temp [280, 330]
for i in range(4, d_a):
    synthetic_inputs[:, :, :, i] = np.random.randn(N, Nx, Ny)

synthetic_outputs = np.random.randn(N, Nx, Ny, d_u) * 10 + 300  # Temperature ~290-310

# Split data
train_idx, val_idx, test_idx = train_val_test_split(N)
print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

# Fit normalizer on training data only
normalizer = DataNormalizer()
normalizer.fit(synthetic_inputs[train_idx], synthetic_outputs[train_idx])

stats = normalizer.get_stats()
print(f"Input mean range: [{stats['input_mean_range'][0]:.2f}, {stats['input_mean_range'][1]:.2f}]")
print(f"Input std range: [{stats['input_std_range'][0]:.2f}, {stats['input_std_range'][1]:.2f}]")
print(f"Output mean: {stats['output_mean']:.2f}, std: {stats['output_std']:.2f}")

# Test augmentation
aug = DataAugmentation()
x_orig = synthetic_inputs[0]
y_orig = synthetic_outputs[0]
x_aug, y_aug = aug.augment(x_orig, y_orig, seed=42)
print(f"\nAugmentation: shape preserved = {x_orig.shape == x_aug.shape}")


#=============================================================================
# SECTION 3: LOSS FUNCTIONS
#=============================================================================
print("\n" + "="*70)
print("SECTION 3: Loss Functions (Standard and Physics-Informed)")
print("="*70)

def mse_loss(pred: np.ndarray, target: np.ndarray, 
             mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Squared Error with optional mask for missing data.
    """
    diff = (pred - target) ** 2
    if mask is not None:
        return np.sum(diff * mask) / np.sum(mask)
    return np.mean(diff)


def mae_loss(pred: np.ndarray, target: np.ndarray,
             mask: Optional[np.ndarray] = None) -> float:
    """Mean Absolute Error with optional mask."""
    diff = np.abs(pred - target)
    if mask is not None:
        return np.sum(diff * mask) / np.sum(mask)
    return np.mean(diff)


def relative_l2_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L2 loss (FNO paper default)."""
    return np.linalg.norm(pred - target) / np.linalg.norm(target)


# Physics-Informed Loss Functions

def compute_laplacian_spectral(field: np.ndarray) -> np.ndarray:
    """
    Compute Laplacian using spectral method.
    ∇²f = F⁻¹[-(k_x² + k_y²) · F[f]]
    
    Parameters:
    -----------
    field : np.ndarray, shape (Nx, Ny) or (Nx, Ny, 1)
    """
    if field.ndim == 3:
        field = field[:, :, 0]
    
    Nx, Ny = field.shape
    
    # Fourier transform
    f_hat = np.fft.fft2(field)
    
    # Wavenumbers
    kx = np.fft.fftfreq(Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Laplacian in Fourier space
    laplacian_hat = -(KX**2 + KY**2) * f_hat
    
    # Inverse transform
    laplacian = np.real(np.fft.ifft2(laplacian_hat))
    
    return laplacian


def smoothness_loss(pred: np.ndarray) -> float:
    """
    Smoothness loss: penalizes large Laplacian (high curvature).
    L_smooth = ||∇²T||²
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    laplacian = compute_laplacian_spectral(pred)
    return np.mean(laplacian ** 2)


def gradient_magnitude_loss(pred: np.ndarray) -> float:
    """
    Gradient magnitude loss: penalizes large gradients.
    L_grad = ||∇T||²
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    
    # Compute gradients using finite differences
    grad_x = np.diff(pred, axis=0, append=pred[-1:, :])
    grad_y = np.diff(pred, axis=1, append=pred[:, -1:])
    
    return np.mean(grad_x**2 + grad_y**2)


def vegetation_cooling_loss(pred: np.ndarray, ndvi: np.ndarray) -> float:
    """
    Physics constraint: vegetation should cool (negative correlation with temp).
    L_veg = ReLU(corr(T, NDVI))²
    
    Penalizes positive correlation (warm vegetation).
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if ndvi.ndim == 3:
        ndvi = ndvi[:, :, 0]
    
    # Compute correlation
    pred_flat = pred.flatten() - np.mean(pred)
    ndvi_flat = ndvi.flatten() - np.mean(ndvi)
    
    corr = np.sum(pred_flat * ndvi_flat) / (
        np.sqrt(np.sum(pred_flat**2)) * np.sqrt(np.sum(ndvi_flat**2)) + 1e-8
    )
    
    # Penalize positive correlation
    return max(0, corr) ** 2


def urban_heating_loss(pred: np.ndarray, ndbi: np.ndarray) -> float:
    """
    Physics constraint: urban surfaces should warm (positive correlation with temp).
    L_urban = ReLU(-corr(T, NDBI))²
    
    Penalizes negative correlation (cool urban).
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if ndbi.ndim == 3:
        ndbi = ndbi[:, :, 0]
    
    pred_flat = pred.flatten() - np.mean(pred)
    ndbi_flat = ndbi.flatten() - np.mean(ndbi)
    
    corr = np.sum(pred_flat * ndbi_flat) / (
        np.sqrt(np.sum(pred_flat**2)) * np.sqrt(np.sum(ndbi_flat**2)) + 1e-8
    )
    
    # Penalize negative correlation
    return max(0, -corr) ** 2


def boundary_loss(pred: np.ndarray, boundary_values: np.ndarray,
                  boundary_mask: np.ndarray) -> float:
    """
    Boundary condition loss: predictions should match known values at boundaries.
    L_BC = ||pred - BC||² at boundary points
    """
    diff = (pred - boundary_values) ** 2
    return np.sum(diff * boundary_mask) / (np.sum(boundary_mask) + 1e-8)


def conservation_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Conservation loss: total heat content should be preserved.
    L_conserve = (∫pred - ∫target)²
    """
    return (np.sum(pred) - np.sum(target)) ** 2 / pred.size


class PhysicsInformedLoss:
    """
    Combined physics-informed loss function for PINO training.
    """
    
    def __init__(self, 
                 lambda_smooth: float = 0.01,
                 lambda_veg: float = 0.1,
                 lambda_urban: float = 0.1,
                 lambda_conserve: float = 0.01):
        
        self.lambda_smooth = lambda_smooth
        self.lambda_veg = lambda_veg
        self.lambda_urban = lambda_urban
        self.lambda_conserve = lambda_conserve
    
    def __call__(self, pred: np.ndarray, target: np.ndarray,
                 ndvi: Optional[np.ndarray] = None,
                 ndbi: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all loss components.
        
        Returns dictionary with individual losses and total.
        """
        losses = {}
        
        # Data loss
        losses['data'] = mse_loss(pred, target)
        
        # Smoothness
        losses['smooth'] = smoothness_loss(pred)
        
        # Vegetation cooling
        if ndvi is not None:
            losses['veg'] = vegetation_cooling_loss(pred, ndvi)
        else:
            losses['veg'] = 0.0
        
        # Urban heating
        if ndbi is not None:
            losses['urban'] = urban_heating_loss(pred, ndbi)
        else:
            losses['urban'] = 0.0
        
        # Conservation
        losses['conserve'] = conservation_loss(pred, target)
        
        # Total
        losses['total'] = (losses['data'] + 
                          self.lambda_smooth * losses['smooth'] +
                          self.lambda_veg * losses['veg'] +
                          self.lambda_urban * losses['urban'] +
                          self.lambda_conserve * losses['conserve'])
        
        return losses
    
    def update_lambdas(self, epoch: int, schedule: str = 'gradual'):
        """
        Update physics loss weights based on training progress.
        """
        if schedule == 'gradual':
            # Gradually increase physics weights
            if epoch < 50:
                factor = 0.1
            elif epoch < 100:
                factor = 0.5
            else:
                factor = 1.0
            
            self.lambda_smooth *= factor
            self.lambda_veg *= factor
            self.lambda_urban *= factor


# Demonstrate loss functions
print("\n--- Loss Functions Demo ---")

# Create synthetic prediction and target
Nx, Ny = 64, 64
x = np.linspace(0, 2*np.pi, Nx)
y = np.linspace(0, 2*np.pi, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

target = np.sin(X) + np.cos(Y) + 0.5 * np.sin(2*X) * np.cos(2*Y)
pred_good = target + np.random.randn(Nx, Ny) * 0.1
pred_bad = target + np.random.randn(Nx, Ny) * 0.5

print("Standard losses:")
print(f"  MSE (good): {mse_loss(pred_good, target):.6f}")
print(f"  MSE (bad):  {mse_loss(pred_bad, target):.6f}")
print(f"  Relative L2 (good): {relative_l2_loss(pred_good, target):.4f}")
print(f"  Relative L2 (bad):  {relative_l2_loss(pred_bad, target):.4f}")

print("\nPhysics losses:")
print(f"  Smoothness (smooth field): {smoothness_loss(target):.6f}")
noisy_field = target + np.random.randn(Nx, Ny) * 2
print(f"  Smoothness (noisy field):  {smoothness_loss(noisy_field):.6f}")

# Create synthetic NDVI (anticorrelated with temperature)
ndvi = -0.3 * target + np.random.randn(Nx, Ny) * 0.1  # Vegetation cools
print(f"  Vegetation loss (correct): {vegetation_cooling_loss(target, ndvi):.6f}")
ndvi_wrong = 0.3 * target + np.random.randn(Nx, Ny) * 0.1  # Wrong sign
print(f"  Vegetation loss (wrong):   {vegetation_cooling_loss(target, ndvi_wrong):.6f}")

# Physics-informed loss
pino_loss = PhysicsInformedLoss(lambda_smooth=0.01, lambda_veg=0.1)
losses = pino_loss(pred_good, target, ndvi=ndvi)
print("\nPhysics-Informed Loss breakdown:")
for name, value in losses.items():
    print(f"  {name:10s}: {value:.6f}")


#=============================================================================
# SECTION 4: EVALUATION METRICS
#=============================================================================
print("\n" + "="*70)
print("SECTION 4: Comprehensive Evaluation Metrics")
print("="*70)

def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def compute_spatial_anomaly_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Spatial Anomaly R² - THE KEY METRIC for urban temperature!
    
    Tests if model captures WHERE is hotter/cooler than average.
    """
    pred_anomaly = pred - np.mean(pred)
    target_anomaly = target - np.mean(target)
    
    ss_res = np.sum((target_anomaly - pred_anomaly) ** 2)
    ss_tot = np.sum(target_anomaly ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    return 1 - ss_res / ss_tot


def compute_spectral_error(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute error in spectral domain at different frequency bands.
    """
    if pred.ndim == 3:
        pred = pred[:, :, 0]
        target = target[:, :, 0]
    
    pred_hat = np.fft.fft2(pred)
    target_hat = np.fft.fft2(target)
    
    Nx, Ny = pred.shape
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    
    errors = {}
    
    # Low frequency (k < 5)
    mask_low = K < 5
    errors['low_freq'] = np.sum(np.abs(pred_hat - target_hat)[mask_low]) / np.sum(mask_low)
    
    # Medium frequency (5 <= k < 15)
    mask_med = (K >= 5) & (K < 15)
    errors['med_freq'] = np.sum(np.abs(pred_hat - target_hat)[mask_med]) / np.sum(mask_med)
    
    # High frequency (k >= 15)
    mask_high = K >= 15
    errors['high_freq'] = np.sum(np.abs(pred_hat - target_hat)[mask_high]) / np.sum(mask_high)
    
    return errors


def evaluate_model(model: FNO2d, inputs: np.ndarray, targets: np.ndarray,
                   normalizer: DataNormalizer) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    model : FNO2d model
    inputs : np.ndarray, shape (N, Nx, Ny, d_a)
    targets : np.ndarray, shape (N, Nx, Ny, d_u)
    normalizer : fitted DataNormalizer
    """
    N = len(inputs)
    
    all_preds = []
    all_targets = []
    
    for i in range(N):
        # Normalize input
        x_norm = normalizer.normalize_input(inputs[i:i+1])[0]
        
        # Predict
        pred_norm = model.forward(x_norm, training=False)
        
        # Denormalize
        pred = normalizer.denormalize_output(pred_norm)
        
        all_preds.append(pred)
        all_targets.append(targets[i])
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    metrics = {
        'rmse': compute_rmse(all_preds, all_targets),
        'mae': compute_mae(all_preds, all_targets),
        'r2': compute_r2(all_preds.flatten(), all_targets.flatten()),
    }
    
    # Spatial anomaly R² per sample, then average
    spatial_r2s = []
    for p, t in zip(all_preds, all_targets):
        spatial_r2s.append(compute_spatial_anomaly_r2(p[:,:,0], t[:,:,0]))
    metrics['spatial_anomaly_r2'] = np.mean(spatial_r2s)
    metrics['spatial_anomaly_r2_std'] = np.std(spatial_r2s)
    
    # Spectral errors (averaged over samples)
    spec_errors = {'low_freq': [], 'med_freq': [], 'high_freq': []}
    for p, t in zip(all_preds, all_targets):
        se = compute_spectral_error(p, t)
        for k in spec_errors:
            spec_errors[k].append(se[k])
    
    for k in spec_errors:
        metrics[f'spectral_{k}'] = np.mean(spec_errors[k])
    
    return metrics


# Demonstrate metrics
print("\n--- Evaluation Metrics Demo ---")

# Create test data
target = np.sin(X) + np.cos(Y)
pred_good = target + np.random.randn(Nx, Ny) * 0.1
pred_mean_only = np.ones_like(target) * np.mean(target)

print("Good prediction (captures pattern):")
print(f"  RMSE: {compute_rmse(pred_good, target):.4f}")
print(f"  R²:   {compute_r2(pred_good.flatten(), target.flatten()):.4f}")
print(f"  Spatial Anomaly R²: {compute_spatial_anomaly_r2(pred_good, target):.4f}")

print("\nMean-only prediction (no pattern):")
print(f"  RMSE: {compute_rmse(pred_mean_only, target):.4f}")
print(f"  R²:   {compute_r2(pred_mean_only.flatten(), target.flatten()):.4f}")
print(f"  Spatial Anomaly R²: {compute_spatial_anomaly_r2(pred_mean_only, target):.4f}")

print("\n→ Spatial Anomaly R² correctly identifies that mean-only has no skill!")

# Spectral error analysis
spec_err = compute_spectral_error(pred_good, target)
print(f"\nSpectral errors: low={spec_err['low_freq']:.4f}, "
      f"med={spec_err['med_freq']:.4f}, high={spec_err['high_freq']:.4f}")


#=============================================================================
# SECTION 5: TRAINING PIPELINE
#=============================================================================
print("\n" + "="*70)
print("SECTION 5: Complete Training Pipeline")
print("="*70)

@dataclass
class TrainingConfig:
    """Configuration for FNO training."""
    # Model
    d_a: int = 42
    d_v: int = 32
    d_u: int = 1
    k_max: int = 12
    n_layers: int = 4
    dropout: float = 0.1
    
    # Training
    n_epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    
    # Schedule
    lr_schedule: str = 'cosine'  # 'cosine', 'step', 'constant'
    lr_min: float = 1e-6
    
    # Early stopping
    patience: int = 50
    min_delta: float = 1e-6
    
    # Physics
    use_physics: bool = False
    lambda_smooth: float = 0.01
    lambda_veg: float = 0.1
    
    # Data
    augment: bool = True
    
    # Logging
    log_interval: int = 10
    save_best: bool = True


class TrainingLogger:
    """Logger for training progress."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def log(self, epoch: int, train_loss: float, val_loss: float,
            val_metrics: Dict, lr: float):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_metrics.append(val_metrics)
        self.learning_rates.append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss curves
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='Train')
        axes[0].plot(self.epochs, self.val_losses, 'r-', label='Val')
        axes[0].axvline(x=self.best_epoch, color='g', linestyle='--', 
                       label=f'Best (epoch {self.best_epoch})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Curves')
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Spatial Anomaly R²
        spatial_r2 = [m.get('spatial_anomaly_r2', 0) for m in self.val_metrics]
        axes[1].plot(self.epochs, spatial_r2, 'purple')
        axes[1].axhline(y=0.7, color='g', linestyle='--', label='Target 0.7')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Spatial Anomaly R²')
        axes[1].set_title('Key Metric Progress')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Learning rate
        axes[2].plot(self.epochs, self.learning_rates, 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('LR Schedule')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def get_lr(epoch: int, config: TrainingConfig) -> float:
    """Get learning rate for current epoch."""
    if config.lr_schedule == 'constant':
        return config.learning_rate
    elif config.lr_schedule == 'cosine':
        return config.lr_min + 0.5 * (config.learning_rate - config.lr_min) * (
            1 + np.cos(np.pi * epoch / config.n_epochs))
    elif config.lr_schedule == 'step':
        # Reduce by 0.5 every 100 epochs
        factor = 0.5 ** (epoch // 100)
        return config.learning_rate * factor
    return config.learning_rate


def simulate_training_step(model: FNO2d, x: np.ndarray, y: np.ndarray,
                          lr: float, config: TrainingConfig) -> float:
    """
    Simulate a training step.
    
    NOTE: Real training requires autograd (PyTorch/JAX).
    This simulates the forward pass and loss computation.
    """
    # Forward pass
    pred = model.forward(x, training=True)
    
    # Compute loss
    loss = mse_loss(pred, y)
    
    # In real training: loss.backward(), optimizer.step()
    # Here we just return the loss for demonstration
    
    return loss


def train_epoch(model: FNO2d, train_inputs: np.ndarray, train_targets: np.ndarray,
                normalizer: DataNormalizer, config: TrainingConfig,
                augmentation: Optional[DataAugmentation] = None,
                lr: float = 1e-3) -> float:
    """
    Simulate one training epoch.
    """
    N = len(train_inputs)
    indices = np.random.permutation(N)
    
    total_loss = 0
    n_batches = 0
    
    for start in range(0, N, config.batch_size):
        end = min(start + config.batch_size, N)
        batch_idx = indices[start:end]
        
        batch_loss = 0
        for idx in batch_idx:
            x = train_inputs[idx]
            y = train_targets[idx]
            
            # Augmentation
            if augmentation and config.augment:
                x, y = augmentation.augment(x, y)
            
            # Normalize
            x_norm = normalizer.normalize_input(x[np.newaxis])[0]
            y_norm = normalizer.normalize_output(y)
            
            # Forward and loss
            loss = simulate_training_step(model, x_norm, y_norm, lr, config)
            batch_loss += loss
        
        total_loss += batch_loss / len(batch_idx)
        n_batches += 1
    
    return total_loss / n_batches


def validate(model: FNO2d, val_inputs: np.ndarray, val_targets: np.ndarray,
             normalizer: DataNormalizer) -> Tuple[float, Dict]:
    """
    Validate model on validation set.
    """
    metrics = evaluate_model(model, val_inputs, val_targets, normalizer)
    
    # Compute validation loss
    val_loss = 0
    for i in range(len(val_inputs)):
        x_norm = normalizer.normalize_input(val_inputs[i:i+1])[0]
        y_norm = normalizer.normalize_output(val_targets[i])
        pred_norm = model.forward(x_norm, training=False)
        val_loss += mse_loss(pred_norm, y_norm)
    val_loss /= len(val_inputs)
    
    return val_loss, metrics


# Demonstrate training pipeline
print("\n--- Training Pipeline Demo ---")

config = TrainingConfig(
    d_a=5,  # Simplified for demo
    d_v=16,
    d_u=1,
    k_max=8,
    n_layers=2,
    n_epochs=50,
    batch_size=8,
)

# Create simple synthetic data
N_demo = 50
Nx_demo, Ny_demo = 32, 32
np.random.seed(42)

demo_inputs = np.random.randn(N_demo, Nx_demo, Ny_demo, config.d_a)
demo_targets = np.random.randn(N_demo, Nx_demo, Ny_demo, config.d_u)

train_idx_demo, val_idx_demo, _ = train_val_test_split(N_demo, train_frac=0.7, val_frac=0.15)

# Normalizer
normalizer_demo = DataNormalizer()
normalizer_demo.fit(demo_inputs[train_idx_demo], demo_targets[train_idx_demo])

# Model
model_demo = FNO2d(config.d_a, config.d_v, config.d_u, config.k_max, 
                   config.n_layers, seed=42)
print(f"Model parameters: {model_demo.n_params:,}")

# Logger
logger_demo = TrainingLogger()

# Augmentation
aug_demo = DataAugmentation()

# Simulate training
print("\nSimulating training...")
best_weights = model_demo.get_weights()
patience_counter = 0

for epoch in range(config.n_epochs):
    lr = get_lr(epoch, config)
    
    # Train
    train_loss = train_epoch(
        model_demo, 
        demo_inputs[train_idx_demo], 
        demo_targets[train_idx_demo],
        normalizer_demo, config, aug_demo, lr
    )
    
    # Validate
    val_loss, val_metrics = validate(
        model_demo,
        demo_inputs[val_idx_demo],
        demo_targets[val_idx_demo],
        normalizer_demo
    )
    
    # Log
    logger_demo.log(epoch, train_loss, val_loss, val_metrics, lr)
    
    # Early stopping check
    if val_loss < logger_demo.best_val_loss - config.min_delta:
        best_weights = model_demo.get_weights()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= config.patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"spatial_r2={val_metrics['spatial_anomaly_r2']:.4f}")

# Restore best weights
model_demo.set_weights(best_weights)
print(f"\nBest epoch: {logger_demo.best_epoch}, best val_loss: {logger_demo.best_val_loss:.4f}")

# Plot training history
logger_demo.plot_history('figures/01_training_history.png')
print("✓ Saved: figures/01_training_history.png")


#=============================================================================
# SECTION 6: ADVANCED FNO ARCHITECTURES
#=============================================================================
print("\n" + "="*70)
print("SECTION 6: Advanced FNO Architectures")
print("="*70)

class FactorizedSpectralConv2d:
    """
    Factorized Spectral Convolution for parameter efficiency.
    
    Instead of R(kx, ky) as full tensor, use:
    R(kx, ky) ≈ Rx(kx) ⊗ Ry(ky)
    
    Reduces parameters from O(k²d²) to O(kd²)
    """
    
    def __init__(self, d_in: int, d_out: int, k_max: int, 
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_in = d_in
        self.d_out = d_out
        self.k_max = k_max
        
        scale = 1.0 / np.sqrt(d_in * d_out)
        
        # Factorized weights
        self.Rx = (np.random.randn(k_max, d_in, d_out) + 
                   1j * np.random.randn(k_max, d_in, d_out)) * scale
        self.Ry = (np.random.randn(k_max, d_in, d_out) + 
                   1j * np.random.randn(k_max, d_in, d_out)) * scale
        
        # Mixing matrices
        self.U = np.random.randn(d_out, d_out) * scale
        
        self.n_params = 2 * 2 * k_max * d_in * d_out + d_out * d_out
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        Nx, Ny, _ = v.shape
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        w_hat = np.zeros((Nx, Ny//2+1, self.d_out), dtype=complex)
        
        kx_max = min(self.k_max, Nx)
        ky_max = min(self.k_max, Ny//2+1)
        
        for kx in range(kx_max):
            for ky in range(ky_max):
                # Factorized multiplication
                temp = v_hat[kx, ky, :] @ self.Rx[kx, :, :]
                temp = temp @ self.Ry[ky, :, :].T
                w_hat[kx, ky, :] = temp @ self.U
        
        return np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))


class UFNOBlock:
    """
    U-FNO Block: Combines FNO with U-Net style skip connections.
    
    Architecture:
        Input → Downsample → FourierLayer → Upsample → Add(Input) → Output
    """
    
    def __init__(self, d_v: int, k_max: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_v = d_v
        self.k_max = k_max
        
        self.fourier_layer = FourierLayer(d_v, k_max, seed=seed)
        
        # Downsampling and upsampling are done via pooling/interpolation
        self.n_params = self.fourier_layer.n_params
    
    def downsample(self, x: np.ndarray, factor: int = 2) -> np.ndarray:
        """Simple average pooling downsampling."""
        Nx, Ny, C = x.shape
        new_Nx, new_Ny = Nx // factor, Ny // factor
        x_down = x.reshape(new_Nx, factor, new_Ny, factor, C)
        return x_down.mean(axis=(1, 3))
    
    def upsample(self, x: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Simple nearest neighbor upsampling."""
        Nx, Ny, C = x.shape
        target_Nx, target_Ny = target_shape
        
        # Repeat each element
        factor_x = target_Nx // Nx
        factor_y = target_Ny // Ny
        
        x_up = np.repeat(np.repeat(x, factor_x, axis=0), factor_y, axis=1)
        return x_up[:target_Nx, :target_Ny, :]
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        original_shape = v.shape[:2]
        
        # Downsample
        v_down = self.downsample(v)
        
        # Fourier layer at lower resolution
        v_processed = self.fourier_layer.forward(v_down)
        
        # Upsample
        v_up = self.upsample(v_processed, original_shape)
        
        # Skip connection
        return v + v_up


class MultiScaleFNO:
    """
    Multi-Scale FNO: Parallel branches with different k_max.
    
    Captures patterns at multiple spatial scales:
    - Low k_max: City-wide patterns
    - Medium k_max: Neighborhood patterns
    - High k_max: Block-scale patterns
    """
    
    def __init__(self, d_a: int, d_v: int, d_u: int,
                 k_max_values: List[int] = [4, 12, 24],
                 n_layers: int = 2, seed: Optional[int] = None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.k_max_values = k_max_values
        self.n_branches = len(k_max_values)
        
        # Shared lifting
        scale = np.sqrt(2.0 / (d_a + d_v))
        self.P = np.random.randn(d_a, d_v) * scale
        self.b_P = np.zeros(d_v)
        
        # Separate branches for each scale
        self.branches = []
        for i, k_max in enumerate(k_max_values):
            branch_seed = seed + i * 100 if seed else None
            branch = []
            for j in range(n_layers):
                layer_seed = branch_seed + j if branch_seed else None
                branch.append(FourierLayer(d_v, k_max, seed=layer_seed))
            self.branches.append(branch)
        
        # Projection from concatenated features
        d_concat = d_v * self.n_branches
        scale1 = np.sqrt(2.0 / (d_concat + d_v))
        self.Q1 = np.random.randn(d_concat, d_v) * scale1
        self.b_Q1 = np.zeros(d_v)
        
        scale2 = np.sqrt(2.0 / (d_v + d_u))
        self.Q2 = np.random.randn(d_v, d_u) * scale2
        self.b_Q2 = np.zeros(d_u)
        
        # Count parameters
        self.n_params = (d_a * d_v + d_v +  # Lifting
                        sum(sum(fl.n_params for fl in branch) for branch in self.branches) +
                        d_concat * d_v + d_v + d_v * d_u + d_u)
    
    def forward(self, a: np.ndarray) -> np.ndarray:
        # Lifting (shared)
        v = a @ self.P + self.b_P
        
        # Process through each branch
        branch_outputs = []
        for branch in self.branches:
            v_branch = v.copy()
            for fl in branch:
                v_branch = fl.forward(v_branch)
            branch_outputs.append(v_branch)
        
        # Concatenate branch outputs
        v_concat = np.concatenate(branch_outputs, axis=-1)
        
        # Projection
        h = gelu(v_concat @ self.Q1 + self.b_Q1)
        u = h @ self.Q2 + self.b_Q2
        
        return u


# Demonstrate advanced architectures
print("\n--- Advanced Architectures Demo ---")

# Standard vs Factorized comparison
d_v, k_max = 32, 12
standard_conv = SpectralConv2d(d_v, d_v, k_max, k_max)
factorized_conv = FactorizedSpectralConv2d(d_v, d_v, k_max)

print("Standard SpectralConv2d:")
print(f"  Parameters: {standard_conv.n_params:,}")

print("Factorized SpectralConv2d:")
print(f"  Parameters: {factorized_conv.n_params:,}")
print(f"  Reduction: {standard_conv.n_params / factorized_conv.n_params:.1f}x")

# Multi-Scale FNO
ms_fno = MultiScaleFNO(d_a=42, d_v=32, d_u=1, k_max_values=[4, 12, 20], n_layers=2)
print(f"\nMulti-Scale FNO (k_max=[4,12,20]):")
print(f"  Parameters: {ms_fno.n_params:,}")

# Test forward pass
test_input = np.random.randn(64, 64, 42)
test_output = ms_fno.forward(test_input)
print(f"  Input: {test_input.shape} → Output: {test_output.shape}")

# U-FNO block
ufno = UFNOBlock(d_v=32, k_max=12)
test_v = np.random.randn(64, 64, 32)
test_out = ufno.forward(test_v)
print(f"\nU-FNO Block: {test_v.shape} → {test_out.shape} (with skip connection)")


#=============================================================================
# SECTION 7: MODEL INTERPRETATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 7: Model Interpretation Tools")
print("="*70)

def analyze_spectral_weights(model: FNO2d, layer_idx: int = 0) -> Dict:
    """
    Analyze spectral weights R of a Fourier layer.
    
    Returns:
    --------
    Dict with weight statistics and important frequencies
    """
    R = model.fourier_layers[layer_idx].spectral_conv.R
    
    # Magnitude at each frequency
    R_magnitude = np.abs(R).mean(axis=(2, 3))  # Average over channels
    
    # Find most important frequencies
    flat_idx = np.argsort(R_magnitude.flatten())[::-1]
    top_k_indices = np.unravel_index(flat_idx[:10], R_magnitude.shape)
    
    analysis = {
        'mean_magnitude': float(np.mean(np.abs(R))),
        'max_magnitude': float(np.max(np.abs(R))),
        'magnitude_by_k': R_magnitude,
        'top_frequencies': list(zip(top_k_indices[0], top_k_indices[1])),
        'low_freq_energy': float(np.sum(R_magnitude[:4, :4])),
        'high_freq_energy': float(np.sum(R_magnitude[4:, 4:])),
    }
    
    return analysis


def compute_channel_importance(model: FNO2d, inputs: np.ndarray, 
                               targets: np.ndarray, normalizer: DataNormalizer,
                               n_permutations: int = 5) -> np.ndarray:
    """
    Compute feature importance via permutation.
    
    Shuffle each input channel and measure prediction degradation.
    """
    d_a = model.d_a
    importance = np.zeros(d_a)
    
    # Baseline error
    baseline_error = 0
    for i in range(len(inputs)):
        x_norm = normalizer.normalize_input(inputs[i:i+1])[0]
        pred = model.forward(x_norm, training=False)
        pred_denorm = normalizer.denormalize_output(pred)
        baseline_error += compute_rmse(pred_denorm, targets[i])
    baseline_error /= len(inputs)
    
    # Permutation importance for each channel
    for c in range(d_a):
        errors = []
        for _ in range(n_permutations):
            inputs_permuted = inputs.copy()
            # Shuffle this channel across samples
            perm = np.random.permutation(len(inputs))
            inputs_permuted[:, :, :, c] = inputs[perm, :, :, c]
            
            # Compute error with permuted feature
            error = 0
            for i in range(len(inputs)):
                x_norm = normalizer.normalize_input(inputs_permuted[i:i+1])[0]
                pred = model.forward(x_norm, training=False)
                pred_denorm = normalizer.denormalize_output(pred)
                error += compute_rmse(pred_denorm, targets[i])
            error /= len(inputs)
            errors.append(error)
        
        # Importance = error increase when shuffled
        importance[c] = np.mean(errors) - baseline_error
    
    return importance


def frequency_ablation_study(model: FNO2d, inputs: np.ndarray, 
                             targets: np.ndarray, normalizer: DataNormalizer,
                             k_cutoffs: List[int] = [2, 4, 8, 12, 16]) -> Dict:
    """
    Study effect of zeroing out different frequency bands.
    """
    results = {}
    
    # Store original weights
    original_weights = model.get_weights()
    
    for k_cut in k_cutoffs:
        # Zero out modes >= k_cut
        model.set_weights(original_weights)
        for layer_idx in range(model.n_layers):
            R = model.fourier_layers[layer_idx].spectral_conv.R
            R[k_cut:, :, :, :] = 0
            R[:, k_cut:, :, :] = 0
        
        # Evaluate
        metrics = evaluate_model(model, inputs, targets, normalizer)
        results[k_cut] = metrics
    
    # Restore original weights
    model.set_weights(original_weights)
    
    return results


def spatial_saliency_map(model: FNO2d, input_field: np.ndarray,
                        target_point: Tuple[int, int],
                        normalizer: DataNormalizer,
                        epsilon: float = 1e-4) -> np.ndarray:
    """
    Compute saliency map: which input locations affect prediction at target point.
    
    Uses finite differences to approximate gradient.
    """
    x_norm = normalizer.normalize_input(input_field[np.newaxis])[0]
    
    Nx, Ny, d_a = x_norm.shape
    saliency = np.zeros((Nx, Ny))
    
    # Baseline prediction at target
    pred_base = model.forward(x_norm, training=False)
    base_value = pred_base[target_point[0], target_point[1], 0]
    
    # Perturb each location
    for i in range(Nx):
        for j in range(Ny):
            x_perturbed = x_norm.copy()
            x_perturbed[i, j, :] += epsilon
            
            pred_perturbed = model.forward(x_perturbed, training=False)
            perturbed_value = pred_perturbed[target_point[0], target_point[1], 0]
            
            saliency[i, j] = np.abs(perturbed_value - base_value) / epsilon
    
    return saliency


# Demonstrate interpretation tools
print("\n--- Model Interpretation Demo ---")

# Create and train a small model
model_interp = FNO2d(d_a=5, d_v=16, d_u=1, k_max=8, n_layers=2, seed=42)

# Analyze spectral weights
analysis = analyze_spectral_weights(model_interp, layer_idx=0)
print("Spectral Weight Analysis (Layer 0):")
print(f"  Mean magnitude: {analysis['mean_magnitude']:.4f}")
print(f"  Max magnitude:  {analysis['max_magnitude']:.4f}")
print(f"  Low-freq energy (k<4):  {analysis['low_freq_energy']:.4f}")
print(f"  High-freq energy (k≥4): {analysis['high_freq_energy']:.4f}")
print(f"  Top 5 frequencies: {analysis['top_frequencies'][:5]}")

# Visualize spectral weights
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(analysis['magnitude_by_k'], cmap='hot', origin='lower')
axes[0].set_xlabel('ky')
axes[0].set_ylabel('kx')
axes[0].set_title('Spectral Weight Magnitude |R(kx, ky)|')
plt.colorbar(im1, ax=axes[0])

# Create synthetic data for saliency
Nx_sal, Ny_sal = 32, 32
x_sal = np.linspace(0, 2*np.pi, Nx_sal)
y_sal = np.linspace(0, 2*np.pi, Ny_sal)
X_sal, Y_sal = np.meshgrid(x_sal, y_sal, indexing='ij')

input_sal = np.stack([
    np.sin(X_sal),
    np.cos(Y_sal),
    np.sin(2*X_sal),
    np.cos(2*Y_sal),
    np.random.randn(Nx_sal, Ny_sal) * 0.1
], axis=-1)

normalizer_sal = DataNormalizer()
normalizer_sal.fit(input_sal[np.newaxis], input_sal[np.newaxis, :, :, :1])

# Compute saliency for center point
saliency = spatial_saliency_map(model_interp, input_sal, (16, 16), normalizer_sal)

im2 = axes[1].imshow(saliency, cmap='hot', origin='lower')
axes[1].plot(16, 16, 'c*', markersize=15, label='Target point')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Saliency Map: What affects prediction at (16,16)?')
axes[1].legend()
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('figures/02_model_interpretation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: figures/02_model_interpretation.png")


#=============================================================================
# SECTION 8: PHYSICS VERIFICATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 8: Physics Verification Tests")
print("="*70)

def verify_vegetation_cooling(model: FNO2d, base_input: np.ndarray,
                              ndvi_channel: int, normalizer: DataNormalizer,
                              ndvi_increase: float = 0.3) -> Dict:
    """
    Test: Does increasing NDVI decrease predicted temperature?
    """
    x_norm = normalizer.normalize_input(base_input[np.newaxis])[0]
    
    # Baseline prediction
    pred_base = model.forward(x_norm, training=False)
    pred_base_denorm = normalizer.denormalize_output(pred_base)
    
    # Increase NDVI in center region
    x_modified = x_norm.copy()
    Nx, Ny = x_norm.shape[:2]
    region = slice(Nx//4, 3*Nx//4), slice(Ny//4, 3*Ny//4)
    x_modified[region[0], region[1], ndvi_channel] += ndvi_increase
    
    pred_modified = model.forward(x_modified, training=False)
    pred_modified_denorm = normalizer.denormalize_output(pred_modified)
    
    # Compare temperatures in modified region
    temp_base_region = pred_base_denorm[region[0], region[1], 0].mean()
    temp_modified_region = pred_modified_denorm[region[0], region[1], 0].mean()
    
    return {
        'base_temp': float(temp_base_region),
        'modified_temp': float(temp_modified_region),
        'temp_change': float(temp_modified_region - temp_base_region),
        'physics_correct': temp_modified_region < temp_base_region,
        'description': 'Vegetation should COOL (negative temp change expected)'
    }


def verify_urban_heating(model: FNO2d, base_input: np.ndarray,
                         ndbi_channel: int, normalizer: DataNormalizer,
                         ndbi_increase: float = 0.3) -> Dict:
    """
    Test: Does increasing NDBI increase predicted temperature?
    """
    x_norm = normalizer.normalize_input(base_input[np.newaxis])[0]
    
    pred_base = model.forward(x_norm, training=False)
    pred_base_denorm = normalizer.denormalize_output(pred_base)
    
    x_modified = x_norm.copy()
    Nx, Ny = x_norm.shape[:2]
    region = slice(Nx//4, 3*Nx//4), slice(Ny//4, 3*Ny//4)
    x_modified[region[0], region[1], ndbi_channel] += ndbi_increase
    
    pred_modified = model.forward(x_modified, training=False)
    pred_modified_denorm = normalizer.denormalize_output(pred_modified)
    
    temp_base_region = pred_base_denorm[region[0], region[1], 0].mean()
    temp_modified_region = pred_modified_denorm[region[0], region[1], 0].mean()
    
    return {
        'base_temp': float(temp_base_region),
        'modified_temp': float(temp_modified_region),
        'temp_change': float(temp_modified_region - temp_base_region),
        'physics_correct': temp_modified_region > temp_base_region,
        'description': 'Urban surfaces should WARM (positive temp change expected)'
    }


def verify_park_cooling_extent(model: FNO2d, normalizer: DataNormalizer,
                               park_radius: int = 5, ndvi_channel: int = 0) -> Dict:
    """
    Test: Does a park create a cooling effect that extends beyond its boundary?
    """
    Nx, Ny = 64, 64
    
    # Create urban background (high NDBI, low NDVI)
    input_field = np.zeros((Nx, Ny, model.d_a))
    input_field[:, :, ndvi_channel] = -0.2  # Low vegetation
    input_field[:, :, 1] = 0.5  # High urban (assuming channel 1 is NDBI)
    
    # Add a circular park in center
    cx, cy = Nx // 2, Ny // 2
    Y, X = np.ogrid[:Nx, :Ny]
    park_mask = (X - cx)**2 + (Y - cy)**2 <= park_radius**2
    input_field[park_mask, ndvi_channel] = 0.6  # High vegetation in park
    input_field[park_mask, 1] = -0.3  # Low urban in park
    
    # Normalize and predict
    x_norm = normalizer.normalize_input(input_field[np.newaxis])[0]
    pred = model.forward(x_norm, training=False)
    pred_denorm = normalizer.denormalize_output(pred)[:, :, 0]
    
    # Analyze temperature at different distances from park center
    distances = [0, park_radius, park_radius*2, park_radius*3]
    temps_by_distance = {}
    
    for d in distances:
        if d == 0:
            temp = pred_denorm[cx, cy]
        else:
            # Average temperature at distance d
            ring_mask = ((X - cx)**2 + (Y - cy)**2 >= (d-1)**2) & \
                       ((X - cx)**2 + (Y - cy)**2 <= (d+1)**2)
            if np.sum(ring_mask) > 0:
                temp = pred_denorm[ring_mask].mean()
            else:
                temp = np.nan
        temps_by_distance[d] = float(temp)
    
    return {
        'temps_by_distance': temps_by_distance,
        'park_radius': park_radius,
        'cooling_extends_beyond': temps_by_distance.get(park_radius*2, 0) < 
                                  temps_by_distance.get(park_radius*3, 0),
        'description': 'Temperature should increase with distance from park'
    }


# Demonstrate physics verification
print("\n--- Physics Verification Demo ---")

# Create model and normalizer
model_phys = FNO2d(d_a=5, d_v=16, d_u=1, k_max=8, n_layers=2, seed=42)

test_input_phys = np.random.randn(64, 64, 5) * 0.3
test_output_phys = np.random.randn(64, 64, 1) * 10 + 300

normalizer_phys = DataNormalizer()
normalizer_phys.fit(test_input_phys[np.newaxis], test_output_phys[np.newaxis])

# Test vegetation cooling (channel 0 = NDVI)
veg_result = verify_vegetation_cooling(model_phys, test_input_phys, 
                                        ndvi_channel=0, normalizer=normalizer_phys)
print("\nVegetation Cooling Test:")
print(f"  Base temp: {veg_result['base_temp']:.2f}")
print(f"  After NDVI increase: {veg_result['modified_temp']:.2f}")
print(f"  Change: {veg_result['temp_change']:.2f}")
print(f"  Physics correct: {veg_result['physics_correct']} ({veg_result['description']})")

# Test urban heating (channel 1 = NDBI)
urban_result = verify_urban_heating(model_phys, test_input_phys,
                                    ndbi_channel=1, normalizer=normalizer_phys)
print("\nUrban Heating Test:")
print(f"  Base temp: {urban_result['base_temp']:.2f}")
print(f"  After NDBI increase: {urban_result['modified_temp']:.2f}")
print(f"  Change: {urban_result['temp_change']:.2f}")
print(f"  Physics correct: {urban_result['physics_correct']} ({urban_result['description']})")

# Test park cooling extent
park_result = verify_park_cooling_extent(model_phys, normalizer_phys, park_radius=8)
print("\nPark Cooling Extent Test:")
print(f"  Temperature by distance from park:")
for d, t in park_result['temps_by_distance'].items():
    print(f"    d={d:2d}: {t:.2f}")
print(f"  Cooling extends beyond park: {park_result['cooling_extends_beyond']}")

print("\n[Note: Random weights show physics patterns are not yet learned]")
print("[After training, physics tests should pass]")


#=============================================================================
# SECTION 9: VISUALIZATION SUITE
#=============================================================================
print("\n" + "="*70)
print("SECTION 9: Comprehensive Visualization Suite")
print("="*70)

def visualize_predictions(model: FNO2d, inputs: np.ndarray, targets: np.ndarray,
                         normalizer: DataNormalizer, n_samples: int = 4,
                         save_path: Optional[str] = None):
    """
    Visualize model predictions vs ground truth.
    """
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    indices = np.random.choice(len(inputs), n_samples, replace=False)
    
    for row, idx in enumerate(indices):
        x_norm = normalizer.normalize_input(inputs[idx:idx+1])[0]
        pred_norm = model.forward(x_norm, training=False)
        pred = normalizer.denormalize_output(pred_norm)
        target = targets[idx]
        
        # Input (first channel)
        im1 = axes[row, 0].imshow(inputs[idx, :, :, 0], cmap='viridis', origin='lower')
        axes[row, 0].set_title(f'Input Ch0 (Sample {idx})')
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.046)
        
        # Target
        im2 = axes[row, 1].imshow(target[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 1].set_title('Target')
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046)
        
        # Prediction
        im3 = axes[row, 2].imshow(pred[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 2].set_title('Prediction')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046)
        
        # Error
        error = pred[:, :, 0] - target[:, :, 0]
        max_err = max(abs(error.min()), abs(error.max()))
        im4 = axes[row, 3].imshow(error, cmap='RdBu_r', origin='lower',
                                  vmin=-max_err, vmax=max_err)
        axes[row, 3].set_title(f'Error (RMSE={compute_rmse(pred, target):.4f})')
        plt.colorbar(im4, ax=axes[row, 3], fraction=0.046)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_frequency_response(model: FNO2d, save_path: Optional[str] = None):
    """
    Visualize how FNO processes different frequency components.
    """
    Nx, Ny = 64, 64
    x = np.linspace(0, 2*np.pi, Nx)
    y = np.linspace(0, 2*np.pi, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create inputs at different frequencies
    frequencies = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(len(frequencies), 3, figsize=(12, 4*len(frequencies)))
    
    normalizer_temp = DataNormalizer()
    dummy_input = np.random.randn(1, Nx, Ny, model.d_a)
    dummy_output = np.random.randn(1, Nx, Ny, model.d_u)
    normalizer_temp.fit(dummy_input, dummy_output)
    
    for row, freq in enumerate(frequencies):
        # Create sinusoidal input
        input_field = np.zeros((Nx, Ny, model.d_a))
        input_field[:, :, 0] = np.sin(freq * X) * np.cos(freq * Y)
        
        # Normalize and predict
        x_norm = normalizer_temp.normalize_input(input_field[np.newaxis])[0]
        output = model.forward(x_norm, training=False)
        
        # Input
        axes[row, 0].imshow(input_field[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 0].set_title(f'Input (k={freq})')
        
        # Output
        axes[row, 1].imshow(output[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 1].set_title(f'Output (k={freq})')
        
        # Spectra comparison
        in_spec = np.abs(np.fft.fft2(input_field[:, :, 0]))[:Nx//2, :Ny//2]
        out_spec = np.abs(np.fft.fft2(output[:, :, 0]))[:Nx//2, :Ny//2]
        
        axes[row, 2].plot(in_spec.sum(axis=1), 'b-', label='Input', alpha=0.7)
        axes[row, 2].plot(out_spec.sum(axis=1), 'r-', label='Output', alpha=0.7)
        axes[row, 2].axvline(x=model.k_max, color='g', linestyle='--', label=f'k_max={model.k_max}')
        axes[row, 2].set_xlabel('k')
        axes[row, 2].set_ylabel('Spectral Energy')
        axes[row, 2].legend()
        axes[row, 2].set_title('Spectrum (kx marginalized)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_resolution_invariance(model: FNO2d, save_path: Optional[str] = None):
    """
    Demonstrate resolution invariance.
    """
    resolutions = [32, 64, 128]
    
    fig, axes = plt.subplots(len(resolutions), 3, figsize=(12, 4*len(resolutions)))
    
    for row, res in enumerate(resolutions):
        # Create smooth input field
        x = np.linspace(0, 2*np.pi, res)
        y = np.linspace(0, 2*np.pi, res)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        input_field = np.zeros((res, res, model.d_a))
        input_field[:, :, 0] = np.sin(X) + 0.5*np.cos(2*Y)
        
        # Simple normalization
        input_norm = input_field / (np.abs(input_field).max() + 1e-8)
        
        # Forward pass
        output = model.forward(input_norm, training=False)
        
        # Visualize
        axes[row, 0].imshow(input_field[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 0].set_title(f'Input ({res}×{res})')
        
        axes[row, 1].imshow(output[:, :, 0], cmap='RdBu_r', origin='lower')
        axes[row, 1].set_title(f'Output ({res}×{res})')
        
        # Spectrum
        out_spec = np.abs(np.fft.rfft2(output[:, :, 0]))
        axes[row, 2].imshow(np.log10(out_spec[:res//4, :res//4] + 1e-10), 
                           cmap='viridis', origin='lower')
        axes[row, 2].set_title(f'Output Spectrum (log scale)')
    
    plt.suptitle(f'Resolution Invariance: Same {model.n_params:,} parameters at all resolutions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Generate visualizations
print("\n--- Generating Visualizations ---")

# Predictions
visualize_predictions(model_demo, demo_inputs[val_idx_demo], demo_targets[val_idx_demo],
                     normalizer_demo, n_samples=3, save_path='figures/03_predictions.png')
print("✓ Saved: figures/03_predictions.png")

# Frequency response
visualize_frequency_response(model_demo, save_path='figures/04_frequency_response.png')
print("✓ Saved: figures/04_frequency_response.png")

# Resolution invariance
visualize_resolution_invariance(model_demo, save_path='figures/05_resolution_invariance.png')
print("✓ Saved: figures/05_resolution_invariance.png")


#=============================================================================
# SECTION 10: URBAN TEMPERATURE APPLICATION TEMPLATE
#=============================================================================
print("\n" + "="*70)
print("SECTION 10: Urban Temperature Application Template")
print("="*70)

class UrbanTemperatureFNO:
    """
    Complete FNO pipeline for urban temperature prediction.
    Configured for your ECOSTRESS-based thesis work.
    """
    
    # Feature configuration for your 42 features
    FEATURE_CONFIG = {
        'NDVI': 0,
        'NDBI': 1,
        'NDWI': 2,
        'building_height': 3,
        'building_density': 4,
        'SVF': 5,
        'ERA5_temp': 6,
        'ERA5_RH': 7,
        'ERA5_u': 8,
        'ERA5_v': 9,
        'solar_radiation': 10,
        # ... additional features
    }
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        if config is None:
            config = TrainingConfig(
                d_a=42,
                d_v=32,
                d_u=1,
                k_max=12,
                n_layers=4,
                dropout=0.1,
                use_physics=True,
                lambda_smooth=0.01,
                lambda_veg=0.1,
            )
        
        self.config = config
        self.model = None
        self.normalizer = None
        self.logger = None
        self.is_trained = False
    
    def prepare_data(self, raw_inputs: np.ndarray, raw_outputs: np.ndarray,
                     train_frac: float = 0.7, val_frac: float = 0.15,
                     seed: int = 42) -> Dict:
        """
        Prepare data for training.
        
        Parameters:
        -----------
        raw_inputs : np.ndarray, shape (N, Nx, Ny, 42)
            Your 42-feature inputs
        raw_outputs : np.ndarray, shape (N, Nx, Ny, 1)
            ECOSTRESS LST
        """
        # Split
        train_idx, val_idx, test_idx = train_val_test_split(
            len(raw_inputs), train_frac, val_frac, seed
        )
        
        # Fit normalizer on training only
        self.normalizer = DataNormalizer()
        self.normalizer.fit(raw_inputs[train_idx], raw_outputs[train_idx])
        
        return {
            'train_inputs': raw_inputs[train_idx],
            'train_targets': raw_outputs[train_idx],
            'val_inputs': raw_inputs[val_idx],
            'val_targets': raw_outputs[val_idx],
            'test_inputs': raw_inputs[test_idx],
            'test_targets': raw_outputs[test_idx],
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
        }
    
    def build_model(self, seed: int = 42):
        """Initialize the FNO model."""
        self.model = FNO2d(
            d_a=self.config.d_a,
            d_v=self.config.d_v,
            d_u=self.config.d_u,
            k_max=self.config.k_max,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            seed=seed
        )
        self.logger = TrainingLogger()
        return self.model
    
    def train(self, data: Dict) -> TrainingLogger:
        """
        Train the model.
        
        NOTE: This is a simulation. Real training requires PyTorch/JAX.
        """
        if self.model is None:
            self.build_model()
        
        augmentation = DataAugmentation() if self.config.augment else None
        best_weights = self.model.get_weights()
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            lr = get_lr(epoch, self.config)
            
            # Train
            train_loss = train_epoch(
                self.model, data['train_inputs'], data['train_targets'],
                self.normalizer, self.config, augmentation, lr
            )
            
            # Validate
            val_loss, val_metrics = validate(
                self.model, data['val_inputs'], data['val_targets'],
                self.normalizer
            )
            
            self.logger.log(epoch, train_loss, val_loss, val_metrics, lr)
            
            # Early stopping
            if val_loss < self.logger.best_val_loss - self.config.min_delta:
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch:3d}: loss={train_loss:.4f}/{val_loss:.4f}, "
                      f"spatial_r2={val_metrics['spatial_anomaly_r2']:.4f}")
        
        # Restore best weights
        self.model.set_weights(best_weights)
        self.is_trained = True
        
        return self.logger
    
    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> Dict:
        """Comprehensive evaluation."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        return evaluate_model(self.model, inputs, targets, self.normalizer)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        predictions = []
        for i in range(len(inputs)):
            x_norm = self.normalizer.normalize_input(inputs[i:i+1])[0]
            pred_norm = self.model.forward(x_norm, training=False)
            pred = self.normalizer.denormalize_output(pred_norm)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def verify_physics(self, test_input: np.ndarray) -> Dict:
        """Run physics verification tests."""
        results = {}
        
        # Vegetation cooling
        results['vegetation'] = verify_vegetation_cooling(
            self.model, test_input, 
            ndvi_channel=self.FEATURE_CONFIG['NDVI'],
            normalizer=self.normalizer
        )
        
        # Urban heating
        results['urban'] = verify_urban_heating(
            self.model, test_input,
            ndbi_channel=self.FEATURE_CONFIG['NDBI'],
            normalizer=self.normalizer
        )
        
        return results
    
    def save(self, path: str):
        """Save model and normalizer."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.model.get_weights(),
                'config': self.config,
                'normalizer_stats': self.normalizer.get_stats() if self.normalizer else None,
            }, f)
    
    def summary(self) -> str:
        """Get model summary."""
        n_params = f"{self.model.n_params:,}" if self.model else "N/A"
        return f"""
Urban Temperature FNO
=====================
Input features: {self.config.d_a}
Hidden dimension: {self.config.d_v}
Fourier modes: {self.config.k_max}
Layers: {self.config.n_layers}
Total parameters: {n_params}
Physics-informed: {self.config.use_physics}
Trained: {self.is_trained}
"""


# Demonstrate application template
print("\n--- Urban Temperature Application Demo ---")

# Create application
app = UrbanTemperatureFNO()
print(app.summary())

# Simulate with synthetic data (mimicking your 42-feature problem)
N_app = 80
Nx_app, Ny_app = 32, 32
np.random.seed(42)

# Create synthetic "ECOSTRESS-like" data
app_inputs = np.random.randn(N_app, Nx_app, Ny_app, 42)
app_targets = np.random.randn(N_app, Nx_app, Ny_app, 1) * 10 + 300

# Prepare data
data = app.prepare_data(app_inputs, app_targets)
print(f"\nData split: {len(data['train_inputs'])} train, "
      f"{len(data['val_inputs'])} val, {len(data['test_inputs'])} test")

# Build model
app.build_model(seed=42)
print(f"Model parameters: {app.model.n_params:,}")

# Quick training simulation (reduced epochs for demo)
app.config.n_epochs = 30
app.config.log_interval = 10
print("\nTraining (simulated)...")
logger = app.train(data)

# Evaluate
print("\nTest set evaluation:")
test_metrics = app.evaluate(data['test_inputs'], data['test_targets'])
for name, value in test_metrics.items():
    if isinstance(value, float):
        print(f"  {name}: {value:.4f}")

print(app.summary())


#=============================================================================
# SECTION 11: DIAGNOSTIC TOOLS
#=============================================================================
print("\n" + "="*70)
print("SECTION 11: Diagnostic and Debugging Tools")
print("="*70)

def check_data_quality(inputs: np.ndarray, outputs: np.ndarray) -> Dict:
    """
    Check data quality and report potential issues.
    """
    issues = []
    
    # Check for NaN/Inf
    if np.any(np.isnan(inputs)):
        issues.append(f"NaN values in inputs: {np.sum(np.isnan(inputs))} total")
    if np.any(np.isinf(inputs)):
        issues.append(f"Inf values in inputs: {np.sum(np.isinf(inputs))} total")
    if np.any(np.isnan(outputs)):
        issues.append(f"NaN values in outputs: {np.sum(np.isnan(outputs))} total")
    
    # Check for constant features
    for c in range(inputs.shape[-1]):
        if np.std(inputs[:, :, :, c]) < 1e-10:
            issues.append(f"Channel {c} is constant (zero variance)")
    
    # Check for extreme values
    for c in range(inputs.shape[-1]):
        max_val = np.max(np.abs(inputs[:, :, :, c]))
        if max_val > 1000:
            issues.append(f"Channel {c} has extreme values (max |x| = {max_val:.1f})")
    
    # Check for imbalanced scales
    stds = [np.std(inputs[:, :, :, c]) for c in range(inputs.shape[-1])]
    if max(stds) / (min(stds) + 1e-10) > 1000:
        issues.append(f"Highly imbalanced scales: std ratio = {max(stds)/(min(stds)+1e-10):.1f}")
    
    return {
        'n_samples': inputs.shape[0],
        'spatial_shape': inputs.shape[1:3],
        'n_features': inputs.shape[-1],
        'input_range': (float(inputs.min()), float(inputs.max())),
        'output_range': (float(outputs.min()), float(outputs.max())),
        'issues': issues,
        'quality_ok': len(issues) == 0
    }


def diagnose_training_issues(logger: TrainingLogger) -> List[str]:
    """
    Analyze training history and diagnose potential issues.
    """
    diagnoses = []
    
    if len(logger.train_losses) < 10:
        diagnoses.append("Training too short for diagnosis")
        return diagnoses
    
    train_losses = np.array(logger.train_losses)
    val_losses = np.array(logger.val_losses)
    
    # Check for NaN
    if np.any(np.isnan(train_losses)):
        diagnoses.append("NaN in training loss - check data normalization and learning rate")
    
    # Check for no learning
    if train_losses[-1] > 0.95 * train_losses[0]:
        diagnoses.append("Model not learning - try increasing learning rate or checking data")
    
    # Check for overfitting
    if len(train_losses) > 20:
        recent_train = np.mean(train_losses[-10:])
        recent_val = np.mean(val_losses[-10:])
        if recent_val > 2 * recent_train:
            diagnoses.append("Severe overfitting - reduce model size or add regularization")
        elif recent_val > 1.5 * recent_train:
            diagnoses.append("Moderate overfitting - consider more dropout or early stopping")
    
    # Check for underfitting
    if np.mean(train_losses[-10:]) > 0.1:
        diagnoses.append("Possible underfitting - try larger model or more training")
    
    # Check learning rate
    if logger.learning_rates[-1] < 1e-7:
        diagnoses.append("Learning rate very small - training may have stalled")
    
    if not diagnoses:
        diagnoses.append("No obvious issues detected - training looks healthy")
    
    return diagnoses


# Demonstrate diagnostics
print("\n--- Data Quality Check ---")
quality = check_data_quality(app_inputs, app_targets)
print(f"Samples: {quality['n_samples']}")
print(f"Spatial shape: {quality['spatial_shape']}")
print(f"Features: {quality['n_features']}")
print(f"Input range: {quality['input_range']}")
print(f"Output range: {quality['output_range']}")
print(f"Quality OK: {quality['quality_ok']}")
if quality['issues']:
    print("Issues found:")
    for issue in quality['issues']:
        print(f"  - {issue}")

print("\n--- Training Diagnosis ---")
diagnoses = diagnose_training_issues(logger)
for d in diagnoses:
    print(f"  • {d}")


#=============================================================================
# SECTION 12: PYTORCH TEMPLATE WITH PHYSICS
#=============================================================================
print("\n" + "="*70)
print("SECTION 12: PyTorch Template with Physics-Informed Training")
print("="*70)

pytorch_pino_template = '''
"""
Physics-Informed FNO (PINO) Implementation in PyTorch
Complete template for training on urban temperature data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution with proper complex handling"""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 
                              dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 
                              dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), 
                            x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """Complete FNO with configurable architecture"""
    
    def __init__(self, d_a, d_v, d_u, modes, n_layers=4, dropout=0.1):
        super().__init__()
        self.d_a = d_a
        self.d_v = d_v
        self.modes = modes
        
        # Lifting
        self.fc0 = nn.Linear(d_a, d_v)
        
        # Fourier layers
        self.convs = nn.ModuleList([
            SpectralConv2d(d_v, d_v, modes, modes) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(d_v, d_v, 1) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Projection
        self.fc1 = nn.Linear(d_v, d_v * 2)
        self.fc2 = nn.Linear(d_v * 2, d_u)
    
    def forward(self, x):
        # x: (batch, nx, ny, d_a)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, d_v, nx, ny)
        
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = self.dropout(w(x))
            x = F.gelu(x1 + x2)
        
        x = x.permute(0, 2, 3, 1)  # (batch, nx, ny, d_v)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Physics-Informed Loss Functions
def compute_laplacian(field):
    """Compute Laplacian using spectral method"""
    ft = torch.fft.fft2(field)
    nx, ny = field.shape[-2:]
    
    kx = torch.fft.fftfreq(nx, device=field.device) * 2 * np.pi
    ky = torch.fft.fftfreq(ny, device=field.device) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    laplacian_ft = -(KX**2 + KY**2) * ft
    return torch.real(torch.fft.ifft2(laplacian_ft))


def smoothness_loss(pred):
    """Penalize non-smooth predictions"""
    laplacian = compute_laplacian(pred.squeeze(-1))
    return torch.mean(laplacian**2)


def vegetation_cooling_loss(pred, ndvi):
    """Penalize positive correlation between temperature and vegetation"""
    pred_flat = pred.flatten() - pred.mean()
    ndvi_flat = ndvi.flatten() - ndvi.mean()
    
    corr = torch.sum(pred_flat * ndvi_flat) / (
        torch.sqrt(torch.sum(pred_flat**2)) * 
        torch.sqrt(torch.sum(ndvi_flat**2)) + 1e-8
    )
    return F.relu(corr)**2


class PINOLoss(nn.Module):
    """Combined data + physics loss"""
    
    def __init__(self, lambda_smooth=0.01, lambda_veg=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_veg = lambda_veg
    
    def forward(self, pred, target, ndvi=None):
        # Data loss
        loss_data = F.mse_loss(pred, target)
        
        # Smoothness
        loss_smooth = smoothness_loss(pred)
        
        # Vegetation
        loss_veg = 0.0
        if ndvi is not None:
            loss_veg = vegetation_cooling_loss(pred.squeeze(-1), ndvi)
        
        total = loss_data + self.lambda_smooth * loss_smooth + self.lambda_veg * loss_veg
        
        return total, {
            'data': loss_data.item(),
            'smooth': loss_smooth.item(),
            'veg': loss_veg if isinstance(loss_veg, float) else loss_veg.item()
        }


def train_pino(model, train_loader, val_loader, n_epochs=300, lr=1e-3, 
               device='cuda', use_physics=True):
    """Complete training loop with physics"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    if use_physics:
        criterion = PINOLoss(lambda_smooth=0.01, lambda_veg=0.1)
    else:
        criterion = lambda p, t, n=None: (F.mse_loss(p, t), {'data': F.mse_loss(p, t).item()})
    
    best_val_loss = float('inf')
    history = {'train': [], 'val': [], 'physics': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            ndvi = batch[2].to(device) if len(batch) > 2 else None
            
            optimizer.zero_grad()
            pred = model(x)
            loss, loss_dict = criterion(pred, y, ndvi)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                val_loss += F.mse_loss(pred, y).item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pino_model.pt')
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_pino_model.pt'))
    return model, history


# Evaluation with spatial anomaly R²
def compute_spatial_anomaly_r2(pred, target):
    """THE KEY METRIC for urban temperature"""
    pred_anom = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target_anom = target - target.mean(dim=(-2, -1), keepdim=True)
    
    ss_res = torch.sum((target_anom - pred_anom)**2)
    ss_tot = torch.sum(target_anom**2)
    
    return 1 - ss_res / ss_tot


# Usage Example:
"""
# Data preparation
X_train = torch.tensor(train_inputs, dtype=torch.float32)  # (N, Nx, Ny, 42)
y_train = torch.tensor(train_outputs, dtype=torch.float32)  # (N, Nx, Ny, 1)
ndvi_train = X_train[:, :, :, 0:1]  # Extract NDVI channel

train_dataset = TensorDataset(X_train, y_train, ndvi_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
model = FNO2d(d_a=42, d_v=32, d_u=1, modes=12, n_layers=4, dropout=0.1)

# Train with physics
trained_model, history = train_pino(
    model, train_loader, val_loader, 
    n_epochs=300, use_physics=True
)

# Evaluate
model.eval()
with torch.no_grad():
    pred = model(X_test.to(device))
    spatial_r2 = compute_spatial_anomaly_r2(pred, y_test.to(device))
    print(f"Spatial Anomaly R²: {spatial_r2.item():.4f}")
"""
'''

# Save template
with open('pytorch_pino_template.py', 'w') as f:
    f.write(pytorch_pino_template)
print("✓ Saved: pytorch_pino_template.py")


#=============================================================================
# FINAL SUMMARY
#=============================================================================
print("\n" + "="*70)
print("CHUNK 4 COMPLETE: Advanced FNO Implementation")
print("="*70)

print("""
Files Created:

Code:
  chunk4_code.py                - This comprehensive implementation (~1500 lines)
  pytorch_pino_template.py      - PyTorch PINO implementation

Figures:
  figures/01_training_history.png     - Training curves visualization
  figures/02_model_interpretation.png - Spectral weights and saliency
  figures/03_predictions.png          - Model predictions vs targets
  figures/04_frequency_response.png   - Frequency response analysis
  figures/05_resolution_invariance.png - Resolution invariance demo

Key Components Implemented:

1. DATA PREPARATION
   - DataNormalizer: Per-channel Z-score normalization
   - DataAugmentation: Flips, rotations for spatial data
   - train_val_test_split: Proper data splitting

2. LOSS FUNCTIONS
   Standard:
   - mse_loss, mae_loss, relative_l2_loss
   
   Physics-Informed:
   - smoothness_loss (∇²T penalty)
   - vegetation_cooling_loss (NDVI-temperature)
   - urban_heating_loss (NDBI-temperature)
   - conservation_loss (integral preservation)
   - PhysicsInformedLoss (combined PINO loss)

3. EVALUATION METRICS
   - compute_rmse, compute_mae, compute_r2
   - compute_spatial_anomaly_r2 ← YOUR KEY METRIC
   - compute_spectral_error (frequency-band analysis)
   - evaluate_model (comprehensive evaluation)

4. TRAINING PIPELINE
   - TrainingConfig: All hyperparameters
   - TrainingLogger: Tracks history
   - train_epoch, validate functions
   - Early stopping, checkpointing

5. ADVANCED ARCHITECTURES
   - FactorizedSpectralConv2d: 12x parameter reduction
   - UFNOBlock: U-Net style skip connections
   - MultiScaleFNO: Parallel branches for different scales

6. MODEL INTERPRETATION
   - analyze_spectral_weights: Which frequencies matter
   - compute_channel_importance: Permutation importance
   - frequency_ablation_study: Effect of zeroing frequencies
   - spatial_saliency_map: What affects each prediction

7. PHYSICS VERIFICATION
   - verify_vegetation_cooling: Does NDVI↑ → T↓?
   - verify_urban_heating: Does NDBI↑ → T↑?
   - verify_park_cooling_extent: Does cooling extend beyond parks?

8. APPLICATION TEMPLATE
   - UrbanTemperatureFNO: Complete pipeline for your thesis
   - Configured for 42 features → temperature

9. DIAGNOSTICS
   - check_data_quality: Find data issues
   - diagnose_training_issues: Identify training problems

Recommended Configuration for Your Thesis:
=========================================
d_a = 42          # Your ECOSTRESS features
d_v = 32          # Hidden dimension
d_u = 1           # Temperature output
k_max = 12        # Fourier modes  
n_layers = 4      # Depth
dropout = 0.1     # Regularization
use_physics = True
lambda_smooth = 0.01
lambda_veg = 0.1

Target: Spatial Anomaly R² > 0.70 (vs RF's 0.48-0.75)

Next Steps:
===========
1. Prepare your ECOSTRESS data: (N, Nx, Ny, 42) format
2. Run data quality checks
3. Start with small model (d_v=32, k_max=12)
4. Add physics constraints with λ=0.01-0.1
5. Monitor spatial_anomaly_r2 (not just loss!)
6. Verify physics learning after training
7. Document for PhD applications
""")
