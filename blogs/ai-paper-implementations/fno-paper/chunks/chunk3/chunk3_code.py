"""
Fourier Neural Operator: Chunk 3 - Complete FNO Architecture
Comprehensive Implementation with Training and Evaluation

This file implements everything from the Chunk 3 theory guide:
1. Lifting Layer (P): Input → Hidden representation
2. Fourier Layer Stack: Multiple spectral convolution layers
3. Projection Layer (Q): Hidden → Output
4. Complete FNO2d class
5. Loss functions (MSE, Relative L2, MAE)
6. Data normalization utilities
7. Training loop with validation
8. Evaluation metrics (RMSE, MAE, R², Spatial Anomaly R²)
9. Synthetic PDE data for testing
10. Comprehensive visualizations

Author: FNO Tutorial for PhD Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from typing import Tuple, Dict, List, Optional
import time

# Create output directory
os.makedirs('figures', exist_ok=True)

print("="*70)
print("CHUNK 3: COMPLETE FNO ARCHITECTURE")
print("="*70)


#=============================================================================
# SECTION 1: ACTIVATION FUNCTION (from Chunk 2)
#=============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.
    GELU(x) = x * Φ(x) where Φ is standard normal CDF.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


#=============================================================================
# SECTION 2: LIFTING LAYER
#=============================================================================
print("\n" + "="*70)
print("SECTION 2: Lifting Layer (P)")
print("="*70)

class LiftingLayer:
    """
    Lifting Layer: Projects input features to hidden representation.
    
    Operates POINTWISE at each spatial location:
        v_0(x) = P @ a(x) + b_P
    
    Parameters:
    -----------
    d_in : int
        Input dimension (number of input features, e.g., 42)
    d_out : int
        Output dimension (hidden dimension, e.g., 64)
    """
    
    def __init__(self, d_in: int, d_out: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_in = d_in
        self.d_out = d_out
        
        # Initialize weights using Xavier initialization
        scale = np.sqrt(2.0 / (d_in + d_out))
        self.P = np.random.randn(d_in, d_out) * scale
        self.b = np.zeros(d_out)
        
        self.n_params = d_in * d_out + d_out
    
    def forward(self, a: np.ndarray) -> np.ndarray:
        """
        Forward pass through lifting layer.
        
        Parameters:
        -----------
        a : np.ndarray, shape (Nx, Ny, d_in)
            Input features at each spatial location
            
        Returns:
        --------
        v : np.ndarray, shape (Nx, Ny, d_out)
            Hidden representation
        """
        # Simple matrix multiply at each point: v = a @ P + b
        return a @ self.P + self.b
    
    def __repr__(self):
        return f"LiftingLayer({self.d_in} → {self.d_out}, params={self.n_params:,})"


# Demonstrate lifting layer
print("\n--- Lifting Layer Demo ---")

d_in = 42   # Your 42 input features
d_out = 64  # Hidden dimension

lifting = LiftingLayer(d_in, d_out, seed=42)
print(lifting)

# Create sample input
Nx, Ny = 32, 32
a_sample = np.random.randn(Nx, Ny, d_in)

# Forward pass
v0 = lifting.forward(a_sample)

print(f"\nInput shape:  {a_sample.shape} = (Nx, Ny, d_in)")
print(f"Output shape: {v0.shape} = (Nx, Ny, d_out)")
print(f"Weight P shape: {lifting.P.shape}")
print(f"Bias b shape: {lifting.b.shape}")

# Verify pointwise operation
point = (10, 15)
manual_result = a_sample[point] @ lifting.P + lifting.b
auto_result = v0[point]
print(f"\nVerify pointwise: max difference = {np.max(np.abs(manual_result - auto_result)):.2e}")


#=============================================================================
# SECTION 3: FOURIER LAYER (Enhanced from Chunk 2)
#=============================================================================
print("\n" + "="*70)
print("SECTION 3: Fourier Layer (Enhanced)")
print("="*70)

class FourierLayer:
    """
    Complete Fourier Layer: v^(l+1) = σ(Wv + Kv + b)
    
    Components:
    - Spectral convolution K: FFT → truncate → multiply R → IFFT
    - Local linear path W: pointwise transformation
    - Bias b
    - GELU activation
    
    Parameters:
    -----------
    d_v : int
        Hidden dimension (input and output channels are same)
    k_max_x : int
        Number of Fourier modes in x direction
    k_max_y : int
        Number of Fourier modes in y direction
    """
    
    def __init__(self, d_v: int, k_max_x: int, k_max_y: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_v = d_v
        self.k_max_x = k_max_x
        self.k_max_y = k_max_y
        
        # Spectral weights R (complex)
        scale = 1.0 / np.sqrt(d_v * d_v)
        self.R = (np.random.randn(k_max_x, k_max_y, d_v, d_v) + 
                  1j * np.random.randn(k_max_x, k_max_y, d_v, d_v)) * scale
        
        # Local path weights W (real)
        self.W = np.random.randn(d_v, d_v) * scale
        
        # Bias
        self.b = np.zeros(d_v)
        
        # Parameter count
        self.n_params = (2 * k_max_x * k_max_y * d_v * d_v +  # R (complex)
                         d_v * d_v +  # W
                         d_v)  # b
    
    def spectral_conv(self, v: np.ndarray) -> np.ndarray:
        """Spectral convolution: K(v)"""
        Nx, Ny, _ = v.shape
        
        # 2D FFT
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        
        # Initialize output spectrum
        w_hat = np.zeros((Nx, Ny//2+1, self.d_v), dtype=complex)
        
        # Multiply by R for low frequencies
        kx_max = min(self.k_max_x, Nx)
        ky_max = min(self.k_max_y, Ny//2+1)
        
        # Positive kx
        for kx in range(kx_max):
            for ky in range(ky_max):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ self.R[kx, ky, :, :]
        
        # Negative kx (wrap around)
        for kx in range(1, kx_max):
            for ky in range(ky_max):
                w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(self.R[kx, ky, :, :])
        
        # Inverse FFT
        w = np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))
        
        return w
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        """Complete forward pass: v^(l+1) = GELU(Wv + Kv + b)"""
        # Spectral path
        spectral_out = self.spectral_conv(v)
        
        # Local path
        local_out = v @ self.W
        
        # Combine and activate
        return gelu(spectral_out + local_out + self.b)
    
    def __repr__(self):
        return (f"FourierLayer(d_v={self.d_v}, k_max=({self.k_max_x}, {self.k_max_y}), "
                f"params={self.n_params:,})")


# Demonstrate Fourier layer
print("\n--- Fourier Layer Demo ---")

d_v = 64
k_max = 12

fl = FourierLayer(d_v, k_max, k_max, seed=42)
print(fl)

# Test with lifted output
v1 = fl.forward(v0)
print(f"\nInput shape:  {v0.shape}")
print(f"Output shape: {v1.shape}")
print(f"Shape preserved: {v0.shape == v1.shape}")


#=============================================================================
# SECTION 4: PROJECTION LAYER
#=============================================================================
print("\n" + "="*70)
print("SECTION 4: Projection Layer (Q)")
print("="*70)

class ProjectionLayer:
    """
    Projection Layer: Maps hidden representation to output.
    
    Two-layer MLP applied POINTWISE:
        h(x) = GELU(Q1 @ v(x) + b_Q1)
        u(x) = Q2 @ h(x) + b_Q2
    
    Parameters:
    -----------
    d_in : int
        Input dimension (hidden dimension from Fourier layers)
    d_mid : int
        Intermediate dimension
    d_out : int
        Output dimension (e.g., 1 for temperature)
    """
    
    def __init__(self, d_in: int, d_mid: int, d_out: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_in = d_in
        self.d_mid = d_mid
        self.d_out = d_out
        
        # First layer: d_in → d_mid
        scale1 = np.sqrt(2.0 / (d_in + d_mid))
        self.Q1 = np.random.randn(d_in, d_mid) * scale1
        self.b1 = np.zeros(d_mid)
        
        # Second layer: d_mid → d_out
        scale2 = np.sqrt(2.0 / (d_mid + d_out))
        self.Q2 = np.random.randn(d_mid, d_out) * scale2
        self.b2 = np.zeros(d_out)
        
        self.n_params = (d_in * d_mid + d_mid +  # First layer
                         d_mid * d_out + d_out)   # Second layer
    
    def forward(self, v: np.ndarray) -> np.ndarray:
        """
        Forward pass through projection.
        
        Parameters:
        -----------
        v : np.ndarray, shape (Nx, Ny, d_in)
            Hidden representation from Fourier layers
            
        Returns:
        --------
        u : np.ndarray, shape (Nx, Ny, d_out)
            Output (e.g., temperature field)
        """
        # First layer with GELU
        h = gelu(v @ self.Q1 + self.b1)
        
        # Second layer (NO activation - allows any output value)
        u = h @ self.Q2 + self.b2
        
        return u
    
    def __repr__(self):
        return (f"ProjectionLayer({self.d_in} → {self.d_mid} → {self.d_out}, "
                f"params={self.n_params:,})")


# Demonstrate projection layer
print("\n--- Projection Layer Demo ---")

d_in_proj = 64   # From Fourier layers
d_mid = 128      # Intermediate (expansion)
d_out_proj = 1   # Temperature

projection = ProjectionLayer(d_in_proj, d_mid, d_out_proj, seed=42)
print(projection)

# Test with Fourier layer output
u_sample = projection.forward(v1)

print(f"\nInput shape:  {v1.shape} = (Nx, Ny, d_v)")
print(f"Output shape: {u_sample.shape} = (Nx, Ny, d_out)")

# Verify no activation constrains output
print(f"\nOutput range: [{u_sample.min():.4f}, {u_sample.max():.4f}]")
print("(Can be any real value - no activation on final layer)")


#=============================================================================
# SECTION 5: COMPLETE FNO2D CLASS
#=============================================================================
print("\n" + "="*70)
print("SECTION 5: Complete FNO2d Architecture")
print("="*70)

class FNO2d:
    """
    Complete 2D Fourier Neural Operator.
    
    Architecture:
        Input a(x) → Lifting P → [Fourier Layer]×n_layers → Projection Q → Output u(x)
    
    Parameters:
    -----------
    d_a : int
        Input dimension (number of input features)
    d_v : int
        Hidden dimension for Fourier layers
    d_u : int
        Output dimension
    k_max : int
        Number of Fourier modes in each direction
    n_layers : int
        Number of Fourier layers
    d_mid : int, optional
        Intermediate dimension in projection (default: d_v * 2)
    """
    
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int, 
                 n_layers: int = 4, d_mid: Optional[int] = None, 
                 seed: Optional[int] = None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.k_max = k_max
        self.n_layers = n_layers
        self.d_mid = d_mid if d_mid is not None else d_v * 2
        
        # 1. Lifting layer
        self.lifting = LiftingLayer(d_a, d_v, seed=seed)
        
        # 2. Fourier layers
        self.fourier_layers = []
        for i in range(n_layers):
            layer_seed = seed + i + 1 if seed is not None else None
            self.fourier_layers.append(
                FourierLayer(d_v, k_max, k_max, seed=layer_seed)
            )
        
        # 3. Projection layer
        proj_seed = seed + n_layers + 1 if seed is not None else None
        self.projection = ProjectionLayer(d_v, self.d_mid, d_u, seed=proj_seed)
        
        # Total parameters
        self.n_params = (self.lifting.n_params + 
                         sum(fl.n_params for fl in self.fourier_layers) +
                         self.projection.n_params)
    
    def forward(self, a: np.ndarray, return_intermediates: bool = False) -> np.ndarray:
        """
        Complete forward pass through FNO.
        
        Parameters:
        -----------
        a : np.ndarray, shape (Nx, Ny, d_a)
            Input features
        return_intermediates : bool
            If True, return all intermediate activations
            
        Returns:
        --------
        u : np.ndarray, shape (Nx, Ny, d_u)
            Output prediction
        intermediates : list (only if return_intermediates=True)
            List of intermediate tensors
        """
        intermediates = []
        
        # 1. Lifting
        v = self.lifting.forward(a)
        if return_intermediates:
            intermediates.append(('After Lifting', v.copy()))
        
        # 2. Fourier layers
        for i, fl in enumerate(self.fourier_layers):
            v = fl.forward(v)
            if return_intermediates:
                intermediates.append((f'After FL{i+1}', v.copy()))
        
        # 3. Projection
        u = self.projection.forward(v)
        if return_intermediates:
            intermediates.append(('Output', u.copy()))
        
        if return_intermediates:
            return u, intermediates
        return u
    
    def __repr__(self):
        lines = [
            f"FNO2d(",
            f"  Input:      d_a = {self.d_a}",
            f"  Hidden:     d_v = {self.d_v}",
            f"  Output:     d_u = {self.d_u}",
            f"  Modes:      k_max = {self.k_max}",
            f"  Layers:     {self.n_layers}",
            f"  Projection: {self.d_v} → {self.d_mid} → {self.d_u}",
            f"  Total params: {self.n_params:,}",
            f")"
        ]
        return "\n".join(lines)
    
    def summary(self) -> Dict:
        """Return detailed parameter breakdown."""
        return {
            'lifting': self.lifting.n_params,
            'fourier_layers': [fl.n_params for fl in self.fourier_layers],
            'fourier_total': sum(fl.n_params for fl in self.fourier_layers),
            'projection': self.projection.n_params,
            'total': self.n_params
        }


# Create and demonstrate complete FNO
print("\n--- Complete FNO2d Demo ---")

# Configuration for your problem
config = {
    'd_a': 42,       # Your 42 input features
    'd_v': 32,       # Hidden dimension (conservative for limited data)
    'd_u': 1,        # Temperature output
    'k_max': 12,     # Fourier modes
    'n_layers': 4,   # Standard depth
    'd_mid': 64,     # Projection intermediate
}

fno = FNO2d(**config, seed=42)
print(fno)

# Parameter breakdown
print("\n--- Parameter Breakdown ---")
summary = fno.summary()
print(f"Lifting:        {summary['lifting']:>10,}")
print(f"Fourier Layers: {summary['fourier_total']:>10,}")
for i, p in enumerate(summary['fourier_layers']):
    print(f"  - Layer {i+1}:   {p:>10,}")
print(f"Projection:     {summary['projection']:>10,}")
print(f"{'─'*25}")
print(f"TOTAL:          {summary['total']:>10,}")

# Test forward pass
print("\n--- Forward Pass Test ---")
Nx, Ny = 64, 64
a_test = np.random.randn(Nx, Ny, config['d_a'])

start = time.time()
u_test, intermediates = fno.forward(a_test, return_intermediates=True)
elapsed = time.time() - start

print(f"Input shape:  {a_test.shape}")
print(f"Output shape: {u_test.shape}")
print(f"Forward time: {elapsed*1000:.2f} ms")

print("\n--- Shape Flow ---")
for name, tensor in intermediates:
    print(f"{name:20s}: {tensor.shape}")


#=============================================================================
# SECTION 6: LOSS FUNCTIONS
#=============================================================================
print("\n" + "="*70)
print("SECTION 6: Loss Functions")
print("="*70)

def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Squared Error loss.
    
    L = (1/n) Σ (pred - target)²
    """
    return np.mean((pred - target) ** 2)


def mae_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean Absolute Error loss.
    
    L = (1/n) Σ |pred - target|
    """
    return np.mean(np.abs(pred - target))


def relative_l2_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Relative L2 loss (FNO paper default).
    
    L = ||pred - target||_2 / ||target||_2
    
    This is scale-invariant: 10% error is 10% regardless of value magnitude.
    """
    return np.linalg.norm(pred - target) / np.linalg.norm(target)


def relative_l2_loss_batch(preds: List[np.ndarray], targets: List[np.ndarray]) -> float:
    """
    Relative L2 loss averaged over batch.
    """
    losses = [relative_l2_loss(p, t) for p, t in zip(preds, targets)]
    return np.mean(losses)


# Demonstrate loss functions
print("\n--- Loss Functions Demo ---")

# Create sample predictions and targets
np.random.seed(42)
target = np.random.randn(64, 64) * 10 + 300  # Temperature ~290-310 K
pred_good = target + np.random.randn(64, 64) * 1  # Small error
pred_bad = target + np.random.randn(64, 64) * 5   # Large error

print("Good prediction (small noise):")
print(f"  MSE:          {mse_loss(pred_good, target):.4f}")
print(f"  MAE:          {mae_loss(pred_good, target):.4f}")
print(f"  Relative L2:  {relative_l2_loss(pred_good, target):.4f} ({relative_l2_loss(pred_good, target)*100:.2f}%)")

print("\nBad prediction (large noise):")
print(f"  MSE:          {mse_loss(pred_bad, target):.4f}")
print(f"  MAE:          {mae_loss(pred_bad, target):.4f}")
print(f"  Relative L2:  {relative_l2_loss(pred_bad, target):.4f} ({relative_l2_loss(pred_bad, target)*100:.2f}%)")


#=============================================================================
# SECTION 7: DATA NORMALIZATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 7: Data Normalization")
print("="*70)

class DataNormalizer:
    """
    Handles normalization and denormalization for FNO training.
    
    Uses Z-score normalization: x_norm = (x - mean) / std
    """
    
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.is_fitted = False
    
    def fit(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Compute normalization statistics from training data.
        
        Parameters:
        -----------
        inputs : np.ndarray, shape (N, Nx, Ny, d_a)
            Training input samples
        outputs : np.ndarray, shape (N, Nx, Ny, d_u)
            Training output samples
        """
        # Compute per-channel statistics for inputs
        # Mean over samples and spatial dimensions
        self.input_mean = np.mean(inputs, axis=(0, 1, 2), keepdims=True)
        self.input_std = np.std(inputs, axis=(0, 1, 2), keepdims=True)
        self.input_std = np.maximum(self.input_std, 1e-8)  # Prevent division by zero
        
        # Statistics for outputs
        self.output_mean = np.mean(outputs)
        self.output_std = np.std(outputs)
        self.output_std = max(self.output_std, 1e-8)
        
        self.is_fitted = True
        
        print(f"Normalizer fitted:")
        print(f"  Input channels: {self.input_mean.shape[-1]}")
        print(f"  Input mean range: [{self.input_mean.min():.2f}, {self.input_mean.max():.2f}]")
        print(f"  Input std range: [{self.input_std.min():.2f}, {self.input_std.max():.2f}]")
        print(f"  Output mean: {self.output_mean:.2f}")
        print(f"  Output std: {self.output_std:.2f}")
    
    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Normalize input data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (x - self.input_mean) / self.input_std
    
    def normalize_output(self, y: np.ndarray) -> np.ndarray:
        """Normalize output data."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (y - self.output_mean) / self.output_std
    
    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized output back to original scale."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return y_norm * self.output_std + self.output_mean


# Demonstrate normalization
print("\n--- Normalization Demo ---")

# Simulate data like your problem
N_samples = 100
Nx, Ny = 32, 32
d_a = 42
d_u = 1

# Create fake training data
np.random.seed(42)
train_inputs = np.random.randn(N_samples, Nx, Ny, d_a) * np.random.rand(d_a) * 10  # Different scales
train_outputs = np.random.randn(N_samples, Nx, Ny, d_u) * 10 + 300  # Temperature ~290-310

print(f"Before normalization:")
print(f"  Input range: [{train_inputs.min():.2f}, {train_inputs.max():.2f}]")
print(f"  Output range: [{train_outputs.min():.2f}, {train_outputs.max():.2f}]")

# Fit normalizer
normalizer = DataNormalizer()
normalizer.fit(train_inputs, train_outputs)

# Normalize
inputs_norm = normalizer.normalize_input(train_inputs)
outputs_norm = normalizer.normalize_output(train_outputs)

print(f"\nAfter normalization:")
print(f"  Input range: [{inputs_norm.min():.2f}, {inputs_norm.max():.2f}]")
print(f"  Output range: [{outputs_norm.min():.2f}, {outputs_norm.max():.2f}]")
print(f"  Input mean ≈ 0: {np.abs(inputs_norm.mean()):.4f}")
print(f"  Input std ≈ 1: {inputs_norm.std():.4f}")


#=============================================================================
# SECTION 8: EVALUATION METRICS
#=============================================================================
print("\n" + "="*70)
print("SECTION 8: Evaluation Metrics")
print("="*70)

def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    
    R² = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


def compute_spatial_anomaly_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Spatial Anomaly R² - THE KEY METRIC for urban heat island!
    
    This tests if the model captures WHERE is hotter/cooler than average,
    not just the overall temperature level.
    
    1. Compute spatial anomaly: u_anomaly = u - mean(u)
    2. Compute R² on anomalies
    """
    # Compute spatial means (average over spatial dimensions)
    pred_mean = np.mean(pred)
    target_mean = np.mean(target)
    
    # Spatial anomalies
    pred_anomaly = pred - pred_mean
    target_anomaly = target - target_mean
    
    # R² on anomalies
    ss_res = np.sum((target_anomaly - pred_anomaly) ** 2)
    ss_tot = np.sum(target_anomaly ** 2)
    
    if ss_tot < 1e-10:
        return 0.0  # No variance in target
    
    return 1 - ss_res / ss_tot


def evaluate_predictions(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        'RMSE': compute_rmse(pred, target),
        'MAE': compute_mae(pred, target),
        'R2': compute_r2(pred, target),
        'Spatial_Anomaly_R2': compute_spatial_anomaly_r2(pred, target),
        'Relative_L2': relative_l2_loss(pred, target)
    }


# Demonstrate metrics
print("\n--- Evaluation Metrics Demo ---")

# Create target with spatial structure (like temperature field)
x = np.linspace(0, 2*np.pi, 64)
y = np.linspace(0, 2*np.pi, 64)
X, Y = np.meshgrid(x, y)

target = 300 + 5*np.sin(X) + 3*np.cos(Y) + 2*np.sin(2*X)*np.cos(2*Y)

# Good prediction (captures spatial pattern)
pred_good = target + np.random.randn(64, 64) * 0.5

# Bad prediction (wrong spatial pattern)
pred_bad_pattern = 300 + 5*np.cos(X) + 3*np.sin(Y) + np.random.randn(64, 64) * 0.5

# Prediction with right mean but no pattern
pred_no_pattern = np.ones_like(target) * np.mean(target) + np.random.randn(64, 64) * 0.5

print("Good prediction (captures pattern):")
metrics_good = evaluate_predictions(pred_good, target)
for name, value in metrics_good.items():
    print(f"  {name:20s}: {value:.4f}")

print("\nBad prediction (wrong pattern):")
metrics_bad = evaluate_predictions(pred_bad_pattern, target)
for name, value in metrics_bad.items():
    print(f"  {name:20s}: {value:.4f}")

print("\nNo pattern (just mean):")
metrics_none = evaluate_predictions(pred_no_pattern, target)
for name, value in metrics_none.items():
    print(f"  {name:20s}: {value:.4f}")

print("\n→ Spatial Anomaly R² distinguishes pattern capture from mean-only prediction!")


#=============================================================================
# SECTION 9: SYNTHETIC PDE DATA GENERATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 9: Synthetic PDE Data Generation")
print("="*70)

def generate_darcy_flow_data(n_samples: int, nx: int = 64, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Darcy flow data.
    
    Darcy flow: -∇·(a(x)∇u(x)) = f(x)
    
    We generate:
    - Input a(x): Random permeability field (positive)
    - Output u(x): Solution to Darcy equation (simplified)
    
    This is a simplified version - real Darcy requires PDE solver.
    We use a smoothed random field as input and a related smooth output.
    """
    np.random.seed(seed)
    
    inputs = []
    outputs = []
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x, y)
    
    for i in range(n_samples):
        # Generate random Fourier coefficients for input field
        n_modes = 5
        a = np.zeros((nx, nx))
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                amp = np.random.randn() / (kx + ky)
                phase = np.random.rand() * 2 * np.pi
                a += amp * np.sin(2*np.pi*kx*X + phase) * np.sin(2*np.pi*ky*Y + phase)
        
        # Make positive (permeability must be positive)
        a = np.exp(a)
        
        # Generate "solution" - in reality this would require solving PDE
        # We use a smoothed, related field
        u = np.zeros((nx, nx))
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                # Solution tends to be smoother than input
                amp = np.random.randn() / (kx + ky)**2
                phase = np.random.rand() * 2 * np.pi
                u += amp * np.sin(2*np.pi*kx*X + phase) * np.sin(2*np.pi*ky*Y + phase)
        
        # Add some correlation with input
        u = u + 0.3 * np.log(a + 1)
        
        inputs.append(a)
        outputs.append(u)
    
    inputs = np.array(inputs)[:, :, :, np.newaxis]   # Add channel dim
    outputs = np.array(outputs)[:, :, :, np.newaxis]
    
    return inputs, outputs


def generate_heat_equation_data(n_samples: int, nx: int = 64, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heat equation data.
    
    Input: Initial temperature + heat sources
    Output: Equilibrium temperature
    """
    np.random.seed(seed)
    
    inputs = []
    outputs = []
    
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, nx)
    X, Y = np.meshgrid(x, y)
    
    for i in range(n_samples):
        # Heat source locations (like urban heat islands)
        n_sources = np.random.randint(3, 8)
        source = np.zeros((nx, nx))
        for _ in range(n_sources):
            cx, cy = np.random.rand(2) * 2 * np.pi
            strength = np.random.randn() * 2
            sigma = 0.3 + np.random.rand() * 0.5
            source += strength * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # "Solution" - smoothed version (heat diffuses)
        # In reality, solve: ∇²u = -source
        # Simplified: apply smoothing (like low-pass filter)
        source_hat = np.fft.fft2(source)
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(nx) * nx
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1  # Avoid division by zero
        
        # Laplacian inverse (approximately)
        u_hat = source_hat / (K2 + 0.1)  # Regularized
        u = np.real(np.fft.ifft2(u_hat))
        
        inputs.append(source)
        outputs.append(u)
    
    inputs = np.array(inputs)[:, :, :, np.newaxis]
    outputs = np.array(outputs)[:, :, :, np.newaxis]
    
    return inputs, outputs


# Generate demo data
print("\n--- Generating Synthetic Data ---")

n_train = 200
n_test = 50
nx = 64

train_inputs, train_outputs = generate_heat_equation_data(n_train, nx, seed=42)
test_inputs, test_outputs = generate_heat_equation_data(n_test, nx, seed=123)

print(f"Training data: {train_inputs.shape} → {train_outputs.shape}")
print(f"Test data:     {test_inputs.shape} → {test_outputs.shape}")

# Visualize samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(4):
    axes[0, i].imshow(train_inputs[i, :, :, 0], cmap='hot', origin='lower')
    axes[0, i].set_title(f'Input (heat source) {i}')
    axes[0, i].axis('off')
    
    axes[1, i].imshow(train_outputs[i, :, :, 0], cmap='RdBu_r', origin='lower')
    axes[1, i].set_title(f'Output (temperature) {i}')
    axes[1, i].axis('off')

plt.suptitle('Synthetic Heat Equation Data: Input (sources) → Output (temperature)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/01_synthetic_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/01_synthetic_data.png")


#=============================================================================
# SECTION 10: TRAINING LOOP (Simplified - No Backprop)
#=============================================================================
print("\n" + "="*70)
print("SECTION 10: Training Framework (Conceptual)")
print("="*70)

"""
NOTE: Full training requires automatic differentiation (gradients).
NumPy doesn't provide this - you'd use PyTorch or JAX in practice.

This section demonstrates the STRUCTURE of training.
We'll show how predictions evolve and how loss would be computed.
"""

class FNOTrainer:
    """
    Training framework for FNO (conceptual).
    
    In practice, use PyTorch:
        model = FNO2d(...)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(n_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                pred = model(batch.input)
                loss = loss_fn(pred, batch.target)
                loss.backward()
                optimizer.step()
    """
    
    def __init__(self, model: FNO2d, normalizer: DataNormalizer, 
                 loss_fn: str = 'mse'):
        self.model = model
        self.normalizer = normalizer
        self.loss_fn = loss_fn
        
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute loss value."""
        if self.loss_fn == 'mse':
            return mse_loss(pred, target)
        elif self.loss_fn == 'relative_l2':
            return relative_l2_loss(pred, target)
        elif self.loss_fn == 'mae':
            return mae_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss: {self.loss_fn}")
    
    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        # Normalize inputs
        inputs_norm = self.normalizer.normalize_input(inputs)
        targets_norm = self.normalizer.normalize_output(targets)
        
        # Predict (batch)
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for i in range(len(inputs)):
            pred_norm = self.model.forward(inputs_norm[i])
            pred = self.normalizer.denormalize_output(pred_norm)
            
            all_preds.append(pred)
            all_targets.append(targets[i])
            total_loss += self.compute_loss(pred_norm, targets_norm[i])
        
        avg_loss = total_loss / len(inputs)
        
        # Concatenate for metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute all metrics
        metrics = {
            'loss': avg_loss,
            'rmse': compute_rmse(all_preds, all_targets),
            'mae': compute_mae(all_preds, all_targets),
            'r2': compute_r2(all_preds.flatten(), all_targets.flatten()),
        }
        
        # Spatial anomaly R² (average over samples)
        spatial_r2s = []
        for p, t in zip(all_preds, all_targets):
            spatial_r2s.append(compute_spatial_anomaly_r2(p[:,:,0], t[:,:,0]))
        metrics['spatial_anomaly_r2'] = np.mean(spatial_r2s)
        
        return metrics


print("""
Training Loop Structure (use PyTorch in practice):

1. INITIALIZATION:
   - Create FNO model
   - Create optimizer (Adam, lr=1e-3)
   - Create learning rate scheduler
   - Initialize normalizer from training data

2. TRAINING LOOP:
   for epoch in range(n_epochs):
       # Training phase
       model.train()
       for batch_inputs, batch_targets in train_loader:
           # Normalize
           inputs_norm = normalize(batch_inputs)
           targets_norm = normalize(batch_targets)
           
           # Forward
           predictions = model(inputs_norm)
           
           # Loss
           loss = loss_fn(predictions, targets_norm)
           
           # Backward (automatic differentiation)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       
       # Validation phase
       model.eval()
       val_loss = evaluate(val_data)
       
       # Learning rate schedule
       scheduler.step()
       
       # Early stopping check
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           save_checkpoint(model)
       elif patience_exceeded:
           break

3. FINAL EVALUATION:
   - Load best checkpoint
   - Evaluate on test set
   - Compute all metrics
""")


#=============================================================================
# SECTION 11: DEMONSTRATION - FORWARD PASS VISUALIZATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 11: Forward Pass Visualization")
print("="*70)

# Create model
fno_demo = FNO2d(
    d_a=1,        # Single input channel (heat source)
    d_v=32,       # Hidden dimension
    d_u=1,        # Single output (temperature)
    k_max=12,
    n_layers=4,
    d_mid=64,
    seed=42
)

print(fno_demo)

# Get one sample
sample_input = train_inputs[0]
sample_output = train_outputs[0]

# Forward pass with intermediates
pred, intermediates = fno_demo.forward(sample_input, return_intermediates=True)

# Visualize
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 6, figure=fig)

# Row 1: Input and target
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(sample_input[:, :, 0], cmap='hot', origin='lower')
ax1.set_title('Input\n(Heat Source)', fontsize=11)
plt.colorbar(im, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(sample_output[:, :, 0], cmap='RdBu_r', origin='lower')
ax2.set_title('Target\n(True Temperature)', fontsize=11)
plt.colorbar(im, ax=ax2, fraction=0.046)

ax3 = fig.add_subplot(gs[0, 2])
im = ax3.imshow(pred[:, :, 0], cmap='RdBu_r', origin='lower')
ax3.set_title('Prediction\n(FNO Output)', fontsize=11)
plt.colorbar(im, ax=ax3, fraction=0.046)

ax4 = fig.add_subplot(gs[0, 3])
error = pred[:, :, 0] - sample_output[:, :, 0]
im = ax4.imshow(error, cmap='RdBu_r', origin='lower')
ax4.set_title('Error\n(Pred - Target)', fontsize=11)
plt.colorbar(im, ax=ax4, fraction=0.046)

# Row 2: Intermediate representations (first 4 channels of each)
intermediates_to_show = [
    ('After Lifting', intermediates[0][1]),
    ('After FL1', intermediates[1][1]),
    ('After FL2', intermediates[2][1]),
    ('After FL3', intermediates[3][1]),
    ('After FL4', intermediates[4][1]),
    ('Output', intermediates[5][1])
]

for i, (name, tensor) in enumerate(intermediates_to_show):
    ax = fig.add_subplot(gs[1, i])
    if tensor.shape[-1] > 1:
        im = ax.imshow(tensor[:, :, 0], cmap='RdBu_r', origin='lower')
    else:
        im = ax.imshow(tensor[:, :, 0], cmap='RdBu_r', origin='lower')
    ax.set_title(f'{name}\nChannel 0', fontsize=10)
    ax.axis('off')

# Row 3: Spectra at each stage
for i, (name, tensor) in enumerate(intermediates_to_show):
    ax = fig.add_subplot(gs[2, i])
    spectrum = np.fft.rfft2(tensor[:, :, 0])
    log_spec = np.log10(np.abs(spectrum[:32, :]) + 1e-10)
    im = ax.imshow(log_spec, cmap='viridis', origin='lower', aspect='auto')
    ax.set_title(f'Spectrum', fontsize=10)
    ax.axhline(y=12, color='r', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.set_xlabel('ky')
    ax.set_ylabel('kx')

plt.suptitle('FNO Forward Pass: Input → Lifting → Fourier Layers → Projection → Output', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_forward_pass.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/02_forward_pass.png")


#=============================================================================
# SECTION 12: RESOLUTION INVARIANCE DEMONSTRATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 12: Resolution Invariance")
print("="*70)

def generate_smooth_field(nx: int, ny: int, seed: int = 42) -> np.ndarray:
    """Generate a smooth field at any resolution."""
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    field = np.sin(X) + 0.5*np.cos(2*Y) + 0.3*np.sin(X)*np.cos(Y)
    return field[:, :, np.newaxis]  # Add channel dim


# Create model with small k_max
fno_res = FNO2d(
    d_a=1, d_v=16, d_u=1, k_max=8, n_layers=2, d_mid=32, seed=42
)

print(f"Model: k_max=8, {fno_res.n_params:,} parameters")
print("Testing at different resolutions...\n")

resolutions = [32, 64, 128, 256]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, res in enumerate(resolutions):
    # Generate input at this resolution
    input_field = generate_smooth_field(res, res, seed=42)
    
    # Apply FNO (same weights!)
    output_field = fno_res.forward(input_field)
    
    # Show
    axes[0, i].imshow(input_field[:, :, 0], cmap='viridis', origin='lower')
    axes[0, i].set_title(f'Input {res}×{res}', fontsize=12)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(output_field[:, :, 0], cmap='RdBu_r', origin='lower')
    axes[1, i].set_title(f'Output {res}×{res}', fontsize=12)
    axes[1, i].axis('off')
    
    print(f"Resolution {res:3d}×{res:3d}: input {input_field.shape} → output {output_field.shape}")

plt.suptitle('Resolution Invariance: Same Model Weights at Different Grid Sizes', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_resolution_invariance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: figures/03_resolution_invariance.png")


#=============================================================================
# SECTION 13: URBAN TEMPERATURE MODEL TEMPLATE
#=============================================================================
print("\n" + "="*70)
print("SECTION 13: Urban Temperature FNO Template")
print("="*70)

def create_urban_temperature_fno(n_features: int = 42, 
                                  hidden_dim: int = 32,
                                  k_max: int = 12,
                                  n_layers: int = 4) -> FNO2d:
    """
    Create FNO configured for urban temperature prediction.
    
    Parameters matching your thesis problem:
    - 42 input features (NDVI, building height, ERA5, etc.)
    - 1 output (land surface temperature)
    - Conservative hidden dim for limited data (~230 samples)
    """
    return FNO2d(
        d_a=n_features,
        d_v=hidden_dim,
        d_u=1,
        k_max=k_max,
        n_layers=n_layers,
        d_mid=hidden_dim * 2,
        seed=42
    )


# Create models of different sizes
print("\n--- Model Size Comparison ---\n")

configs = [
    {'hidden_dim': 16, 'k_max': 8, 'n_layers': 2, 'name': 'Tiny'},
    {'hidden_dim': 32, 'k_max': 12, 'n_layers': 4, 'name': 'Small (Recommended)'},
    {'hidden_dim': 64, 'k_max': 12, 'n_layers': 4, 'name': 'Medium'},
    {'hidden_dim': 64, 'k_max': 16, 'n_layers': 4, 'name': 'Large'},
]

print(f"{'Config':<25} {'Params':>12} {'Notes'}")
print("-" * 60)

for cfg in configs:
    model = create_urban_temperature_fno(
        hidden_dim=cfg['hidden_dim'],
        k_max=cfg['k_max'],
        n_layers=cfg['n_layers']
    )
    notes = "← Start here" if cfg['name'] == 'Small (Recommended)' else ""
    print(f"{cfg['name']:<25} {model.n_params:>12,} {notes}")


#=============================================================================
# SECTION 14: COMPARISON WITH RANDOM FOREST (CONCEPTUAL)
#=============================================================================
print("\n" + "="*70)
print("SECTION 14: FNO vs Random Forest Comparison")
print("="*70)

print("""
Feature Comparison: FNO vs Your Current Random Forest

┌─────────────────────────────────────────────────────────────────────────┐
│ Aspect              │ Random Forest          │ FNO                      │
├─────────────────────┼───────────────────────┼──────────────────────────┤
│ Spatial Context     │ None (pixel-by-pixel) │ Global (entire field)    │
│ Feature Engineering │ Manual distances      │ Learned spectral         │
│ Park Cooling Effect │ Needs explicit feature│ Captured automatically   │
│ Urban-Rural Gradient│ Needs explicit feature│ Low-k modes capture it   │
│ Resolution          │ Fixed                 │ Invariant (train@64,eval@256)│
│ Parameters          │ ~100 trees × depth    │ ~500K-1M trainable       │
│ Training Data       │ 31M observations      │ 230 scenes               │
│ Output              │ Point predictions     │ Full spatial field       │
└─────────────────────┴───────────────────────┴──────────────────────────┘

Expected Improvements with FNO:
1. Spatial Anomaly R²: Currently 0.48-0.75 → Potentially 0.70-0.85
2. Better generalization to unseen spatial patterns
3. Physics-informed: Smooth outputs match thermal diffusion
4. Scale interactions: Automatically learns multi-scale relationships
""")


#=============================================================================
# SECTION 15: PYTORCH IMPLEMENTATION TEMPLATE
#=============================================================================
print("\n" + "="*70)
print("SECTION 15: PyTorch Implementation Template")
print("="*70)

pytorch_template = '''
"""
PyTorch FNO Implementation for Training
Copy this to use with real training!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), 
                            x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \\
            torch.einsum("bixy,ioxy->boxy", 
                        x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \\
            torch.einsum("bixy,ioxy->boxy", 
                        x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # IFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """Complete FNO for 2D spatial data"""
    
    def __init__(self, d_a, d_v, d_u, modes, n_layers=4):
        super().__init__()
        self.d_a = d_a
        self.d_v = d_v
        self.modes = modes
        self.n_layers = n_layers
        
        # Lifting
        self.fc0 = nn.Linear(d_a, d_v)
        
        # Fourier layers
        self.convs = nn.ModuleList([
            SpectralConv2d(d_v, d_v, modes, modes) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(d_v, d_v, 1) for _ in range(n_layers)
        ])
        
        # Projection
        self.fc1 = nn.Linear(d_v, d_v * 2)
        self.fc2 = nn.Linear(d_v * 2, d_u)
    
    def forward(self, x):
        # x: (batch, nx, ny, d_a)
        
        # Lifting
        x = self.fc0(x)  # (batch, nx, ny, d_v)
        x = x.permute(0, 3, 1, 2)  # (batch, d_v, nx, ny)
        
        # Fourier layers
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)
        
        # Projection
        x = x.permute(0, 2, 3, 1)  # (batch, nx, ny, d_v)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Training loop
def train_fno(model, train_loader, val_loader, n_epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += F.mse_loss(pred, y).item()
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: train={train_loss/len(train_loader):.4f}, "
                  f"val={val_loss/len(val_loader):.4f}")
    
    return model


# Usage:
# model = FNO2d(d_a=42, d_v=32, d_u=1, modes=12)
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
# val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8)
# trained_model = train_fno(model, train_loader, val_loader)
'''

print(pytorch_template)

# Save template to file
with open('pytorch_fno_template.py', 'w') as f:
    f.write(pytorch_template)
print("\n✓ Saved: pytorch_fno_template.py")


#=============================================================================
# SECTION 16: COMPLETE VISUALIZATION SUITE
#=============================================================================
print("\n" + "="*70)
print("SECTION 16: Architecture Visualization")
print("="*70)

# Create architecture diagram
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
colors = {
    'input': '#E8F5E9',
    'lifting': '#BBDEFB',
    'fourier': '#FFF3E0',
    'projection': '#F3E5F5',
    'output': '#FFEBEE',
    'arrow': '#455A64'
}

# Draw boxes
def draw_box(ax, x, y, w, h, color, text, fontsize=10):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

# Input
draw_box(ax, 0.5, 4, 2, 2, colors['input'], 'INPUT\na(x)\n(42 features)', 11)

# Lifting
draw_box(ax, 3.5, 4, 2, 2, colors['lifting'], 'LIFTING\nP\n42→64', 11)

# Fourier Layers
for i in range(4):
    draw_box(ax, 6.5 + i*1.5, 4, 1.3, 2, colors['fourier'], f'FL{i+1}', 10)

# Projection
draw_box(ax, 13, 4, 2, 2, colors['projection'], 'PROJECTION\nQ\n64→128→1', 11)

# Output
draw_box(ax, 15.5, 4, 1.5, 2, colors['output'], 'OUTPUT\nu(x)\n(temp)', 10)

# Arrows
arrow_style = dict(arrowstyle='->', color=colors['arrow'], lw=2)
ax.annotate('', xy=(3.4, 5), xytext=(2.6, 5), arrowprops=arrow_style)
ax.annotate('', xy=(6.4, 5), xytext=(5.6, 5), arrowprops=arrow_style)
for i in range(3):
    ax.annotate('', xy=(8 + i*1.5, 5), xytext=(7.9 + i*1.5, 5), arrowprops=arrow_style)
ax.annotate('', xy=(12.9, 5), xytext=(12.1, 5), arrowprops=arrow_style)
ax.annotate('', xy=(15.4, 5), xytext=(15.1, 5), arrowprops=arrow_style)

# Dimension labels
ax.text(3, 3.5, '(64,64,42)', ha='center', fontsize=9, style='italic')
ax.text(4.5, 3.5, '(64,64,64)', ha='center', fontsize=9, style='italic')
ax.text(8.75, 3.5, '(64,64,64)', ha='center', fontsize=9, style='italic')
ax.text(14, 3.5, '(64,64,1)', ha='center', fontsize=9, style='italic')

# Title and details
ax.text(8, 8.5, 'FNO2d Architecture for Urban Temperature Prediction', 
        ha='center', fontsize=16, fontweight='bold')
ax.text(8, 7.5, 'd_a=42, d_v=64, d_u=1, k_max=12, 4 layers', 
        ha='center', fontsize=12)

# Fourier Layer detail
ax.text(8.75, 2.5, 'Each Fourier Layer:', ha='center', fontsize=11, fontweight='bold')
ax.text(8.75, 2, 'v_out = GELU(W·v + FFT⁻¹[R·FFT[v]] + b)', ha='center', fontsize=10, family='monospace')

# Legend
legend_x = 1
ax.add_patch(plt.Rectangle((legend_x, 0.5), 0.3, 0.3, facecolor=colors['input'], edgecolor='black'))
ax.text(legend_x + 0.5, 0.65, 'Input', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((legend_x + 2, 0.5), 0.3, 0.3, facecolor=colors['lifting'], edgecolor='black'))
ax.text(legend_x + 2.5, 0.65, 'Lifting', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((legend_x + 4.5, 0.5), 0.3, 0.3, facecolor=colors['fourier'], edgecolor='black'))
ax.text(legend_x + 5, 0.65, 'Fourier Layer', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((legend_x + 7.5, 0.5), 0.3, 0.3, facecolor=colors['projection'], edgecolor='black'))
ax.text(legend_x + 8, 0.65, 'Projection', fontsize=9, va='center')
ax.add_patch(plt.Rectangle((legend_x + 10.5, 0.5), 0.3, 0.3, facecolor=colors['output'], edgecolor='black'))
ax.text(legend_x + 11, 0.65, 'Output', fontsize=9, va='center')

plt.savefig('figures/04_architecture_diagram.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: figures/04_architecture_diagram.png")


#=============================================================================
# SECTION 17: TRAINING METRICS SIMULATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 17: Simulated Training Curves")
print("="*70)

# Simulate realistic training curves
np.random.seed(42)
epochs = np.arange(300)

# Simulate training loss (decreasing with noise)
train_loss = 0.5 * np.exp(-epochs/50) + 0.02 + 0.01 * np.random.randn(300)
train_loss = np.maximum(train_loss, 0.01)

# Simulate validation loss (decreases then plateaus, slight increase for overfitting)
val_loss = 0.6 * np.exp(-epochs/60) + 0.03 + 0.015 * np.random.randn(300)
val_loss = np.maximum(val_loss, 0.02)
val_loss[200:] += 0.005 * (epochs[200:] - 200) / 100  # Slight overfitting

# Simulate learning rate (cosine annealing)
lr = 0.001 * (1 + np.cos(np.pi * epochs / 300)) / 2

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curves
axes[0].plot(epochs, train_loss, 'b-', alpha=0.7, label='Train Loss')
axes[0].plot(epochs, val_loss, 'r-', alpha=0.7, label='Val Loss')
axes[0].axhline(y=0.025, color='g', linestyle='--', label='Target', alpha=0.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Training Curves')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Learning rate
axes[1].plot(epochs, lr * 1000, 'g-')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate (×10⁻³)')
axes[1].set_title('Cosine Annealing LR Schedule')
axes[1].grid(True, alpha=0.3)

# Simulated R² improvement
r2 = 1 - val_loss / 0.6
r2 = np.clip(r2, 0, 0.98)
axes[2].plot(epochs, r2, 'purple')
axes[2].axhline(y=0.95, color='g', linestyle='--', label='Target R²=0.95', alpha=0.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Validation R²')
axes[2].set_title('R² Improvement')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0, 1])

plt.suptitle('Simulated FNO Training Progress', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/05_training_curves.png")


#=============================================================================
# SECTION 18: PARAMETER SCALING ANALYSIS
#=============================================================================
print("\n" + "="*70)
print("SECTION 18: Parameter Scaling Analysis")
print("="*70)

# Analyze how parameters scale with different choices
d_v_values = [16, 32, 64, 128]
k_max_values = [8, 12, 16, 20]

param_grid = np.zeros((len(d_v_values), len(k_max_values)))

for i, d_v in enumerate(d_v_values):
    for j, k_max in enumerate(k_max_values):
        model = FNO2d(d_a=42, d_v=d_v, d_u=1, k_max=k_max, n_layers=4, d_mid=d_v*2)
        param_grid[i, j] = model.n_params / 1e6  # In millions

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(param_grid, cmap='YlOrRd', aspect='auto')

# Labels
ax.set_xticks(range(len(k_max_values)))
ax.set_xticklabels(k_max_values)
ax.set_yticks(range(len(d_v_values)))
ax.set_yticklabels(d_v_values)
ax.set_xlabel('k_max (Fourier modes)', fontsize=12)
ax.set_ylabel('d_v (Hidden dimension)', fontsize=12)

# Add values
for i in range(len(d_v_values)):
    for j in range(len(k_max_values)):
        color = 'white' if param_grid[i, j] > 2 else 'black'
        ax.text(j, i, f'{param_grid[i, j]:.2f}M', ha='center', va='center', 
                fontsize=11, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Parameters (Millions)')
ax.set_title('FNO Parameter Count: d_v × k_max Scaling\n(4 layers, d_a=42)', fontsize=14)

# Highlight recommended region
rect = plt.Rectangle((-0.5, 0.5), 2.2, 1.2, fill=False, edgecolor='green', linewidth=3)
ax.add_patch(rect)
ax.text(0.5, 0.1, 'Recommended\nfor ~230 samples', ha='center', va='top', 
        fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/06_parameter_scaling.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/06_parameter_scaling.png")


#=============================================================================
# SUMMARY
#=============================================================================
print("\n" + "="*70)
print("CHUNK 3 COMPLETE: Full FNO Architecture Implemented")
print("="*70)

print("""
Files created:

Code:
  chunk3_code.py              - This comprehensive implementation
  pytorch_fno_template.py     - Ready-to-use PyTorch training code

Figures:
  figures/01_synthetic_data.png       - Synthetic heat equation data
  figures/02_forward_pass.png         - Complete forward pass visualization
  figures/03_resolution_invariance.png - Same weights at different resolutions
  figures/04_architecture_diagram.png  - FNO architecture diagram
  figures/05_training_curves.png      - Simulated training progress
  figures/06_parameter_scaling.png    - Parameter count analysis

Key Classes:
  LiftingLayer      - P: d_a → d_v (42 → 64)
  FourierLayer      - Spectral conv + local path + GELU
  ProjectionLayer   - Q: d_v → d_mid → d_u (64 → 128 → 1)
  FNO2d             - Complete architecture

Key Functions:
  gelu()                      - Activation function
  mse_loss(), mae_loss()      - Standard losses
  relative_l2_loss()          - FNO paper default
  compute_rmse/mae/r2()       - Evaluation metrics
  compute_spatial_anomaly_r2() - YOUR KEY METRIC

Recommended Configuration for Your Problem:
  d_a = 42          # Your input features
  d_v = 32          # Hidden dimension (conservative)
  d_u = 1           # Temperature output
  k_max = 12        # Fourier modes
  n_layers = 4      # Standard depth
  Parameters: ~530K (appropriate for 230 samples)

Next Steps:
  1. Convert your ECOSTRESS data to (N, Nx, Ny, 42) format
  2. Use pytorch_fno_template.py for real training
  3. Normalize data carefully!
  4. Start with small model, increase if underfitting
  5. Monitor spatial anomaly R² (not just raw R²)
""")
