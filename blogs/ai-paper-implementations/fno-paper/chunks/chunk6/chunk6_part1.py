"""
Fourier Neural Operator: Chunk 6 - Advanced Extensions
Part 1: Core Components and Temporal FNO Extensions

This implements:
1. Core FNO components (from previous chunks)
2. Autoregressive Temporal FNO
3. Direct Multi-Step Temporal FNO
4. 3D Space-Time FNO
5. Factorized Space-Time FNO
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*70)
print("CHUNK 6: ADVANCED EXTENSIONS - PART 1: TEMPORAL FNO")
print("="*70)

#=============================================================================
# SECTION 1: CORE FNO COMPONENTS
#=============================================================================

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class SpectralConv2d:
    """2D Spectral Convolution Layer."""
    def __init__(self, d_in, d_out, k_max_x, k_max_y, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d_in, self.d_out = d_in, d_out
        self.k_max_x, self.k_max_y = k_max_x, k_max_y
        scale = 1.0 / np.sqrt(d_in * d_out)
        self.R = (np.random.randn(k_max_x, k_max_y, d_in, d_out) + 
                  1j * np.random.randn(k_max_x, k_max_y, d_in, d_out)) * scale
    
    def forward(self, v):
        Nx, Ny, _ = v.shape
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        w_hat = np.zeros((Nx, Ny//2+1, self.d_out), dtype=complex)
        kx_max, ky_max = min(self.k_max_x, Nx), min(self.k_max_y, Ny//2+1)
        for kx in range(kx_max):
            for ky in range(ky_max):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ self.R[kx, ky, :, :]
        for kx in range(1, kx_max):
            for ky in range(ky_max):
                w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(self.R[kx, ky, :, :])
        return np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))

class FourierLayer:
    """Complete Fourier Layer with spectral + local paths."""
    def __init__(self, d_v, k_max, dropout=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d_v, self.k_max = d_v, k_max
        self.spectral_conv = SpectralConv2d(d_v, d_v, k_max, k_max, seed)
        scale = 1.0 / np.sqrt(d_v * d_v)
        self.W = np.random.randn(d_v, d_v) * scale
        self.b = np.zeros(d_v)
    
    def forward(self, v):
        return gelu(self.spectral_conv.forward(v) + v @ self.W + self.b)

class FNO2d:
    """Standard 2D FNO for single-time prediction."""
    def __init__(self, d_a, d_v, d_u, k_max, n_layers=4, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d_a, self.d_v, self.d_u, self.k_max = d_a, d_v, d_u, k_max
        self.n_layers = n_layers
        
        self.P = np.random.randn(d_a, d_v) * np.sqrt(2/(d_a+d_v))
        self.b_P = np.zeros(d_v)
        self.layers = [FourierLayer(d_v, k_max, seed=seed+i if seed else None) 
                       for i in range(n_layers)]
        d_mid = d_v * 2
        self.Q1 = np.random.randn(d_v, d_mid) * np.sqrt(2/(d_v+d_mid))
        self.b_Q1 = np.zeros(d_mid)
        self.Q2 = np.random.randn(d_mid, d_u) * np.sqrt(2/(d_mid+d_u))
        self.b_Q2 = np.zeros(d_u)
    
    def forward(self, a):
        v = a @ self.P + self.b_P
        for layer in self.layers:
            v = layer.forward(v)
        return gelu(v @ self.Q1 + self.b_Q1) @ self.Q2 + self.b_Q2

print("✓ Core FNO components loaded")

#=============================================================================
# SECTION 2: AUTOREGRESSIVE TEMPORAL FNO
#=============================================================================
print("\n" + "="*70)
print("SECTION 2: Autoregressive Temporal FNO")
print("="*70)

class AutoregressiveTemporalFNO:
    """
    Temporal FNO using autoregressive prediction.
    T(t+1) = FNO(T(t), T(t-1), features)
    T(t+2) = FNO(T(t+1), T(t), features)
    """
    
    def __init__(self, d_features: int, d_v: int = 32, k_max: int = 12,
                 n_layers: int = 4, n_history: int = 2, seed: int = 42):
        self.d_features = d_features
        self.n_history = n_history
        d_input = n_history + d_features
        self.fno = FNO2d(d_a=d_input, d_v=d_v, d_u=1, 
                        k_max=k_max, n_layers=n_layers, seed=seed)
        print(f"AutoregressiveTemporalFNO: {n_history} history + {d_features} features → 1 output")
    
    def predict_single_step(self, temp_history, features):
        x = np.concatenate([temp_history, features], axis=-1)
        return self.fno.forward(x)
    
    def predict_trajectory(self, temp_history, features, n_steps):
        Nx, Ny = temp_history.shape[:2]
        predictions = np.zeros((Nx, Ny, n_steps))
        history = temp_history.copy()
        
        for t in range(n_steps):
            pred = self.predict_single_step(history, features)
            predictions[:, :, t] = pred[:, :, 0]
            history = np.concatenate([history[:, :, 1:], pred], axis=-1)
        
        return predictions

# Demo
print("\n--- Demo: Autoregressive FNO ---")
np.random.seed(42)
Nx, Ny, d_features, n_history = 32, 32, 10, 2
temp_history = np.random.randn(Nx, Ny, n_history) * 5 + 290
features = np.random.randn(Nx, Ny, d_features)

ar_fno = AutoregressiveTemporalFNO(d_features=d_features, d_v=16, k_max=8, n_layers=2)
trajectory = ar_fno.predict_trajectory(temp_history, features, n_steps=5)
print(f"Trajectory shape: {trajectory.shape} (5 future timesteps)")

#=============================================================================
# SECTION 3: DIRECT MULTI-STEP TEMPORAL FNO
#=============================================================================
print("\n" + "="*70)
print("SECTION 3: Direct Multi-Step Temporal FNO")
print("="*70)

class DirectMultiStepFNO:
    """Predicts all future timesteps simultaneously."""
    
    def __init__(self, d_features: int, d_v: int = 32, k_max: int = 12,
                 n_layers: int = 4, n_history: int = 2, horizon: int = 4, seed: int = 42):
        self.horizon = horizon
        d_input = n_history + d_features
        self.fno = FNO2d(d_a=d_input, d_v=d_v, d_u=horizon, k_max=k_max, n_layers=n_layers, seed=seed)
        print(f"DirectMultiStepFNO: {d_input} inputs → {horizon} timesteps simultaneously")
    
    def predict(self, temp_history, features):
        x = np.concatenate([temp_history, features], axis=-1)
        return self.fno.forward(x)

# Demo
print("\n--- Demo: Direct Multi-Step FNO ---")
direct_fno = DirectMultiStepFNO(d_features=d_features, d_v=16, k_max=8, n_layers=2, horizon=4)
direct_pred = direct_fno.predict(temp_history, features)
print(f"Direct prediction shape: {direct_pred.shape} (all 4 steps at once)")

#=============================================================================
# SECTION 4: 3D SPACE-TIME FNO
#=============================================================================
print("\n" + "="*70)
print("SECTION 4: 3D Space-Time FNO")
print("="*70)

class SpectralConv3d:
    """3D Spectral Convolution for joint space-time processing."""
    
    def __init__(self, d_in, d_out, k_max_x, k_max_y, omega_max, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d_in, self.d_out = d_in, d_out
        self.k_max_x, self.k_max_y, self.omega_max = k_max_x, k_max_y, omega_max
        scale = 1.0 / np.sqrt(d_in * d_out)
        self.R = (np.random.randn(k_max_x, k_max_y, omega_max, d_in, d_out) + 
                  1j * np.random.randn(k_max_x, k_max_y, omega_max, d_in, d_out)) * scale
        self.n_params = 2 * k_max_x * k_max_y * omega_max * d_in * d_out
    
    def forward(self, v):
        Nx, Ny, Nt, _ = v.shape
        v_hat = np.fft.rfftn(v, axes=(0, 1, 2))
        w_hat = np.zeros((Nx, Ny//2+1, Nt, self.d_out), dtype=complex)
        
        kx_max = min(self.k_max_x, Nx)
        ky_max = min(self.k_max_y, Ny//2+1)
        om_max = min(self.omega_max, Nt)
        
        for kx in range(kx_max):
            for ky in range(ky_max):
                for omega in range(om_max):
                    w_hat[kx, ky, omega, :] = v_hat[kx, ky, omega, :] @ self.R[kx, ky, omega, :, :]
        
        return np.fft.irfftn(w_hat, s=(Nx, Ny, Nt), axes=(0, 1, 2))

class SpaceTimeFNO:
    """3D Space-Time FNO treating time as third dimension."""
    
    def __init__(self, d_a, d_v=32, d_u=1, k_max=8, omega_max=4, n_layers=4, seed=42):
        if seed is not None:
            np.random.seed(seed)
        self.d_a, self.d_v, self.d_u = d_a, d_v, d_u
        
        self.P = np.random.randn(d_a, d_v) * np.sqrt(2/(d_a+d_v))
        self.b_P = np.zeros(d_v)
        
        self.spectral_convs = []
        self.Ws, self.bs = [], []
        for i in range(n_layers):
            self.spectral_convs.append(SpectralConv3d(d_v, d_v, k_max, k_max, omega_max, seed+i))
            self.Ws.append(np.random.randn(d_v, d_v) / np.sqrt(d_v*d_v))
            self.bs.append(np.zeros(d_v))
        
        d_mid = d_v * 2
        self.Q1 = np.random.randn(d_v, d_mid) * np.sqrt(2/(d_v+d_mid))
        self.b_Q1 = np.zeros(d_mid)
        self.Q2 = np.random.randn(d_mid, d_u) * np.sqrt(2/(d_mid+d_u))
        self.b_Q2 = np.zeros(d_u)
        
        print(f"SpaceTimeFNO: k_max={k_max}, omega_max={omega_max}")
    
    def forward(self, a):
        v = a @ self.P + self.b_P
        for sc, W, b in zip(self.spectral_convs, self.Ws, self.bs):
            v = gelu(sc.forward(v) + v @ W + b)
        return gelu(v @ self.Q1 + self.b_Q1) @ self.Q2 + self.b_Q2

# Demo
print("\n--- Demo: 3D Space-Time FNO ---")
Nx, Ny, Nt, d_a = 16, 16, 8, 5
spacetime_input = np.random.randn(Nx, Ny, Nt, d_a)
st_fno = SpaceTimeFNO(d_a=d_a, d_v=16, d_u=1, k_max=4, omega_max=3, n_layers=2)
st_output = st_fno.forward(spacetime_input)
print(f"Space-time input: {spacetime_input.shape} → output: {st_output.shape}")

#=============================================================================
# SECTION 5: FACTORIZED SPACE-TIME FNO
#=============================================================================
print("\n" + "="*70)
print("SECTION 5: Factorized Space-Time FNO")
print("="*70)

class TemporalAttention:
    """Simple temporal attention mechanism."""
    
    def __init__(self, d_v, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d_v = d_v
        scale = 1.0 / np.sqrt(d_v)
        self.W_q = np.random.randn(d_v, d_v) * scale
        self.W_k = np.random.randn(d_v, d_v) * scale
        self.W_v = np.random.randn(d_v, d_v) * scale
        self.W_o = np.random.randn(d_v, d_v) * scale
    
    def forward(self, v):
        Nx, Ny, Nt, d_v = v.shape
        v_flat = v.reshape(Nx*Ny, Nt, d_v)
        
        Q = v_flat @ self.W_q
        K = v_flat @ self.W_k
        V = v_flat @ self.W_v
        
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_v)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        
        out = (attn @ V) @ self.W_o
        return out.reshape(Nx, Ny, Nt, d_v)

class FactorizedSpaceTimeFNO:
    """Factorized: Spatial FNO + Temporal Attention (FourCastNet-style)."""
    
    def __init__(self, d_a, d_v=32, d_u=1, k_max=8, n_layers=4, seed=42):
        if seed is not None:
            np.random.seed(seed)
        self.d_a, self.d_v, self.d_u = d_a, d_v, d_u
        
        self.P = np.random.randn(d_a, d_v) * np.sqrt(2/(d_a+d_v))
        self.b_P = np.zeros(d_v)
        
        self.spatial_convs = []
        self.temporal_attns = []
        self.Ws, self.bs = [], []
        
        for i in range(n_layers):
            self.spatial_convs.append(SpectralConv2d(d_v, d_v, k_max, k_max, seed+i))
            self.temporal_attns.append(TemporalAttention(d_v, seed=seed+i+100))
            self.Ws.append(np.random.randn(d_v, d_v) / np.sqrt(d_v*d_v))
            self.bs.append(np.zeros(d_v))
        
        d_mid = d_v * 2
        self.Q1 = np.random.randn(d_v, d_mid) * np.sqrt(2/(d_v+d_mid))
        self.b_Q1 = np.zeros(d_mid)
        self.Q2 = np.random.randn(d_mid, d_u) * np.sqrt(2/(d_mid+d_u))
        self.b_Q2 = np.zeros(d_u)
        
        print(f"FactorizedSpaceTimeFNO: Spatial FFT + Temporal Attention")
    
    def forward(self, a):
        Nx, Ny, Nt, _ = a.shape
        v = a @ self.P + self.b_P
        
        for sc, ta, W, b in zip(self.spatial_convs, self.temporal_attns, self.Ws, self.bs):
            v_spatial = np.zeros_like(v)
            for t in range(Nt):
                v_spatial[:, :, t, :] = sc.forward(v[:, :, t, :])
            v_temporal = ta.forward(v)
            v = gelu(v_spatial + v_temporal + v @ W + b)
        
        return gelu(v @ self.Q1 + self.b_Q1) @ self.Q2 + self.b_Q2

# Demo
print("\n--- Demo: Factorized Space-Time FNO ---")
fact_fno = FactorizedSpaceTimeFNO(d_a=d_a, d_v=16, d_u=1, k_max=4, n_layers=2)
fact_output = fact_fno.forward(spacetime_input)
print(f"Factorized output: {fact_output.shape}")

#=============================================================================
# COMPARISON TABLE
#=============================================================================
print("\n" + "="*70)
print("COMPARISON: Temporal FNO Approaches")
print("="*70)
print("""
┌─────────────────────┬────────────────┬──────────────┬─────────────────┐
│ Approach            │ Error Accum.   │ Horizon      │ Best For        │
├─────────────────────┼────────────────┼──────────────┼─────────────────┤
│ Autoregressive      │ Yes            │ Any          │ Variable gaps   │
│ Direct Multi-Step   │ No             │ Fixed (H)    │ Speed critical  │
│ 3D Space-Time       │ No             │ Fixed (Nt)   │ Periodic data   │
│ Factorized          │ No             │ Fixed (Nt)   │ Long sequences  │
└─────────────────────┴────────────────┴──────────────┴─────────────────┘
""")

print("\n✓ Part 1 complete - Temporal FNO Extensions")
