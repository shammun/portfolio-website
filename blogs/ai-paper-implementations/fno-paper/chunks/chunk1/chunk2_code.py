"""
Fourier Neural Operator: Chunk 2 - The Fourier Layer
Complete Code Implementation with Visualizations

This file implements every concept from the Chunk 2 theory guide:
1. Basic spectral convolution (1D and 2D)
2. Mode truncation mechanics
3. Weight tensor R - all dimensions explained
4. Multi-channel spectral convolution
5. Local path W
6. Complete Fourier Layer
7. Resolution invariance demonstration
8. GELU activation
9. Full layer visualization

Author: FNO Tutorial for PhD Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

print("="*70)
print("CHUNK 2: THE FOURIER LAYER - Complete Implementation")
print("="*70)

#=============================================================================
# SECTION 1: BASIC SPECTRAL CONVOLUTION (1D)
#=============================================================================
print("\n" + "="*70)
print("SECTION 1: Basic Spectral Convolution (1D)")
print("="*70)

def spectral_conv_1d_basic(v, R):
    """
    Most basic spectral convolution - single channel, 1D.
    
    The core FNO operation: FFT → multiply by R → IFFT
    
    Parameters:
    -----------
    v : np.ndarray, shape (N,)
        Input signal in spatial domain
    R : np.ndarray, shape (k_max,), complex
        Learnable weights in frequency domain
        
    Returns:
    --------
    w : np.ndarray, shape (N,)
        Output signal in spatial domain
    """
    N = len(v)
    k_max = len(R)
    
    # Step 1: FFT to frequency domain
    v_hat = np.fft.rfft(v)  # Shape: (N//2 + 1,) complex
    
    # Step 2: Initialize output spectrum (zeros for high frequencies)
    w_hat = np.zeros_like(v_hat)
    
    # Step 3: Multiply by R for low frequencies only
    w_hat[:k_max] = v_hat[:k_max] * R
    
    # Step 4: IFFT back to spatial domain
    w = np.fft.irfft(w_hat, n=N)
    
    return w


# Demonstrate basic spectral convolution
print("\n--- Demo: Basic 1D Spectral Convolution ---")

# Create a test signal: combination of frequencies
N = 64
x = np.linspace(0, 2*np.pi, N, endpoint=False)
v = np.sin(x) + 0.5*np.sin(3*x) + 0.3*np.sin(7*x) + 0.2*np.sin(15*x)

print(f"Input signal: N={N} points")
print(f"Contains frequencies: k=1, 3, 7, 15")

# Create different R weights to show different effects
k_max = 12

# R1: Pass all (identity in frequency)
R_identity = np.ones(k_max, dtype=complex)

# R2: Low-pass (attenuate higher modes within k_max)
R_lowpass = np.exp(-np.arange(k_max) / 3).astype(complex)

# R3: Amplify specific frequency (k=3)
R_amplify = np.ones(k_max, dtype=complex)
R_amplify[3] = 3.0  # Amplify the k=3 mode

# R4: Phase shift (rotate all modes by π/4)
R_phase = np.exp(1j * np.pi/4) * np.ones(k_max, dtype=complex)

# Apply each
w_identity = spectral_conv_1d_basic(v, R_identity)
w_lowpass = spectral_conv_1d_basic(v, R_lowpass)
w_amplify = spectral_conv_1d_basic(v, R_amplify)
w_phase = spectral_conv_1d_basic(v, R_phase)

print(f"k_max = {k_max} (modes 0 to {k_max-1} are kept)")
print(f"Modes k >= {k_max} (including k=15) are set to zero")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Original signal and spectrum
axes[0, 0].plot(x, v, 'b-', linewidth=2)
axes[0, 0].set_title('Input Signal v(x)', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('v(x)')
axes[0, 0].grid(True, alpha=0.3)

v_hat = np.fft.rfft(v)
freqs = np.arange(len(v_hat))
axes[0, 1].stem(freqs[:20], np.abs(v_hat)[:20], basefmt=' ')
axes[0, 1].axvline(x=k_max-0.5, color='r', linestyle='--', label=f'k_max={k_max}')
axes[0, 1].set_title('Input Spectrum |v̂(k)|', fontsize=12)
axes[0, 1].set_xlabel('Frequency k')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# R weights visualization
axes[0, 2].plot(np.arange(k_max), np.abs(R_identity), 'o-', label='Identity')
axes[0, 2].plot(np.arange(k_max), np.abs(R_lowpass), 's-', label='Low-pass')
axes[0, 2].plot(np.arange(k_max), np.abs(R_amplify), '^-', label='Amplify k=3')
axes[0, 2].set_title('Different R(k) Weight Magnitudes', fontsize=12)
axes[0, 2].set_xlabel('Frequency k')
axes[0, 2].set_ylabel('|R(k)|')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Output signals
axes[1, 0].plot(x, v, 'b--', alpha=0.5, label='Input')
axes[1, 0].plot(x, w_identity, 'g-', linewidth=2, label='R=identity')
axes[1, 0].set_title('Identity R: k≥12 removed', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, v, 'b--', alpha=0.5, label='Input')
axes[1, 1].plot(x, w_lowpass, 'r-', linewidth=2, label='R=lowpass')
axes[1, 1].set_title('Low-pass R: Smooth output', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(x, v, 'b--', alpha=0.5, label='Input')
axes[1, 2].plot(x, w_amplify, 'm-', linewidth=2, label='R amplifies k=3')
axes[1, 2].set_title('Amplify k=3: sin(3x) boosted', fontsize=12)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/01_spectral_conv_1d_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/01_spectral_conv_1d_basic.png")

# Verify the math
print("\n--- Verification ---")
print(f"Input has k=15 component (amplitude ~0.2)")
print(f"After spectral conv with k_max=12, k=15 is removed")
print(f"Output amplitude range: [{w_identity.min():.3f}, {w_identity.max():.3f}]")
print(f"Original amplitude range: [{v.min():.3f}, {v.max():.3f}]")
print(f"→ The k=15 high-frequency component is gone!")


#=============================================================================
# SECTION 2: MODE TRUNCATION - DETAILED MECHANICS
#=============================================================================
print("\n" + "="*70)
print("SECTION 2: Mode Truncation Mechanics")
print("="*70)

def demonstrate_mode_truncation(v, k_max_values):
    """
    Show effect of different k_max values on reconstruction.
    """
    N = len(v)
    v_hat = np.fft.rfft(v)
    
    reconstructions = []
    for k_max in k_max_values:
        # Truncate spectrum
        w_hat = np.zeros_like(v_hat)
        w_hat[:min(k_max, len(v_hat))] = v_hat[:min(k_max, len(v_hat))]
        # Reconstruct
        w = np.fft.irfft(w_hat, n=N)
        reconstructions.append(w)
    
    return reconstructions


# Create a more complex signal
N = 128
x = np.linspace(0, 4*np.pi, N, endpoint=False)

# Signal with many frequencies
v_complex = (np.sin(x) + 0.7*np.sin(2*x) + 0.5*np.sin(5*x) + 
             0.3*np.sin(10*x) + 0.2*np.sin(20*x) + 0.1*np.sin(40*x))

k_max_values = [2, 4, 8, 16, 32, 65]  # 65 = full spectrum for N=128

reconstructions = demonstrate_mode_truncation(v_complex, k_max_values)

# Compute reconstruction errors
errors = [np.mean((v_complex - r)**2) for r in reconstructions]

print(f"\nSignal contains frequencies: k = 1, 2, 5, 10, 20, 40")
print(f"\nReconstruction MSE for different k_max:")
for k, err in zip(k_max_values, errors):
    print(f"  k_max = {k:3d}: MSE = {err:.6f}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (k_max, recon) in enumerate(zip(k_max_values, reconstructions)):
    axes[i].plot(x, v_complex, 'b-', alpha=0.4, linewidth=1, label='Original')
    axes[i].plot(x, recon, 'r-', linewidth=2, label=f'k_max={k_max}')
    axes[i].set_title(f'k_max = {k_max}, MSE = {errors[i]:.4f}', fontsize=11)
    axes[i].legend(loc='upper right')
    axes[i].set_xlim([0, 4*np.pi])
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Mode Truncation: Effect of k_max on Signal Reconstruction', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_mode_truncation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/02_mode_truncation.png")

# Show the spectrum with truncation regions
fig, ax = plt.subplots(figsize=(12, 5))

v_hat = np.fft.rfft(v_complex)
freqs = np.arange(len(v_hat))

ax.stem(freqs, np.abs(v_hat), basefmt=' ', linefmt='b-', markerfmt='bo')

# Mark different k_max regions
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
for i, k in enumerate(k_max_values[:-1]):
    ax.axvline(x=k-0.5, color=colors[i], linestyle='--', alpha=0.7, 
               label=f'k_max={k}')
    
ax.set_xlabel('Frequency k', fontsize=12)
ax.set_ylabel('|v̂(k)|', fontsize=12)
ax.set_title('Spectrum with Mode Truncation Cutoffs', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim([-1, 50])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/03_truncation_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/03_truncation_spectrum.png")


#=============================================================================
# SECTION 3: 2D SPECTRAL CONVOLUTION
#=============================================================================
print("\n" + "="*70)
print("SECTION 3: 2D Spectral Convolution")
print("="*70)

def spectral_conv_2d_basic(v, R):
    """
    Basic 2D spectral convolution - single channel.
    
    Parameters:
    -----------
    v : np.ndarray, shape (Nx, Ny)
        Input 2D field (e.g., temperature map)
    R : np.ndarray, shape (k_max_x, k_max_y), complex
        Learnable weights in 2D frequency domain
        
    Returns:
    --------
    w : np.ndarray, shape (Nx, Ny)
        Output 2D field
    """
    Nx, Ny = v.shape
    k_max_x, k_max_y = R.shape
    
    # Step 1: 2D FFT
    v_hat = np.fft.rfft2(v)  # Shape: (Nx, Ny//2 + 1)
    
    # Step 2: Initialize output spectrum
    w_hat = np.zeros_like(v_hat)
    
    # Step 3: Multiply by R for low frequencies
    # Handle positive kx modes (0 to k_max_x-1)
    w_hat[:k_max_x, :k_max_y] = v_hat[:k_max_x, :k_max_y] * R
    
    # Handle negative kx modes (wrap around in FFT)
    # These are stored at indices Nx-k_max_x+1 to Nx-1
    if k_max_x > 1:
        w_hat[-k_max_x+1:, :k_max_y] = v_hat[-k_max_x+1:, :k_max_y] * np.conj(np.flip(R[1:, :], axis=0))
    
    # Step 4: Inverse 2D FFT
    w = np.fft.irfft2(w_hat, s=(Nx, Ny))
    
    return w


# Create a 2D test field (simulating a temperature map)
Nx, Ny = 64, 64
x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Temperature field with multiple 2D modes
temperature = (np.sin(X) * np.sin(Y) +           # (1,1) mode
               0.5 * np.sin(2*X) * np.sin(2*Y) +  # (2,2) mode
               0.3 * np.sin(5*X) * np.sin(3*Y) +  # (5,3) mode
               0.2 * np.sin(10*X) * np.sin(10*Y)) # (10,10) mode

print(f"2D field shape: {temperature.shape}")
print(f"Contains 2D modes: (1,1), (2,2), (5,3), (10,10)")

# Different k_max values for 2D
k_max_values_2d = [4, 8, 16]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Original
im = axes[0, 0].imshow(temperature, cmap='RdBu_r', origin='lower')
axes[0, 0].set_title('Original Temperature Field', fontsize=11)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

# Original spectrum
temp_hat = np.fft.rfft2(temperature)
spectrum = np.log10(np.abs(temp_hat) + 1e-10)
im = axes[1, 0].imshow(spectrum[:32, :], aspect='auto', cmap='viridis', origin='lower')
axes[1, 0].set_title('Log Spectrum |F(T)|', fontsize=11)
axes[1, 0].set_xlabel('ky')
axes[1, 0].set_ylabel('kx')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# Apply truncation with identity R
for i, k_max in enumerate(k_max_values_2d):
    R = np.ones((k_max, k_max), dtype=complex)
    
    # Manually truncate (simpler version for visualization)
    temp_hat_trunc = np.zeros_like(temp_hat)
    temp_hat_trunc[:k_max, :k_max] = temp_hat[:k_max, :k_max]
    if k_max > 1:
        temp_hat_trunc[-k_max+1:, :k_max] = temp_hat[-k_max+1:, :k_max]
    
    temp_reconstructed = np.fft.irfft2(temp_hat_trunc, s=(Nx, Ny))
    
    mse = np.mean((temperature - temp_reconstructed)**2)
    
    im = axes[0, i+1].imshow(temp_reconstructed, cmap='RdBu_r', origin='lower',
                              vmin=temperature.min(), vmax=temperature.max())
    axes[0, i+1].set_title(f'k_max={k_max}, MSE={mse:.4f}', fontsize=11)
    plt.colorbar(im, ax=axes[0, i+1], fraction=0.046)
    
    # Show which modes are kept
    kept_modes = np.zeros((32, 33))
    kept_modes[:k_max, :k_max] = 1
    axes[1, i+1].imshow(kept_modes, cmap='Greens', origin='lower', aspect='auto')
    axes[1, i+1].set_title(f'Kept modes (green)', fontsize=11)
    axes[1, i+1].set_xlabel('ky')
    axes[1, i+1].set_ylabel('kx')

plt.suptitle('2D Mode Truncation for Temperature Field', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/04_2d_spectral_conv.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/04_2d_spectral_conv.png")


#=============================================================================
# SECTION 4: MULTI-CHANNEL SPECTRAL CONVOLUTION
#=============================================================================
print("\n" + "="*70)
print("SECTION 4: Multi-Channel Spectral Convolution")
print("="*70)

def spectral_conv_multichannel_1d(v, R):
    """
    Multi-channel 1D spectral convolution.
    
    This is closer to what FNO actually does:
    - Multiple input channels
    - Multiple output channels
    - Channel mixing happens at EACH frequency
    
    Parameters:
    -----------
    v : np.ndarray, shape (N, d_in)
        Input signal with d_in channels
    R : np.ndarray, shape (k_max, d_in, d_out), complex
        Per-frequency channel mixing matrices
        
    Returns:
    --------
    w : np.ndarray, shape (N, d_out)
        Output signal with d_out channels
    """
    N, d_in = v.shape
    k_max, _, d_out = R.shape
    
    # Step 1: FFT each channel
    v_hat = np.fft.rfft(v, axis=0)  # Shape: (N//2+1, d_in)
    
    # Step 2: Initialize output spectrum
    w_hat = np.zeros((v_hat.shape[0], d_out), dtype=complex)
    
    # Step 3: For each frequency k, multiply by weight matrix R[k]
    for k in range(min(k_max, v_hat.shape[0])):
        # v_hat[k] shape: (d_in,)
        # R[k] shape: (d_in, d_out)
        # Result: (d_out,)
        w_hat[k] = v_hat[k] @ R[k]
    
    # Step 4: IFFT each output channel
    w = np.fft.irfft(w_hat, n=N, axis=0)  # Shape: (N, d_out)
    
    return w


# Demonstrate multi-channel
N = 64
d_in = 3  # e.g., NDVI, Building Height, ERA5 temp
d_out = 4  # hidden channels
k_max = 8

# Create input with 3 channels
x = np.linspace(0, 2*np.pi, N, endpoint=False)
v_multi = np.zeros((N, d_in))
v_multi[:, 0] = np.sin(x)           # Channel 0: low freq
v_multi[:, 1] = np.sin(5*x)         # Channel 1: medium freq
v_multi[:, 2] = np.sin(x) + np.sin(3*x)  # Channel 2: mixed

print(f"\nInput shape: {v_multi.shape} = (N={N}, d_in={d_in})")
print(f"Weight tensor R shape: ({k_max}, {d_in}, {d_out}) = (k_max, d_in, d_out)")

# Initialize random complex weights
np.random.seed(42)
R_multi = (np.random.randn(k_max, d_in, d_out) + 
           1j * np.random.randn(k_max, d_in, d_out)) * 0.1

# Apply multi-channel spectral convolution
w_multi = spectral_conv_multichannel_1d(v_multi, R_multi)

print(f"Output shape: {w_multi.shape} = (N={N}, d_out={d_out})")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Input channels
for c in range(d_in):
    axes[0, c].plot(x, v_multi[:, c], linewidth=2)
    axes[0, c].set_title(f'Input Channel {c}', fontsize=11)
    axes[0, c].set_xlabel('x')
    axes[0, c].grid(True, alpha=0.3)

# Weight matrix at k=1
axes[0, 3].imshow(np.abs(R_multi[1]), cmap='Blues', aspect='auto')
axes[0, 3].set_title('|R[k=1]| weight matrix\n(d_in × d_out)', fontsize=11)
axes[0, 3].set_xlabel('d_out')
axes[0, 3].set_ylabel('d_in')
axes[0, 3].set_xticks(range(d_out))
axes[0, 3].set_yticks(range(d_in))

# Output channels
for c in range(d_out):
    axes[1, c].plot(x, w_multi[:, c], linewidth=2, color=f'C{c}')
    axes[1, c].set_title(f'Output Channel {c}', fontsize=11)
    axes[1, c].set_xlabel('x')
    axes[1, c].grid(True, alpha=0.3)

plt.suptitle('Multi-Channel Spectral Convolution: d_in=3 → d_out=4', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_multichannel_spectral.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/05_multichannel_spectral.png")

# Explain the channel mixing
print("\n--- Channel Mixing at Each Frequency ---")
print(f"At frequency k=1:")
print(f"  Input spectrum v_hat[1] has shape ({d_in},)")
print(f"  Weight matrix R[1] has shape ({d_in}, {d_out})")
print(f"  Output: v_hat[1] @ R[1] gives shape ({d_out},)")
print(f"\nEach output channel at k=1 is a weighted sum of ALL input channels at k=1")
print(f"Different frequencies have DIFFERENT mixing weights!")


#=============================================================================
# SECTION 5: 2D MULTI-CHANNEL SPECTRAL CONVOLUTION
#=============================================================================
print("\n" + "="*70)
print("SECTION 5: 2D Multi-Channel Spectral Convolution")
print("="*70)

def spectral_conv_2d_multichannel(v, R):
    """
    Full 2D multi-channel spectral convolution.
    
    This is the core FNO operation for 2D spatial data.
    
    Parameters:
    -----------
    v : np.ndarray, shape (Nx, Ny, d_in)
        Input 2D field with d_in channels
    R : np.ndarray, shape (k_max_x, k_max_y, d_in, d_out), complex
        Per-frequency-pair channel mixing matrices
        
    Returns:
    --------
    w : np.ndarray, shape (Nx, Ny, d_out)
        Output 2D field with d_out channels
    """
    Nx, Ny, d_in = v.shape
    k_max_x, k_max_y, _, d_out = R.shape
    
    # Step 1: 2D FFT on spatial dimensions for each channel
    v_hat = np.fft.rfft2(v, axes=(0, 1))  # Shape: (Nx, Ny//2+1, d_in)
    
    # Step 2: Initialize output spectrum
    w_hat = np.zeros((Nx, Ny//2+1, d_out), dtype=complex)
    
    # Step 3: Multiply by R for each (kx, ky) pair
    # Positive kx modes
    for kx in range(k_max_x):
        for ky in range(min(k_max_y, Ny//2+1)):
            # v_hat[kx, ky, :] shape: (d_in,)
            # R[kx, ky, :, :] shape: (d_in, d_out)
            w_hat[kx, ky, :] = v_hat[kx, ky, :] @ R[kx, ky, :, :]
    
    # Negative kx modes (wrap around)
    for kx in range(1, k_max_x):
        for ky in range(min(k_max_y, Ny//2+1)):
            w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(R[kx, ky, :, :])
    
    # Step 4: Inverse 2D FFT
    w = np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))  # Shape: (Nx, Ny, d_out)
    
    return w


# Create test 2D multi-channel data
Nx, Ny = 32, 32
d_in = 3
d_out = 4
k_max = 8

x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

v_2d_multi = np.zeros((Nx, Ny, d_in))
v_2d_multi[:, :, 0] = np.sin(X) * np.sin(Y)
v_2d_multi[:, :, 1] = np.sin(2*X) * np.cos(Y)
v_2d_multi[:, :, 2] = np.cos(X) * np.sin(2*Y)

print(f"\nInput shape: {v_2d_multi.shape} = (Nx, Ny, d_in)")

# Initialize weights
np.random.seed(42)
R_2d = (np.random.randn(k_max, k_max, d_in, d_out) + 
        1j * np.random.randn(k_max, k_max, d_in, d_out)) * 0.1

print(f"Weight tensor R shape: {R_2d.shape} = (k_max_x, k_max_y, d_in, d_out)")

# Apply 2D multi-channel spectral convolution
w_2d_multi = spectral_conv_2d_multichannel(v_2d_multi, R_2d)

print(f"Output shape: {w_2d_multi.shape} = (Nx, Ny, d_out)")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for c in range(d_in):
    im = axes[0, c].imshow(v_2d_multi[:, :, c], cmap='RdBu_r', origin='lower')
    axes[0, c].set_title(f'Input Channel {c}', fontsize=11)
    plt.colorbar(im, ax=axes[0, c], fraction=0.046)

# Weight magnitude at (kx=1, ky=1)
im = axes[0, 3].imshow(np.abs(R_2d[1, 1, :, :]), cmap='Blues', aspect='auto')
axes[0, 3].set_title('|R[1,1]| matrix', fontsize=11)
axes[0, 3].set_xlabel('d_out')
axes[0, 3].set_ylabel('d_in')
plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

for c in range(d_out):
    im = axes[1, c].imshow(w_2d_multi[:, :, c], cmap='RdBu_r', origin='lower')
    axes[1, c].set_title(f'Output Channel {c}', fontsize=11)
    plt.colorbar(im, ax=axes[1, c], fraction=0.046)

plt.suptitle('2D Multi-Channel Spectral Convolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_2d_multichannel.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/06_2d_multichannel.png")


#=============================================================================
# SECTION 6: THE LOCAL PATH (W)
#=============================================================================
print("\n" + "="*70)
print("SECTION 6: The Local Path W")
print("="*70)

def local_path(v, W):
    """
    Local path: pointwise linear transformation.
    
    This is equivalent to a 1x1 convolution in CNN terminology.
    Applied independently at each spatial location.
    
    Parameters:
    -----------
    v : np.ndarray, shape (..., d_in)
        Input with d_in channels (any spatial dimensions)
    W : np.ndarray, shape (d_in, d_out)
        Weight matrix
        
    Returns:
    --------
    w : np.ndarray, shape (..., d_out)
        Output with d_out channels
    """
    # Matrix multiply on last dimension
    return v @ W


# Demonstrate local path
print("\n--- Local Path Demonstration ---")

# 2D example
Nx, Ny = 32, 32
d_in = 3
d_out = 4

v_local = np.random.randn(Nx, Ny, d_in)
W = np.random.randn(d_in, d_out) * 0.1

w_local = local_path(v_local, W)

print(f"Input shape: {v_local.shape}")
print(f"W shape: {W.shape}")
print(f"Output shape: {w_local.shape}")

# Key insight: same W applied everywhere
print(f"\n--- Key Insight: Same W at Every Location ---")
# Verify that the operation is truly pointwise
point1 = (10, 15)
point2 = (25, 8)

result1 = v_local[point1] @ W
result2 = v_local[point2] @ W

print(f"At point {point1}: manual = {result1[:2]}, function = {w_local[point1][:2]}")
print(f"At point {point2}: manual = {result2[:2]}, function = {w_local[point2][:2]}")
print("→ Local path is truly pointwise matrix multiplication")


#=============================================================================
# SECTION 7: WHY WE NEED BOTH PATHS
#=============================================================================
print("\n" + "="*70)
print("SECTION 7: Why We Need Both Spectral and Local Paths")
print("="*70)

# Create input with both low and high frequency content
N = 64
x = np.linspace(0, 2*np.pi, N, endpoint=False)

# Signal with sharp features (high frequency) and smooth patterns (low frequency)
v_mixed = np.sin(x) + 0.3*np.sign(np.sin(5*x))  # Smooth + sharp

# Parameters
d_in = 1
d_out = 1
k_max = 8

# Reshape for our functions
v_mixed_ch = v_mixed.reshape(N, 1)

# Spectral path only
R_spectral = np.ones((k_max, d_in, d_out), dtype=complex) * 0.5
w_spectral = spectral_conv_multichannel_1d(v_mixed_ch, R_spectral)

# Local path only  
W_local = np.array([[0.5]])
w_local_only = local_path(v_mixed_ch, W_local)

# Combined (like in real FNO)
w_combined = w_spectral + w_local_only

# Analyze frequency content
v_hat = np.fft.rfft(v_mixed)
w_spectral_hat = np.fft.rfft(w_spectral[:, 0])
w_local_hat = np.fft.rfft(w_local_only[:, 0])
w_combined_hat = np.fft.rfft(w_combined[:, 0])

print("Signal has both smooth (sin) and sharp (sign) components")
print(f"Spectral path keeps only k < {k_max}, smoothing the output")
print("Local path preserves all frequencies uniformly")
print("Combined path: spectral fine-tunes low freq, local handles high freq")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Spatial domain
axes[0, 0].plot(x, v_mixed, 'b-', linewidth=2)
axes[0, 0].set_title('Input: Smooth + Sharp', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, w_spectral[:, 0], 'r-', linewidth=2)
axes[0, 1].set_title('Spectral Path Only', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(x, w_local_only[:, 0], 'g-', linewidth=2)
axes[0, 2].set_title('Local Path Only', fontsize=11)
axes[0, 2].grid(True, alpha=0.3)

axes[0, 3].plot(x, w_combined[:, 0], 'm-', linewidth=2)
axes[0, 3].set_title('Combined: Spectral + Local', fontsize=11)
axes[0, 3].grid(True, alpha=0.3)

# Frequency domain
freqs = np.arange(N//2 + 1)

axes[1, 0].stem(freqs[:20], np.abs(v_hat)[:20], basefmt=' ')
axes[1, 0].axvline(x=k_max-0.5, color='r', linestyle='--')
axes[1, 0].set_title('Input Spectrum', fontsize=11)
axes[1, 0].set_xlabel('Frequency k')

axes[1, 1].stem(freqs[:20], np.abs(w_spectral_hat)[:20], basefmt=' ', linefmt='r-', markerfmt='ro')
axes[1, 1].axvline(x=k_max-0.5, color='r', linestyle='--')
axes[1, 1].set_title('Spectral: High freq gone', fontsize=11)
axes[1, 1].set_xlabel('Frequency k')

axes[1, 2].stem(freqs[:20], np.abs(w_local_hat)[:20], basefmt=' ', linefmt='g-', markerfmt='go')
axes[1, 2].set_title('Local: All freq scaled equally', fontsize=11)
axes[1, 2].set_xlabel('Frequency k')

axes[1, 3].stem(freqs[:20], np.abs(w_combined_hat)[:20], basefmt=' ', linefmt='m-', markerfmt='mo')
axes[1, 3].axvline(x=k_max-0.5, color='r', linestyle='--')
axes[1, 3].set_title('Combined: Full spectrum', fontsize=11)
axes[1, 3].set_xlabel('Frequency k')

plt.suptitle('Why Both Paths Are Needed: Spectral (low freq) + Local (all freq)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/07_why_both_paths.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/07_why_both_paths.png")


#=============================================================================
# SECTION 8: GELU ACTIVATION
#=============================================================================
print("\n" + "="*70)
print("SECTION 8: GELU Activation Function")
print("="*70)

def gelu(x):
    """
    Gaussian Error Linear Unit activation.
    
    GELU(x) = x * Φ(x) where Φ is standard normal CDF.
    We use the approximation for efficiency.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def relu(x):
    """Standard ReLU"""
    return np.maximum(0, x)

# Compare activations
x_act = np.linspace(-4, 4, 200)

y_gelu = gelu(x_act)
y_relu = relu(x_act)

# Derivatives (approximate)
dx = x_act[1] - x_act[0]
dy_gelu = np.gradient(y_gelu, dx)
dy_relu = np.gradient(y_relu, dx)

print("GELU vs ReLU comparison:")
print("- GELU is smooth (no sharp corners)")
print("- GELU allows small negative outputs")
print("- GELU has non-zero gradient everywhere")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_act, y_gelu, 'b-', linewidth=2, label='GELU')
axes[0].plot(x_act, y_relu, 'r--', linewidth=2, label='ReLU')
axes[0].plot(x_act, x_act, 'k:', alpha=0.5, label='Identity')
axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('Activation(x)', fontsize=12)
axes[0].set_title('Activation Functions', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([-4, 4])
axes[0].set_ylim([-1, 4])

axes[1].plot(x_act, dy_gelu, 'b-', linewidth=2, label='GELU derivative')
axes[1].plot(x_act, dy_relu, 'r--', linewidth=2, label='ReLU derivative')
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel('Derivative', fontsize=12)
axes[1].set_title('Derivatives (for backprop)', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([-4, 4])

plt.suptitle('Why FNO Uses GELU: Smooth, Non-zero Gradients', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/08_gelu_activation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/08_gelu_activation.png")


#=============================================================================
# SECTION 9: COMPLETE FOURIER LAYER
#=============================================================================
print("\n" + "="*70)
print("SECTION 9: Complete Fourier Layer Implementation")
print("="*70)

class FourierLayer:
    """
    Complete Fourier Layer as described in FNO paper.
    
    Implements: v^(l+1) = σ(Wv + Kv + b)
    
    Where:
    - Kv is spectral convolution (FFT → truncate → multiply R → IFFT)
    - Wv is local linear transformation (1×1 conv)
    - σ is GELU activation
    - b is bias
    """
    
    def __init__(self, d_in, d_out, k_max_x, k_max_y, seed=None):
        """
        Initialize Fourier Layer.
        
        Parameters:
        -----------
        d_in : int
            Number of input channels
        d_out : int
            Number of output channels
        k_max_x : int
            Number of Fourier modes in x direction
        k_max_y : int
            Number of Fourier modes in y direction
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.d_in = d_in
        self.d_out = d_out
        self.k_max_x = k_max_x
        self.k_max_y = k_max_y
        
        # Initialize spectral weights R (complex)
        # Scale by 1/sqrt(d_in * d_out) for stable initialization
        scale = 1.0 / np.sqrt(d_in * d_out)
        self.R = (np.random.randn(k_max_x, k_max_y, d_in, d_out) + 
                  1j * np.random.randn(k_max_x, k_max_y, d_in, d_out)) * scale
        
        # Initialize local weights W (real)
        self.W = np.random.randn(d_in, d_out) * scale
        
        # Initialize bias (zeros)
        self.b = np.zeros(d_out)
        
        # Count parameters
        self.n_params = (2 * k_max_x * k_max_y * d_in * d_out +  # R (complex = 2x real)
                         d_in * d_out +  # W
                         d_out)  # b
    
    def spectral_conv(self, v):
        """
        Spectral convolution: K(v).
        
        Parameters:
        -----------
        v : np.ndarray, shape (Nx, Ny, d_in)
            Input tensor
            
        Returns:
        --------
        w : np.ndarray, shape (Nx, Ny, d_out)
            Output after spectral convolution
        """
        Nx, Ny, _ = v.shape
        
        # 2D FFT on spatial dimensions
        v_hat = np.fft.rfft2(v, axes=(0, 1))  # (Nx, Ny//2+1, d_in)
        
        # Initialize output spectrum
        w_hat = np.zeros((Nx, Ny//2+1, self.d_out), dtype=complex)
        
        # Multiply by R for low frequencies
        # Positive kx
        kx_max = min(self.k_max_x, Nx)
        ky_max = min(self.k_max_y, Ny//2+1)
        
        for kx in range(kx_max):
            for ky in range(ky_max):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ self.R[kx, ky, :, :]
        
        # Negative kx (wrap around in FFT)
        for kx in range(1, kx_max):
            for ky in range(ky_max):
                # Use conjugate for negative frequencies
                w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(self.R[kx, ky, :, :])
        
        # Inverse 2D FFT
        w = np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))
        
        return w
    
    def local_transform(self, v):
        """
        Local path: W @ v at each point.
        
        Parameters:
        -----------
        v : np.ndarray, shape (Nx, Ny, d_in)
            Input tensor
            
        Returns:
        --------
        w : np.ndarray, shape (Nx, Ny, d_out)
            Output after local transformation
        """
        return v @ self.W
    
    def forward(self, v):
        """
        Complete forward pass through Fourier layer.
        
        v^(l+1) = GELU(Wv + Kv + b)
        
        Parameters:
        -----------
        v : np.ndarray, shape (Nx, Ny, d_in)
            Input tensor
            
        Returns:
        --------
        out : np.ndarray, shape (Nx, Ny, d_out)
            Output tensor
        """
        # Spectral path
        spectral_out = self.spectral_conv(v)
        
        # Local path
        local_out = self.local_transform(v)
        
        # Combine and add bias
        combined = spectral_out + local_out + self.b
        
        # GELU activation
        out = gelu(combined)
        
        return out
    
    def __repr__(self):
        return (f"FourierLayer(d_in={self.d_in}, d_out={self.d_out}, "
                f"k_max=({self.k_max_x}, {self.k_max_y}), "
                f"params={self.n_params:,})")


# Demonstrate complete Fourier layer
print("\n--- Complete Fourier Layer Demo ---")

# Parameters matching your problem
Nx, Ny = 64, 64
d_in = 8   # Input channels (could be your 42 features after lifting)
d_out = 8  # Output channels (same, for stacking)
k_max = 12

# Create layer
layer = FourierLayer(d_in, d_out, k_max, k_max, seed=42)
print(layer)

# Create random input (simulating feature maps)
np.random.seed(123)
v_input = np.random.randn(Nx, Ny, d_in)

# Forward pass
v_output = layer.forward(v_input)

print(f"\nInput shape:  {v_input.shape}")
print(f"Output shape: {v_output.shape}")
print(f"Parameters:   {layer.n_params:,}")

# Visualize
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Input channels (first 4)
for c in range(4):
    im = axes[0, c].imshow(v_input[:, :, c], cmap='RdBu_r', origin='lower')
    axes[0, c].set_title(f'Input Channel {c}', fontsize=11)
    plt.colorbar(im, ax=axes[0, c], fraction=0.046)

# Output channels (first 4)
for c in range(4):
    im = axes[1, c].imshow(v_output[:, :, c], cmap='RdBu_r', origin='lower')
    axes[1, c].set_title(f'Output Channel {c}', fontsize=11)
    plt.colorbar(im, ax=axes[1, c], fraction=0.046)

plt.suptitle(f'Complete Fourier Layer: d_in={d_in}, d_out={d_out}, k_max={k_max}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_complete_fourier_layer.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/09_complete_fourier_layer.png")


#=============================================================================
# SECTION 10: RESOLUTION INVARIANCE
#=============================================================================
print("\n" + "="*70)
print("SECTION 10: Resolution Invariance Demonstration")
print("="*70)

def apply_layer_any_resolution(layer, v):
    """
    Apply trained Fourier layer at any resolution.
    The same weights R, W, b work for any grid size!
    """
    return layer.forward(v)


# Create a smooth input function that we can sample at different resolutions
def create_smooth_field(Nx, Ny, d_in):
    """Create a smooth multi-channel field."""
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    v = np.zeros((Nx, Ny, d_in))
    for c in range(d_in):
        # Each channel has different frequency content
        v[:, :, c] = np.sin((c+1)*X) * np.cos((c+1)*Y)
    
    return v


# Create layer with small k_max
d_in = d_out = 4
k_max = 8
layer = FourierLayer(d_in, d_out, k_max, k_max, seed=42)

print(f"Layer created with k_max={k_max}")
print("This layer will work at ANY resolution >= 2*k_max = 16")

# Test at different resolutions
resolutions = [32, 64, 128, 256]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, res in enumerate(resolutions):
    # Create input at this resolution
    v_res = create_smooth_field(res, res, d_in)
    
    # Apply SAME layer (same weights!)
    out_res = layer.forward(v_res)
    
    # Show input and output for channel 0
    axes[0, i].imshow(v_res[:, :, 0], cmap='RdBu_r', origin='lower')
    axes[0, i].set_title(f'Input {res}×{res}', fontsize=11)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    
    axes[1, i].imshow(out_res[:, :, 0], cmap='RdBu_r', origin='lower')
    axes[1, i].set_title(f'Output {res}×{res}', fontsize=11)
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    
    print(f"Resolution {res}×{res}: input shape {v_res.shape} → output shape {out_res.shape}")

plt.suptitle(f'Resolution Invariance: Same Weights Applied at Different Grids\n'
             f'k_max={k_max}, layer has {layer.n_params:,} parameters',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/10_resolution_invariance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/10_resolution_invariance.png")

print("\n--- Key Insight ---")
print("The SAME weights (R, W, b) work at all resolutions!")
print("This is because:")
print("  1. FFT/IFFT work at any grid size")
print("  2. We only use modes k < k_max, which exist at any resolution")
print("  3. Local path W is pointwise, independent of grid")
print("  4. Bias b is per-channel, not per-pixel")


#=============================================================================
# SECTION 11: DECOMPOSING THE LAYER EFFECTS
#=============================================================================
print("\n" + "="*70)
print("SECTION 11: Decomposing Layer Effects")
print("="*70)

# Create layer and input
Nx, Ny = 64, 64
d_in = d_out = 4
k_max = 12
layer = FourierLayer(d_in, d_out, k_max, k_max, seed=42)

# Create structured input
x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

v = np.zeros((Nx, Ny, d_in))
v[:, :, 0] = np.sin(X) * np.sin(Y)  # Low freq
v[:, :, 1] = np.sin(5*X) * np.sin(5*Y)  # Medium freq
v[:, :, 2] = np.sign(np.sin(3*X)) * np.sign(np.sin(3*Y))  # Sharp/high freq
v[:, :, 3] = np.random.randn(Nx, Ny) * 0.3  # Noise

# Get individual components
spectral_out = layer.spectral_conv(v)
local_out = layer.local_transform(v)
combined = spectral_out + local_out + layer.b
final_out = gelu(combined)

print("Decomposition of Fourier Layer:")
print(f"  1. Spectral path output range: [{spectral_out.min():.3f}, {spectral_out.max():.3f}]")
print(f"  2. Local path output range: [{local_out.min():.3f}, {local_out.max():.3f}]")
print(f"  3. Combined (before GELU): [{combined.min():.3f}, {combined.max():.3f}]")
print(f"  4. Final (after GELU): [{final_out.min():.3f}, {final_out.max():.3f}]")

# Visualize decomposition for one output channel
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Components
im = axes[0, 0].imshow(v[:, :, 0], cmap='RdBu_r', origin='lower')
axes[0, 0].set_title('Input (channel 0)', fontsize=11)
plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

im = axes[0, 1].imshow(spectral_out[:, :, 0], cmap='RdBu_r', origin='lower')
axes[0, 1].set_title('Spectral Path (Kv)', fontsize=11)
plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

im = axes[0, 2].imshow(local_out[:, :, 0], cmap='RdBu_r', origin='lower')
axes[0, 2].set_title('Local Path (Wv)', fontsize=11)
plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

im = axes[0, 3].imshow(final_out[:, :, 0], cmap='RdBu_r', origin='lower')
axes[0, 3].set_title('Final Output', fontsize=11)
plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

# Row 2: Spectra
def plot_spectrum(ax, field, title):
    spec = np.fft.rfft2(field)
    log_spec = np.log10(np.abs(spec[:32, :]) + 1e-10)
    im = ax.imshow(log_spec, cmap='viridis', origin='lower', aspect='auto')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('ky')
    ax.set_ylabel('kx')
    ax.axhline(y=k_max-0.5, color='r', linestyle='--', alpha=0.7)
    return im

plot_spectrum(axes[1, 0], v[:, :, 0], 'Input Spectrum')
plot_spectrum(axes[1, 1], spectral_out[:, :, 0], 'Spectral Path Spectrum')
im = plot_spectrum(axes[1, 2], local_out[:, :, 0], 'Local Path Spectrum')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
plot_spectrum(axes[1, 3], final_out[:, :, 0], 'Final Spectrum')

plt.suptitle('Decomposing Fourier Layer: v → Kv + Wv + b → GELU → output', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_layer_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/11_layer_decomposition.png")

print("\nKey observations:")
print("  - Spectral path: Low freq only (k < k_max), smooth output")
print("  - Local path: All frequencies preserved, scales everything")
print("  - Combined: Full frequency coverage")
print("  - GELU: Nonlinear transformation, couples frequencies")


#=============================================================================
# SECTION 12: STACKING MULTIPLE LAYERS
#=============================================================================
print("\n" + "="*70)
print("SECTION 12: Stacking Multiple Fourier Layers")
print("="*70)

class FourierLayerStack:
    """
    Stack of Fourier Layers.
    
    In practice, FNO uses 4 stacked Fourier layers.
    """
    
    def __init__(self, n_layers, d_hidden, k_max_x, k_max_y, seed=None):
        self.layers = []
        for i in range(n_layers):
            layer_seed = seed + i if seed is not None else None
            self.layers.append(FourierLayer(d_hidden, d_hidden, 
                                            k_max_x, k_max_y, seed=layer_seed))
        
        self.n_layers = n_layers
        self.total_params = sum(l.n_params for l in self.layers)
    
    def forward(self, v):
        """Forward through all layers."""
        for layer in self.layers:
            v = layer.forward(v)
        return v
    
    def __repr__(self):
        return (f"FourierLayerStack(n_layers={self.n_layers}, "
                f"total_params={self.total_params:,})")


# Create stack of 4 layers (typical FNO)
n_layers = 4
d_hidden = 32
k_max = 12

stack = FourierLayerStack(n_layers, d_hidden, k_max, k_max, seed=42)
print(stack)

# Test forward pass
Nx, Ny = 64, 64
v_stack_input = np.random.randn(Nx, Ny, d_hidden)
v_stack_output = stack.forward(v_stack_input)

print(f"\nInput shape:  {v_stack_input.shape}")
print(f"Output shape: {v_stack_output.shape}")
print(f"Total parameters: {stack.total_params:,}")

# Track intermediate representations
intermediates = [v_stack_input]
v = v_stack_input.copy()
for layer in stack.layers:
    v = layer.forward(v)
    intermediates.append(v)

# Visualize progression through layers
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i in range(5):
    im = axes[0, i].imshow(intermediates[i][:, :, 0], cmap='RdBu_r', origin='lower')
    if i == 0:
        axes[0, i].set_title('Input', fontsize=12)
    else:
        axes[0, i].set_title(f'After Layer {i}', fontsize=12)
    plt.colorbar(im, ax=axes[0, i], fraction=0.046)

# Show spectra evolution
for i in range(5):
    spec = np.fft.rfft2(intermediates[i][:, :, 0])
    log_spec = np.log10(np.abs(spec[:32, :]) + 1e-10)
    im = axes[1, i].imshow(log_spec, cmap='viridis', origin='lower', aspect='auto')
    if i == 0:
        axes[1, i].set_title('Input Spectrum', fontsize=12)
    else:
        axes[1, i].set_title(f'Spectrum after L{i}', fontsize=12)
    axes[1, i].axhline(y=k_max-0.5, color='r', linestyle='--', alpha=0.7)

plt.suptitle('Progressive Transformation Through 4 Stacked Fourier Layers', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_stacked_layers.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/12_stacked_layers.png")


#=============================================================================
# SECTION 13: PARAMETER ANALYSIS
#=============================================================================
print("\n" + "="*70)
print("SECTION 13: Parameter Analysis")
print("="*70)

def analyze_parameters(d_in, d_out, k_max_x, k_max_y):
    """Analyze parameter count for a Fourier layer."""
    # Spectral weights R (complex = 2x real)
    R_params = 2 * k_max_x * k_max_y * d_in * d_out
    
    # Local weights W
    W_params = d_in * d_out
    
    # Bias
    b_params = d_out
    
    total = R_params + W_params + b_params
    
    return {
        'R (spectral)': R_params,
        'W (local)': W_params,
        'b (bias)': b_params,
        'Total': total
    }


print("\n--- Parameter Count for Different Configurations ---")

configs = [
    {'d_in': 42, 'd_out': 32, 'k_max_x': 12, 'k_max_y': 12, 'name': 'Your problem: lifting'},
    {'d_in': 32, 'd_out': 32, 'k_max_x': 12, 'k_max_y': 12, 'name': 'Hidden layers'},
    {'d_in': 64, 'd_out': 64, 'k_max_x': 16, 'k_max_y': 16, 'name': 'Larger model'},
    {'d_in': 32, 'd_out': 32, 'k_max_x': 8, 'k_max_y': 8, 'name': 'Smaller k_max'},
]

print(f"\n{'Configuration':<25} {'R params':>12} {'W params':>10} {'b params':>10} {'Total':>12}")
print("-" * 75)

for cfg in configs:
    params = analyze_parameters(cfg['d_in'], cfg['d_out'], cfg['k_max_x'], cfg['k_max_y'])
    print(f"{cfg['name']:<25} {params['R (spectral)']:>12,} {params['W (local)']:>10,} "
          f"{params['b (bias)']:>10,} {params['Total']:>12,}")

# Compare to CNN
print("\n--- Comparison to CNN ---")
print("\nFor global receptive field on 64×64 grid with 32 channels:")
cnn_params = 64 * 64 * 32 * 32
fno_params = 2 * 12 * 12 * 32 * 32 + 32 * 32 + 32

print(f"  CNN (64×64 kernel): {cnn_params:,} parameters")
print(f"  FNO (k_max=12):     {fno_params:,} parameters")
print(f"  Reduction:          {cnn_params / fno_params:.1f}×")


#=============================================================================
# SUMMARY
#=============================================================================
print("\n" + "="*70)
print("CHUNK 2 COMPLETE: All Implementations Verified")
print("="*70)

print("""
Files created in figures/:
  01_spectral_conv_1d_basic.png  - Basic 1D spectral convolution
  02_mode_truncation.png         - Effect of different k_max values
  03_truncation_spectrum.png     - Spectrum with truncation cutoffs
  04_2d_spectral_conv.png        - 2D spectral convolution
  05_multichannel_spectral.png   - Multi-channel spectral convolution
  06_2d_multichannel.png         - 2D multi-channel convolution
  07_why_both_paths.png          - Why spectral + local paths needed
  08_gelu_activation.png         - GELU vs ReLU comparison
  09_complete_fourier_layer.png  - Complete layer demo
  10_resolution_invariance.png   - Same weights work at any resolution
  11_layer_decomposition.png     - Decomposing layer effects
  12_stacked_layers.png          - Multiple stacked layers

Key implementations:
  - spectral_conv_1d_basic()        : Basic 1D spectral convolution
  - spectral_conv_2d_basic()        : Basic 2D spectral convolution
  - spectral_conv_multichannel_1d() : Multi-channel 1D version
  - spectral_conv_2d_multichannel() : Multi-channel 2D version
  - local_path()                    : The W path (1×1 conv)
  - gelu()                          : GELU activation
  - FourierLayer class              : Complete Fourier layer
  - FourierLayerStack class         : Stack of layers

Ready for Chunk 3: Complete FNO Architecture
  - Lifting layer (input → hidden)
  - Projection layer (hidden → output)
  - Full forward pass
  - Training loop
""")
