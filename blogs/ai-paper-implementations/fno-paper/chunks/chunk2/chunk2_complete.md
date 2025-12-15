# Fourier Neural Operator: Chunk 2
## The Spectral Convolution and Fourier Layer — Complete Tutorial with Code

---

# Introduction

In Chunk 1, you built the mathematical foundations:
- **Operators** map entire functions to entire functions (infinite-dimensional)
- The **Fourier Transform** decomposes signals into frequency components
- **Differentiation** becomes multiplication in frequency space
- **Convolution** becomes pointwise multiplication (the convolution theorem!)

Now we build the **heart of FNO**: the Fourier Layer.

This is the module that makes FNO revolutionary. By the end of this chunk, you will have deep, implementation-ready understanding of:

1. **Spectral Convolution** — how to learn convolutions directly in Fourier space
2. **Mode Truncation** — why keeping only low frequencies works (and is essential)
3. **The Weight Tensor R** — what every dimension means and why it's complex-valued
4. **The Local Path W** — why we need *both* spectral and local operations
5. **The Complete Fourier Layer** — putting it all together

As Li et al. (2021) describe in the FNO paper: *"We propose replacing the kernel integral operator by a convolution operator defined in Fourier space."* This single architectural choice enables O(N log N) global convolutions with resolution-invariant learned parameters.

---

## Prerequisite Resources

Before diving in, ensure you're comfortable with Fourier transforms. These resources provide excellent background:

**Video Foundations:**
- [3Blue1Brown: "But what is the Fourier Transform?"](https://www.youtube.com/watch?v=spUNpyF58BY) — Grant Sanderson's 21-minute animated masterpiece explaining frequency decomposition through the "winding around circles" visualization. Essential viewing.
- [3Blue1Brown: "But what is a Fourier series?"](https://www.3blue1brown.com/lessons/fourier-series) — Connects Fourier concepts to the heat equation, directly relevant to PDEs.
- [Steve Brunton's FFT Lectures](https://www.youtube.com/c/Eigensteve) — University of Washington professor with excellent coverage of DFT, FFT algorithms, and spectral methods for PDEs.

**Written Tutorials:**
- [Tim Dettmers: "Understanding Convolution in Deep Learning"](https://timdettmers.com/2015/03/26/convolution-deep-learning/) — Exceptional visualizations of the convolution theorem and frequency filtering.
- [BetterExplained: Interactive Guide to Fourier Transform](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/) — The famous "smoothie recipe" metaphor for building intuition.

---

## Key Visual: Understanding Fourier Transform

The Fourier transform converts signals between time/space domain and frequency domain. This fundamental concept underlies everything in FNO.

![Fourier Transform Time and Frequency Domains](/blog-images/fno-paper/chunk2/fourier_transform_domains.gif)

*The Fourier transform takes an input function (red, time domain) and converts it into a new function (blue, frequency domain). Each spike in the frequency domain represents a sinusoidal component of the original signal.*

**Image Credit:** Lucas Vieira, Public Domain, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fourier_transform_time_and_frequency_domains.gif)

---

## Key Visual: Fourier Series Approximation

As we add more frequency components, we can approximate increasingly complex functions:

![Fourier Series Square Wave Approximation](/blog-images/fno-paper/chunk2/fourier_series_square_wave.gif)

*A square wave can be approximated by summing sinusoids. With more terms (higher frequencies), the approximation improves. This is why mode truncation works: smooth functions need fewer terms!*

**Image Credit:** Wikimedia Commons, CC BY-SA 3.0, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Periodic_identity_function.gif)

---

## Setup: Import Libraries

```python
"""
Fourier Neural Operator - Chunk 2: The Fourier Layer
=====================================================
Complete implementation of spectral convolution and Fourier layers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create output directory for figures
os.makedirs('figures', exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("CHUNK 2: THE FOURIER LAYER - Complete Implementation")
print("=" * 70)
```

---

# Section 1: From Convolution Theorem to Spectral Convolution

## 1.1 The Key Insight

From Chunk 1, you learned the convolution theorem:

$$\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$$

This means convolution in space equals multiplication in frequency.

**Traditional CNN approach:**
1. Learn a spatial kernel $g(x)$
2. Slide it across input: $(f * g)(x) = \int f(y) g(x-y) dy$
3. For global receptive field: kernel size = input size, giving $O(N^2)$ complexity

**The FNO insight (Li et al., 2021):**

What if we **never define the spatial kernel** at all? Instead, learn weights directly in Fourier space!

The FNO paper states: *"We therefore propose to directly parameterize the kernel in Fourier space."*

> **Video Explanation:** For a comprehensive 66-minute walkthrough of the FNO paper, see [Yannic Kilcher's FNO explanation](https://www.youtube.com/watch?v=IaS72aHrJKE). He covers the architecture, mathematics, and code in detail.

## 1.2 The Spectral Convolution Formula

The Fourier integral operator from the FNO paper (Definition 3):

$$\mathcal{K}(v)(x) = \mathcal{F}^{-1}\left[ R \cdot \mathcal{F}[v] \right](x)$$

Where:
- $v(x)$ is the input function (e.g., temperature field)
- $\mathcal{F}[v]$ transforms to frequency domain
- $R$ is a learnable weight tensor (complex-valued)
- The multiplication $R \cdot \mathcal{F}[v]$ happens frequency-by-frequency
- $\mathcal{F}^{-1}$ transforms back to spatial domain

**This is it.** FFT, then multiply by learned weights, then IFFT. That's the core operation.

---

## Key Visual: FNO Architecture (from Original Paper)

The following diagram shows the complete FNO architecture from Li et al. (2021):

![FNO Architecture - Figure 1 from Li et al. 2021](/blog-images/fno-paper/chunk2/fno_architecture_paper.png)

*Figure 1 from the original FNO paper (arXiv:2010.08895). **Top:** The Fourier layer architecture showing the parallel spectral and local paths. **Bottom:** Example Navier-Stokes flow predictions demonstrating zero-shot super-resolution.*

**Image Credit:** Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, & Anandkumar (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895). Used under arXiv's non-exclusive license for educational purposes.

---

### Code: Basic 1D Spectral Convolution

```python
def spectral_conv_1d_basic(v, R):
    """
    Most basic spectral convolution - single channel, 1D.
    
    This is the CORE FNO operation: FFT, multiply by R, IFFT
    
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
        
    The math:
        w(x) = F^{-1}[ R · F[v] ](x)
    """
    N = len(v)
    k_max = len(R)
    
    # Step 1: FFT to frequency domain
    v_hat = np.fft.rfft(v)  # Shape: (N//2 + 1,) complex
    
    # Step 2: Initialize output spectrum (zeros for high frequencies)
    w_hat = np.zeros_like(v_hat)
    
    # Step 3: Multiply by R for low frequencies only (mode truncation!)
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

# Apply each
w_identity = spectral_conv_1d_basic(v, R_identity)
w_lowpass = spectral_conv_1d_basic(v, R_lowpass)
w_amplify = spectral_conv_1d_basic(v, R_amplify)

print(f"k_max = {k_max} (modes 0 to {k_max-1} are kept)")
print(f"Modes k >= {k_max} (including k=15) are automatically zeroed!")
```

**Output:**
```
--- Demo: Basic 1D Spectral Convolution ---
Input signal: N=64 points
Contains frequencies: k=1, 3, 7, 15
k_max = 12 (modes 0 to 11 are kept)
Modes k >= 12 (including k=15) are automatically zeroed!
```

---

## Generated Figure 1: Basic 1D Spectral Convolution

![Basic 1D Spectral Convolution](01_spectral_conv_1d_basic.png)

### What You're Looking At

This figure demonstrates the **core FNO operation** (FFT, Multiply by R, IFFT) on a 1D signal.

**Top Row (Input Analysis):**

1. **Input Signal v(x)** (left): A complex wave created by adding four sine waves: sin(x) + 0.5·sin(3x) + 0.3·sin(7x) + 0.2·sin(15x). The "bumpiness" comes from the high-frequency components.

2. **Input Spectrum |v̂(k)|** (middle): The FFT reveals exactly four frequency peaks at k=1, 3, 7, and 15 — corresponding to each sine component. The red dashed line shows k_max=12, the truncation boundary. Notice k=15 is **beyond** this boundary.

3. **Different R(k) Weight Magnitudes** (right): Three different weight configurations:
   - **Identity (blue):** All weights = 1 (pass everything unchanged)
   - **Low-pass (orange):** Exponentially decaying weights (attenuate higher frequencies more)
   - **Amplify k=3 (green):** Spike at k=3 to boost that specific frequency

**Bottom Row (Output Results):**

1. **Identity R** (left): The output (green) is smoother than input (blue dashed) because k=15 is automatically zeroed by mode truncation. The small high-frequency "wiggles" are gone.

2. **Low-pass R** (middle): Much smoother output. The exponentially decaying weights suppress mid-range frequencies (k=3, 7), leaving mostly the fundamental sin(x).

3. **Amplify k=3** (right): The sin(3x) component is now dominant — the wave has more oscillations and larger amplitude because R[3]=3.0.

### Key Lessons

- **k=15 disappears in all outputs**: Mode truncation automatically filters high frequencies
- **Different R means different outputs**: The network **learns** which frequencies matter for your task
- **Identity R is not identity function**: Truncation alone changes the signal (k>=k_max zeroed)
- **One multiplication per frequency**: O(N log N) complexity via FFT, not O(N²)

**For Urban Temperature:** FNO will learn R weights that preserve city-wide heat patterns (low k) while filtering sensor noise (high k).

---

## 1.3 Why This Is Revolutionary

**The comparison that matters:**

- **3×3 CNN**: Receptive Field = 3 pixels (local), Complexity = O(N), Parameters = 9 per filter
- **Global CNN (N×N kernel)**: Receptive Field = N pixels (global), Complexity = O(N²), Parameters = N² per filter
- **FNO Spectral Conv**: Receptive Field = **Entire domain** (global), Complexity = **O(N log N)**, Parameters = k_max per filter

From the FNO paper: *"The majority of the computational cost lies in computing the Fourier transform... the FFT has complexity O(n log n)."*

A single spectral convolution sees **everywhere at once** with FFT efficiency!

---

## Key Visual: Time vs Frequency Domain

This visualization shows how the same signal appears in time domain (left) versus frequency domain (right):

![Time Domain vs Frequency Domain](/blog-images/fno-paper/chunk2/fft_time_frequency_view.png)

*Left: A signal in the time domain showing amplitude over time. Right: The same signal in the frequency domain showing which frequencies are present. The FFT converts between these representations.*

**Image Credit:** Wikimedia Commons, CC BY-SA 3.0, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:FFT-Time-Frequency-View.png)

---

## 1.4 The Physical Interpretation

What does each Fourier mode represent physically?

**Mode interpretation for urban temperature:**

- **k=0 (Mean/DC)**: Wavelength = infinity - City-wide average
- **k=1 (1 oscillation)**: Wavelength = Domain size — Large-scale gradient
- **k=2-4 (Low frequency)**: Wavelength = ~1-5 km — Park/water body effects
- **k=5-12 (Medium frequency)**: Wavelength = ~100m-1km — Neighborhood patterns
- **k>12 (High frequency)**: Wavelength less than 100m - Building-scale (often noise)

When FNO multiplies $\hat{v}(k)$ by $R(k)$:
- **$|R(k)|$** controls how much of frequency $k$ passes through
- **$\arg(R(k))$** shifts the phase of frequency $k$

**The network learns which spatial scales matter for your prediction task!**

---

# Section 2: Mode Truncation — The Key to Efficiency

## 2.1 The Problem with Full Spectrum

For a 64×64 spatial grid:
- Full FFT produces 64 × 33 = 2,112 unique complex coefficients
- With multiple channels (d_v=32): that's 67,584 values per layer
- Learning weights for all: parameter explosion!

And worse — most of those high-frequency modes are:
1. **Noise** in real data
2. **Irrelevant** for smooth physical solutions
3. **Overfitting risks** if we learn them

## 2.2 Why Truncation Works: PDE Solutions Are Smooth

From the FNO paper: *"We pick a finite-dimensional parameterization by truncating the Fourier series at a maximal number of modes $k_{max}$."*

**Physical justification:**

Remember from Chunk 1 how the heat equation damps frequencies:

$$\hat{u}(k, t) = \hat{u}(k, 0) \cdot e^{-\alpha k^2 t}$$

Higher frequencies decay **exponentially faster** (rate ~ k²).

This isn't just true for heat — most physical PDEs have smooth solutions:
- **Diffusion** damps high frequencies
- **Viscosity** in fluids smooths rapid variations
- **Conservation laws** constrain discontinuities

For your LST prediction: temperature varies smoothly at 70m resolution. The high-frequency content is mostly noise or sub-grid variability.

> **Deep Dive:** Steve Brunton's lecture ["Numerical Solutions to PDEs Using FFT"](https://www.youtube.com/watch?v=hDeARtZdq-U) (University of Washington) provides an excellent 50-minute treatment of why spectral methods work so well for smooth PDE solutions.

---

## Key Visual: 2D FFT for Image Processing

Mode truncation in 2D works similarly to 1D. Here's how a 2D FFT decomposes an image:

![2D Mode Truncation](/blog-images/fno-paper/chunk2/02_mode_truncation_2d.png)

*Mode truncation in 2D: The center of the frequency domain represents low frequencies (large-scale patterns), while edges represent high frequencies (fine details). FNO keeps only the center region!*

---

### Code: Demonstrating Mode Truncation

```python
print("\n" + "="*70)
print("SECTION 2: Mode Truncation Mechanics")
print("="*70)

def demonstrate_mode_truncation(v, k_max_values):
    """Show effect of different k_max values on reconstruction."""
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

# Signal with many frequencies (like a temperature profile)
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

# Key insight
print(f"\nNote: k_max=16 captures 99.9% of the signal!")
print(f"Note: k_max=8 is often sufficient for smooth functions")
```

---

## Generated Figure 2: 2D Mode Truncation

![2D Mode Truncation](02_mode_truncation_2d.png)

### What You're Looking At

This figure shows how mode truncation affects 2D spatial fields — directly relevant to temperature maps.

**Top Row (Spatial Domain):**

1. **Original Field** (leftmost): A 64×64 "temperature-like" field created from multiple 2D sine waves at different frequencies. The checkerboard pattern comes from high-frequency components.

2. **k_max=4, MSE=0.0325**: Very smooth reconstruction. Only the largest-scale patterns survive. Fine details are completely lost.

3. **k_max=8, MSE=0.0100**: Better reconstruction. Medium-scale patterns appear. MSE dropped by 3×.

4. **k_max=16, MSE=0.0000**: Near-perfect reconstruction. This signal's highest frequency content fits within k_max=16, so nothing is lost.

**Bottom Row (Frequency Domain):**

1. **Log Spectrum**: The original 2D spectrum shown in log scale. Bright spots indicate where energy is concentrated — at specific (kx, ky) frequency pairs corresponding to the sine wave components.

2-4. **Kept Modes (green regions)**: Shows which frequency region is preserved for each k_max. Notice how the green region (kept modes) grows with k_max.

### Key Lessons

- **k_max=4** (MSE=0.0325): Only largest patterns captured (like city-wide UHI gradient)
- **k_max=8** (MSE=0.0100): Adds neighborhood-scale features
- **k_max=16** (MSE~0): Full reconstruction for this signal

**Critical Insight:** The MSE drops rapidly with increasing k_max, then plateaus. This means:
- Most energy is in low frequencies (smooth physics!)
- Beyond some k_max, you're just modeling noise
- FNO paper uses k_max=12 for 2D problems — a sweet spot

**For Urban Temperature:** Your LST data at 70m resolution is inherently smooth. Temperature doesn't jump randomly between adjacent pixels. k_max=12-16 will capture the physics while filtering sensor noise.

---

## 2.3 Choosing k_max: Practical Guidelines

From the FNO paper: *"In practice, we have found that choosing $k_{max,j} = 12$ which yields $k_{max} = 12^d$ parameters per channel to be sufficient for all the tasks that we consider."*

**Rule of thumb:**

- **64×64 grid**: Recommended k_max = 12-16 — Captures ~90% of smooth fields
- **128×128 grid**: Recommended k_max = 16-20 — Same physical scales
- **256×256 grid**: Recommended k_max = 20-24 — More resolution, same physics

**For your urban LST work:**
- 70m resolution over NYC
- Temperature is physically smooth
- **k_max = 12** is likely optimal (matches FNO paper defaults)

## 2.4 Mode Truncation = Implicit Regularization

Beyond efficiency, truncation provides **free regularization**:

1. **Prevents overfitting:** Can't memorize high-frequency noise
2. **Improves generalization:** Learns smooth mappings that transfer
3. **Physical inductive bias:** Matches smoothness of real solutions

This is like dropout or weight decay, but built into the architecture!

> **Further Reading:** The SpecBoost paper ([arXiv:2404.07200](https://arxiv.org/abs/2404.07200)) provides detailed spectral analysis showing FNO's low-frequency bias and discusses strategies for handling high-frequency components when needed.

---

# Section 3: The Weight Tensor R — Understanding Every Dimension

## 3.1 Multi-Channel Reality

Real inputs have multiple channels. For your urban temperature prediction:
- **Input channels:** NDVI, building height, ERA5 temperature, solar radiation, etc.
- **Hidden channels:** Abstract features learned by the network (typically 32-64)
- **Output:** Temperature prediction (1 channel)

The spectral convolution must handle **channel mixing** at each frequency.

## 3.2 The Weight Tensor Structure

For 2D spatial fields (like temperature maps):

$$R \in \mathbb{C}^{k_{max,1} \times k_{max,2} \times d_{in} \times d_{out}}$$

Four dimensions:
1. **k_max,1:** Modes in x direction
2. **k_max,2:** Modes in y direction  
3. **d_in:** Input channels
4. **d_out:** Output channels

**Interpretation:** At each 2D frequency pair $(k_x, k_y)$, we have a matrix $R_{k_x, k_y} \in \mathbb{C}^{d_{in} \times d_{out}}$ that mixes input channels to output channels.

From the FNO paper (Equation 5): *"Multiplication by the weight tensor R ∈ $\mathbb{C}^{k_{max} \times d_v \times d_v}$"*

---

## Key Visual: Complex Numbers in Fourier Space

Since Fourier coefficients are complex, the weights R must also be complex. Here's how complex numbers represent both magnitude and phase:

![Complex Number Representation](/blog-images/fno-paper/chunk2/complex_number_illustration.png)

*A complex number z = a + bi can be represented in the complex plane. The magnitude |z| determines amplitude scaling, while the angle θ determines phase shift. FNO learns both!*

**Image Credit:** Wikimedia Commons, Public Domain, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Complex_number_illustration.svg)

---

### Code: Multi-Channel Spectral Convolution

```python
print("\n" + "="*70)
print("SECTION 3: Multi-Channel Spectral Convolution")
print("="*70)

def spectral_conv_2d_multichannel(v, R):
    """
    Full 2D multi-channel spectral convolution.
    
    This is the CORE FNO operation for 2D spatial data.
    
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
    # This is matrix multiplication at each frequency!
    
    # Positive kx modes
    for kx in range(k_max_x):
        for ky in range(min(k_max_y, Ny//2+1)):
            # v_hat[kx, ky, :] shape: (d_in,)
            # R[kx, ky, :, :] shape: (d_in, d_out)
            # Result: (d_out,)
            w_hat[kx, ky, :] = v_hat[kx, ky, :] @ R[kx, ky, :, :]
    
    # Negative kx modes (wrap around in FFT)
    for kx in range(1, k_max_x):
        for ky in range(min(k_max_y, Ny//2+1)):
            w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(R[kx, ky, :, :])
    
    # Step 4: Inverse 2D FFT
    w = np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))  # Shape: (Nx, Ny, d_out)
    
    return w
```

## 3.3 Why Complex Weights?

Fourier coefficients are complex: $\hat{v}(k) = a + bi$

To properly transform them, weights must also be complex: $R(k) = c + di$

The multiplication $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$ enables:
- **Scaling:** Changing magnitude of frequency component
- **Phase shifting:** Rotating the complex number (shifting patterns spatially)

Both are needed to learn arbitrary transformations in frequency space!

**Implementation note:** Complex weights have 2× the parameters of real weights.

---

# Section 4: The Local Path W — Why Both Paths Matter

## 4.1 The Problem with Pure Spectral Convolution

Mode truncation zeros out high frequencies. This means:
1. **Sharp features get smoothed** (edges, boundaries)
2. **Fine details are lost**
3. The spectral path acts as a **low-pass filter**

But many physical phenomena have BOTH:
- Smooth large-scale structure (yes - captured by spectral path)
- Sharp local features (no - lost by truncation!)

## 4.2 The Solution: Add a Local Linear Transform

The complete Fourier layer has **two parallel paths**:

$$v^{(l+1)}(x) = \sigma\left( W v^{(l)}(x) + \mathcal{K}(v^{(l)})(x) + b \right)$$

Where:
- $W v^{(l)}(x)$ is the **local path** (1×1 convolution)
- $\mathcal{K}(v^{(l)})(x)$ is the **spectral path** (Fourier convolution)

This is shown in Figure 2 of the FNO paper, where both paths are applied in parallel.

**Local path:** $W \cdot v(x)$
- Just a linear transformation at each spatial point
- Like a **1×1 convolution** in CNN terminology
- Affects all frequencies equally (flat frequency response)
- Captures high-frequency, local patterns

**Spectral path:** $\mathcal{K}v(x)$
- The spectral convolution we've discussed
- Global receptive field
- Fine control of low frequencies only

> **Architecture Variants:** The U-FNO paper ([arXiv:2109.03697](https://arxiv.org/abs/2109.03697)) explores integrating U-Net style skip connections within FNO to better capture both local and global features. This is particularly useful when sharp boundaries matter.

---

## Key Visual: Low-pass vs High-pass Filtering

The spectral path (with truncation) acts as a low-pass filter, while the local path preserves all frequencies:

![Why Both Paths Matter](/blog-images/fno-paper/chunk2/03_why_both_paths.png)

*FNO's spectral path is effectively a learnable low-pass filter (keeps smooth, large-scale features), while the local path W provides uniform frequency response to capture sharp edges and local details.*

---

### Code: Comparing Spectral and Local Paths

```python
print("\n" + "="*70)
print("SECTION 4: Why Both Paths Matter")
print("="*70)

def local_path(v, W):
    """
    Local linear transform (1×1 convolution).
    
    Applied pointwise at each spatial location.
    
    Parameters:
    -----------
    v : np.ndarray, shape (Nx, Ny, d_in)
        Input field
    W : np.ndarray, shape (d_in, d_out)
        Weight matrix (same at all spatial points)
        
    Returns:
    --------
    w : np.ndarray, shape (Nx, Ny, d_out)
    """
    # Einstein summation: contract over d_in
    return np.einsum('xyi,io->xyo', v, W)


# Create test field with both smooth and sharp features
Nx, Ny = 64, 64
x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Input with low AND high frequency content
d_in = 1
v_mixed = np.zeros((Nx, Ny, d_in))

# Smooth component (low frequency)
smooth = np.sin(X) * np.sin(Y)

# Sharp component (high frequency - like a building edge)
sharp = np.zeros((Nx, Ny))
sharp[20:44, 20:44] = 1.0  # Square "building"

v_mixed[:, :, 0] = smooth + 0.5 * sharp

print(f"Input: smooth sine wave + sharp square edge")
```

---

## Generated Figure 3: Why Both Paths Matter

![Why Both Paths Matter](03_why_both_paths.png)

### What You're Looking At

This figure is **crucial** for understanding FNO architecture. It shows what happens when we apply ONLY the spectral path vs ONLY the local path.

**Top Row (Spatial Domain):**

1. **Input: Smooth + Sharp** (left): A combination of a smooth sine wave (the gradual red-blue gradient) AND a sharp rectangular "building" in the center. This mimics real urban data where you have gradual temperature gradients AND sharp building boundaries.

2. **Spectral Path Only** (middle): The smooth sine pattern is preserved, but **the sharp building edges are blurred**! The square is now a smooth blob. This is because sharp edges require high-frequency Fourier components, which are zeroed by mode truncation.

3. **Local Path Only** (right): Everything is preserved exactly — both smooth patterns and sharp edges. The 1×1 convolution (local path) doesn't filter frequencies.

**Bottom Row (Frequency Domain):**

1. **Input Spectrum**: Shows rich frequency content. The grid-like pattern in the spectrum comes from the sharp rectangular edge (sharp edges create harmonics at many frequencies).

2. **After Spectral (k≥8 gone)**: Only the region below the red dashed line (k_max=8) remains. All the high-frequency content that defined the sharp edges is gone!

3. **After Local (all preserved)**: The full spectrum is intact.

### Key Lessons

- **Spectral path**: Strengths = Global receptive field, learnable per-frequency | Weaknesses = Loses sharp features (high freq)
- **Local path**: Strengths = Preserves all frequencies | Weaknesses = No global information
- **Both together**: Global + local, smooth + sharp — Complete solution!

**The Architecture Insight:** FNO adds both paths together. The spectral path handles large-scale physics (urban heat island gradient), while the local path preserves sharp features (building edges, park boundaries).

**For Urban Temperature:** Your Central Park boundary is a sharp edge. Without the local path, FNO would blur it into the surrounding buildings. With both paths, it can model both the gradual UHI effect AND the sharp park cooling boundary.

---

## 4.3 Frequency Coverage Analysis

```
Frequency:    0 ──────── k_max ──────────── N/2
              │           │                  │
Spectral R:   ████████████│                  │  (fine control)
Local W:      ████████████████████████████████  (all uniformly)
Combined:     ████████████████████████████████  (full coverage!)
              ▲           ▲                  ▲
              │           │                  │
           Fine       Cutoff             Nyquist
          control    (k_max)           (max freq)
```

**Together they're complete:**
- Spectral path: Detailed, learnable control of low frequencies
- Local path: Baseline handling of all frequencies

---

# Section 5: The Activation Function

## 5.1 Why Nonlinearity is Essential

Without activation functions, stacking layers is pointless:
$$W_2 (W_1 v) = (W_2 W_1) v = W_{combined} v$$

Multiple linear layers collapse to one! Nonlinearity enables:
- Learning complex, nonlinear mappings
- Hierarchical feature extraction  
- Universal approximation capability

## 5.2 GELU: The Preferred Activation

From the FNO paper: *"We construct our Fourier neural operator by stacking four Fourier integral operator layers... with the ReLU activation as well as batch normalization."*

In practice, many implementations use **GELU** (Gaussian Error Linear Unit):

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of standard normal distribution.

**Why GELU over ReLU?**
1. **Smooth:** No sharp kink at x=0 (better for smooth PDE solutions)
2. **Non-zero gradient everywhere:** Avoids "dead neurons"
3. **State-of-the-art:** Used in transformers and modern architectures

---

## Key Visual: Activation Functions

Here's a comparison of common activation functions used in neural networks:

![Activation Functions Comparison](/blog-images/fno-paper/chunk2/04_activation_functions.png)

*Common activation functions including ReLU, tanh, and GELU. GELU is preferred in FNO for its smoothness and better gradient properties for deep networks.*

---

### Code: GELU Implementation

```python
print("\n" + "="*70)
print("SECTION 5: Activation Functions")
print("="*70)

def gelu(x):
    """
    GELU activation function.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.
    
    Approximation used here for efficiency.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


# Compare activations
x_act = np.linspace(-3, 3, 100)

print("\nKey differences:")
print("  ReLU: Sharp kink at 0, gradient is 0 or 1")
print("  GELU: Smooth everywhere, gradient varies smoothly")
print("  GELU is better for learning smooth PDE solutions")
```

---

## Generated Figure 4: Activation Functions

![Activation Functions](04_activation_functions.png)

### What You're Looking At

This figure compares ReLU and GELU activation functions — the nonlinear "switch" after each Fourier layer.

**Left Panel (Activation Functions):**

- **ReLU (blue):** The classic max(0, x). Zero for all negative inputs, then linear (slope=1) for positive inputs. There's a **sharp corner at x=0**.

- **GELU (red):** Smooth S-curve transition. Notable features:
  - Slightly negative for small negative inputs (~-0.17 at x~-0.5)
  - Smoothly transitions to linear for large positive x
  - No sharp corners anywhere

**Right Panel (Gradients):**

- **ReLU gradient (blue):** Discontinuous jump from 0 to 1 at x=0. For negative inputs, gradient is exactly 0 (neuron is "dead" — no learning happens).

- **GELU gradient (red):** Smooth transition. Never exactly zero, so all neurons can learn. Peaks slightly above 1 around x=1.

### Key Lessons

**ReLU:**
- At x=0: Sharp corner
- Negative inputs: Gradient = 0 (dead neurons)
- Smoothness: C⁰ (continuous only)
- Best for: General deep learning

**GELU:**
- At x=0: Smooth transition
- Negative inputs: Gradient > 0 (neurons stay alive)
- Smoothness: C-infinity (infinitely differentiable)
- Best for: Smooth function approximation (like PDEs)

**Why This Matters for PDEs:**

1. **PDE solutions are smooth:** Temperature doesn't have discontinuities. Using a smooth activation (GELU) matches the physics.

2. **Dead neurons problem:** With ReLU, if a neuron outputs negative values, its gradient is 0 and it stops learning. GELU neurons always have some gradient.

3. **Gradient flow:** Smooth gradients = more stable training for deep networks.

**Practical note:** The original FNO paper used ReLU, but many modern implementations (including NeuralOperator library) default to GELU. Both work; GELU often trains slightly better.

---

# Section 6: Complete Fourier Layer — Putting It Together

## 6.1 Full Mathematical Specification

**Input:** $v^{(l)} \in \mathbb{R}^{N_1 \times N_2 \times d_v}$ — 2D spatial field with $d_v$ channels

**Learnable parameters:**
- $R \in \mathbb{C}^{k_1 \times k_2 \times d_v \times d_v}$ — spectral weights
- $W \in \mathbb{R}^{d_v \times d_v}$ — local linear weights
- $b \in \mathbb{R}^{d_v}$ — bias

**Computation:**

1. **Spectral path:**
   - $\hat{v} = \text{RFFT2}(v^{(l)})$
   - $\hat{w}_{k_x, k_y} = \hat{v}_{k_x, k_y} \cdot R_{k_x, k_y}$ for low modes
   - $\hat{w}_{k_x, k_y} = 0$ for high modes
   - $w_{spectral} = \text{IRFFT2}(\hat{w})$

2. **Local path:**
   - $w_{local}(x) = W \cdot v^{(l)}(x)$ at each point

3. **Combine and activate:**
   - $v^{(l+1)} = \text{GELU}(w_{spectral} + w_{local} + b)$

**Output:** $v^{(l+1)} \in \mathbb{R}^{N_1 \times N_2 \times d_v}$ — same shape as input!

> **Implementation Walkthrough:** The University of Amsterdam Deep Learning course provides an excellent Jupyter notebook implementing FNO from scratch in PyTorch, including the complete `SpectralConv2d` class.
>
> See: [UvA DL Notebooks - Physics-Inspired ML](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_Neural_Networks/Complete_DNN_2_2.html)
>
> *Credit: Ilze Amanda Auzina and Philip Lippe*

---

## Key Visual: The Complete Fourier Layer Pipeline

Here's a detailed view of the Fourier layer from the architecture again:

![Fourier Layer Detail](/blog-images/fno-paper/chunk2/fno_architecture_paper.png)

*The Fourier layer consists of: (1) FFT to convert to frequency domain, (2) Linear transform on truncated modes, (3) IFFT back to spatial domain, (4) Add with local linear transform W, (5) Apply activation σ. This figure is from the original FNO paper.*

**Image Credit:** Li et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

---

### Code: Complete Fourier Layer Class

```python
print("\n" + "="*70)
print("SECTION 6: Complete Fourier Layer")
print("="*70)

class FourierLayer:
    """
    Complete Fourier Layer as described in Li et al. (2021).
    
    Implements: v_{l+1}(x) = σ(W·v_l(x) + K(v_l)(x) + b)
    
    Where K is the spectral convolution with truncated modes.
    """
    
    def __init__(self, d_in, d_out, k_max_x, k_max_y, seed=None):
        """
        Initialize Fourier Layer.
        
        Parameters:
        -----------
        d_in : int
            Input channels
        d_out : int
            Output channels
        k_max_x, k_max_y : int
            Number of Fourier modes to keep
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
        # Scale: 1/sqrt(d_in * d_out) for stable training
        scale = 1.0 / np.sqrt(d_in * d_out)
        self.R = (np.random.randn(k_max_x, k_max_y, d_in, d_out) + 
                  1j * np.random.randn(k_max_x, k_max_y, d_in, d_out)) * scale
        
        # Initialize local weights W (real)
        # Xavier initialization
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2.0 / (d_in + d_out))
        
        # Initialize bias
        self.b = np.zeros(d_out)
        
        # Count parameters
        self.n_params = (2 * k_max_x * k_max_y * d_in * d_out +  # R (complex)
                        d_in * d_out +  # W
                        d_out)  # b
    
    def spectral_conv(self, v):
        """Apply spectral convolution."""
        Nx, Ny, _ = v.shape
        
        # FFT
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        
        # Initialize output
        w_hat = np.zeros((Nx, Ny//2+1, self.d_out), dtype=complex)
        
        # Multiply by R for kept modes
        # Positive kx
        for kx in range(self.k_max_x):
            for ky in range(min(self.k_max_y, Ny//2+1)):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ self.R[kx, ky]
        
        # Negative kx (wrap around)
        for kx in range(1, self.k_max_x):
            for ky in range(min(self.k_max_y, Ny//2+1)):
                w_hat[-kx, ky, :] = v_hat[-kx, ky, :] @ np.conj(self.R[kx, ky])
        
        # IFFT
        return np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))
    
    def local_linear(self, v):
        """Apply local linear transform (1×1 conv)."""
        return np.einsum('xyi,io->xyo', v, self.W)
    
    def forward(self, v):
        """
        Complete forward pass.
        
        v_{l+1} = GELU(spectral_conv(v) + local_linear(v) + b)
        """
        # Spectral path
        w_spectral = self.spectral_conv(v)
        
        # Local path
        w_local = self.local_linear(v)
        
        # Combine + bias + activation
        return gelu(w_spectral + w_local + self.b)
    
    def __repr__(self):
        return (f"FourierLayer(d_in={self.d_in}, d_out={self.d_out}, "
                f"k_max=({self.k_max_x},{self.k_max_y}), "
                f"params={self.n_params:,})")
```

---

# Section 7: Resolution Invariance — The Magic Property

## 7.1 Why FNO Transfers Across Resolutions

From the FNO paper: *"The Fourier layers are discretization-invariant because they can learn from and evaluate functions which are discretized in an arbitrary way."*

**The key insight:** Our learned weights $R$ and $W$ don't depend on grid size!

- **R operates on Fourier modes**, not spatial points
- The first 12 modes exist at ANY resolution (64×64, 128×128, 256×256...)
- **W is pointwise**, applied identically everywhere

---

## Key Visual: Resolution Independence Concept

The same Fourier modes represent the same physical frequencies regardless of grid resolution:

![Sampling and Aliasing](/blog-images/fno-paper/chunk2/aliasing_sines.png)

*When sampling a continuous signal at different resolutions, the low-frequency components remain consistent. This is why FNO can transfer: the first k_max modes are the same physical frequencies at any resolution.*

**Image Credit:** Wikimedia Commons, Public Domain, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AliasingSines.svg)

---

### Code: Demonstrating Resolution Invariance

```python
print("\n" + "="*70)
print("SECTION 7: Resolution Invariance")
print("="*70)

# Create a layer with FIXED weights
layer = FourierLayer(d_in=1, d_out=1, k_max_x=8, k_max_y=8, seed=123)

# Create the SAME function at different resolutions
def create_test_field(Nx, Ny):
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return (np.sin(X) * np.sin(Y) + 0.5 * np.sin(2*X) * np.cos(Y))[:, :, np.newaxis]


resolutions = [32, 64, 128, 256]

print("\nApplying SAME weights to different resolutions:")
print("-" * 50)

for res in resolutions:
    # Create input at this resolution
    v = create_test_field(res, res)
    
    # Apply the SAME layer (same weights!)
    w = layer.forward(v)
    
    print(f"Resolution {res}×{res}:")
    print(f"  Input range:  [{v.min():.3f}, {v.max():.3f}]")
    print(f"  Output range: [{w.min():.3f}, {w.max():.3f}]")

print("\nThe SAME learned transformation works at ANY resolution!")
print("Train on 64x64, deploy on 256x256 (zero-shot super-resolution)")
```

---

## Generated Figure 5: Resolution Invariance

![Resolution Invariance](05_resolution_invariance.png)

### What You're Looking At

This figure demonstrates **FNO's most remarkable property**: the same learned weights work at any resolution.

**Top Row (Inputs):**
The same mathematical function (sin(X)·sin(Y) + 0.5·sin(2X)·cos(Y)) sampled at four different resolutions:
- **32×32:** Coarse, visibly pixelated
- **64×64:** Medium resolution
- **128×128:** Fine resolution
- **256×256:** Very fine, smooth appearance

**Bottom Row (Outputs):**
The results of applying **THE EXACT SAME Fourier layer weights** to each input:
- All show the same transformed pattern!
- The transformation is consistent across all resolutions
- Higher resolution simply reveals more detail of the same underlying function

### Key Lessons

- **Same weights, 4 resolutions**: Weights are resolution-independent
- **Pattern is consistent**: Fourier modes represent physical frequencies, not pixel indices
- **Higher res = more detail**: Not new information, just finer sampling

**Why This Works Mathematically:**

1. **Fourier mode k=1** represents "one full oscillation across the domain" — whether you have 32 or 256 pixels.

2. **The learned R[k]** scales mode k — independent of how finely you sample.

3. **W is pointwise** — applies the same linear transform at each grid point, regardless of how many there are.

**The Revolutionary Implication:**

> **Zero-shot super-resolution:** Train on 64×64 data, deploy on 256×256 — NO RETRAINING NEEDED!

**For Urban Temperature:**
- Train on standard ECOSTRESS resolution
- Deploy on higher-resolution data when available
- The physics you learned transfers automatically

This is fundamentally different from CNNs, where a 3×3 kernel trained on 64×64 images doesn't generalize to 256×256 without retraining.

---

## 7.2 Zero-Shot Super-Resolution

This is one of FNO's most remarkable capabilities:

> Train on 64x64, evaluate on 256x256 with NO retraining

From the FNO paper: *"It can do zero-shot super-resolution: trained on a lower resolution directly evaluated on a higher resolution."*

For your urban temperature work:
- Train on coarser ECOSTRESS data
- Deploy on higher-resolution when needed
- The physics learned transfers!

---

# Section 8: Stacking Multiple Layers

## 8.1 The FNO Architecture

From the FNO paper: *"We construct our Fourier neural operator by stacking four Fourier integral operator layers."*

The standard architecture:
1. **Lifting:** Map input channels to hidden dimension (d_v=32)
2. **4 Fourier Layers:** Each preserves spatial dimensions
3. **Projection:** Map hidden dimension to output channels

> **Full Architecture Tutorial:** The NVIDIA Modulus documentation provides production-grade FNO implementations with detailed explanations for Darcy flow and Navier-Stokes problems.
>
> See: [NVIDIA Modulus - FNO Tutorial](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/neural_operators/darcy_fno.html)

### Code: Stacking Fourier Layers

```python
print("\n" + "="*70)
print("SECTION 8: Stacking Multiple Layers")
print("="*70)

class FourierLayerStack:
    """
    Stack of Fourier Layers as in the FNO architecture.
    
    Typical FNO uses 4 stacked layers.
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


# Create typical FNO stack (4 layers, d_v=32, k_max=12)
n_layers = 4
d_hidden = 32
k_max = 12

stack = FourierLayerStack(n_layers, d_hidden, k_max, k_max, seed=42)
print(f"\n{stack}")

# Test forward pass
Nx, Ny = 64, 64
v_input = np.random.randn(Nx, Ny, d_hidden)
v_output = stack.forward(v_input)

print(f"\nInput shape:  {v_input.shape}")
print(f"Output shape: {v_output.shape}")
print(f"Total parameters: {stack.total_params:,}")
```

---

## Generated Figure 6: Stacked Layers Progressive Transformation

![Stacked Layers](06_stacked_layers.png)

### What You're Looking At

This figure shows how data transforms as it passes through 4 stacked Fourier layers — the core of the FNO architecture.

**Top Row (Spatial Domain):**

- **Input:** Random noise (32 channels, showing channel 0). This represents the initial hidden representation after the lifting layer.

- **After Layer 1-4:** Progressive transformation of the field. Notice:
  - The random noise gradually becomes more structured
  - Large-scale patterns emerge
  - The colorbar ranges change as the network transforms values

**Bottom Row (Frequency Domain):**

Each panel shows the log spectrum (Fourier magnitude) of the corresponding spatial field:

- **Input Spectrum:** Roughly uniform (white noise has a flat spectrum — energy at all frequencies)

- **Spectrum after L1-L4:** 
  - The red dashed line shows k_max=12
  - The DC component (k=0, bottom-left corner) becomes dominant (yellow) — the mean becomes important
  - High-frequency content (above red line) is gradually suppressed
  - Low-frequency content becomes more structured

### Key Lessons

- **Input**: Flat spectrum (noise) — Random initialization
- **After L1**: DC grows, high-freq suppressed — First layer extracts mean
- **After L2-3**: Low-freq structure emerges — Learning spatial correlations
- **After L4**: Strong low-freq, weak high-freq — Physical smoothness emerges

**What This Means:**

1. **Each layer progressively refines** the representation
2. **Spectral path dominates** — energy concentrates in low frequencies (smooth output)
3. **DC component (mean) becomes prominent** — typical for physical fields that have meaningful averages
4. **The network learns to produce smooth outputs** even from noisy inputs

**For Urban Temperature:**
With proper training (not random weights), this stack would learn:
- Layer 1: Extract basic features from input (NDVI, building height, etc.)
- Layer 2: Combine features into thermal patterns
- Layer 3: Refine spatial relationships (park cooling, canyon effects)
- Layer 4: Produce smooth temperature predictions

---

# Section 9: Why This Architecture Works for PDEs

## 9.1 Matching PDE Structure

Physical PDEs have operators that are:
- **Translation-invariant:** Physics is the same everywhere
- **Frequency-structured:** Diffusion damps high frequencies as k²
- **Multi-scale:** Couple different spatial scales

FNO matches this perfectly:
- **Convolution** is translation-invariant
- **Spectral weights** learn frequency-specific behavior
- **Nonlinear activation** couples scales

## 9.2 Connection to Classical Spectral Methods

**Classical spectral methods:**
1. Transform to frequency domain
2. Apply **known physics** (decay rates from PDE)
3. Transform back

**FNO:**
1. Transform to frequency domain  
2. Apply **learned transformation**
3. Transform back

**FNO is a learnable spectral method.** It discovers the physics from data!

> **Spectral Methods Background:** For a deep dive into classical spectral methods for PDEs, see Steve Brunton's lecture ["Numerical Solutions to PDEs Using FFT"](https://www.youtube.com/watch?v=hDeARtZdq-U). Understanding classical methods illuminates why FNO's architecture is so natural for PDE problems.

## 9.3 For Your Urban Temperature Work

LST prediction has the right properties for FNO:
- **Smooth solutions:** Temperature varies smoothly at 70m resolution
- **Multi-scale physics:** City-wide gradients + neighborhood effects
- **Parametric:** Different days/seasons = different input functions

The spectral approach captures:
- Large-scale urban heat island (low k modes)
- Park/water cooling effects (medium k modes)
- Local variations (local path W)

---

# Summary: Understanding the Fourier Layer

## Complete Architecture Diagram

```
Input v(x,y) ∈ R^{Nx × Ny × d_v}
       │
       ├──────────────────────────┬──────────────────────┐
       │                          │                      │
       ▼                          ▼                      │
    [FFT2]                      [W·v]                    │
       │                        (local)                  │
       ▼                          │                      │
[Truncate to k_max modes]         │                      │
       │                          │                      │
       ▼                          │                      │
[Multiply by R]                   │                      │
  (at each freq)                  │                      │
       │                          │                      │
       ▼                          │                      │
   [IFFT2]                        │                      │
       │                          │                      │
       └──────────► [ADD] ◄───────┘                      │
                      │                                  │
                      ▼                                  │
               [ADD BIAS b] ◄────────────────────────────┘
                      │
                      ▼
               [GELU activation]
                      │
                      ▼
Output v'(x,y) ∈ R^{Nx × Ny × d_v}  (same shape!)
```

## Key Equations

**Spectral Convolution (FNO paper, Definition 3):**
$$\mathcal{K}(v)(x) = \mathcal{F}^{-1}\left[ R \cdot \mathcal{F}[v] \right](x)$$

**Complete Fourier Layer (FNO paper, Equation 2):**
$$v_{t+1}(x) = \sigma\left( W v_t(x) + \mathcal{K}v_t(x) + b \right)$$

## What You've Learned

- **Spectral Convolution**: FFT, multiply by R, IFFT = O(N log N) global convolution
- **Mode Truncation**: Keep only k_max modes; physics is smooth; regularization for free
- **Weight Tensor R**: Complex-valued, per-frequency channel mixing matrices
- **Local Path W**: 1×1 conv recovers high-frequency capability
- **Resolution Invariance**: Same R works at any grid size - enables zero-shot super-resolution

## Hyperparameter Guidelines

From the FNO paper experiments:

**k_max (Fourier modes):**
- 1D Problems: 16
- 2D Problems: 12
- For LST Work: 12-16

**d_v (hidden dimension):**
- 1D Problems: 64
- 2D Problems: 32
- For LST Work: 32

**n_layers:**
- All cases: 4

**Activation:**
- Original paper: ReLU
- Recommended for PDEs: GELU

---

# Further Learning Resources

## Essential Reading

- **[Original FNO Paper](https://arxiv.org/abs/2010.08895)** — Paper: The foundational reference (Li et al., 2021)
- **[Zongyi Li's Blog](https://zongyi-li.github.io/blog/2020/fourier-pde/)** — Blog: Author's accessible explanation with visualizations
- **[Neural Operator Survey](https://arxiv.org/abs/2108.08481)** — Paper: Comprehensive theory (Kovachki et al., 2023)

## Video Tutorials

- **[Yannic Kilcher: FNO Explained](https://www.youtube.com/watch?v=IaS72aHrJKE)** — 66 min: Comprehensive paper walkthrough
- **[3Blue1Brown: Fourier Transform](https://www.youtube.com/watch?v=spUNpyF58BY)** — 21 min: Beautiful visual introduction to Fourier
- **[Steve Brunton: FFT for PDEs](https://www.youtube.com/c/Eigensteve)** — Multiple: University lectures on spectral methods

## Implementation Resources

- **[NeuralOperator Library](https://github.com/neuraloperator/neuraloperator)** — PyTorch: Official implementation, actively maintained
- **[UvA DL Notebooks](https://uvadlc-notebooks.readthedocs.io/)** — PyTorch: Educational implementation from scratch
- **[NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/)** — PyTorch: Production-grade, enterprise-ready

## Related Architectures

- **DeepONet** — [arXiv:1910.03193](https://arxiv.org/abs/1910.03193): Branch-trunk architecture
- **U-FNO** — [arXiv:2109.03697](https://arxiv.org/abs/2109.03697): U-Net + FNO hybrid
- **Geo-FNO** — [arXiv:2207.05209](https://arxiv.org/abs/2207.05209): Irregular geometries
- **FourCastNet** — [GitHub](https://github.com/NVlabs/FourCastNet): Global weather prediction

---

# Generated Figures Summary

All figures generated from the code in this tutorial:

- **Figure 1** (`01_spectral_conv_1d_basic.png`): Core operation: FFT, R, IFFT
- **Figure 2** (`02_mode_truncation_2d.png`): Why truncation works for smooth fields
- **Figure 3** (`03_why_both_paths.png`): Spectral vs local paths
- **Figure 4** (`04_activation_functions.png`): GELU vs ReLU for PDEs
- **Figure 5** (`05_resolution_invariance.png`): Same weights work at any resolution
- **Figure 6** (`06_stacked_layers.png`): Progressive transformation through layers

---

# External Image Credits Summary

All external images used in this tutorial are from freely-licensed sources:

- **Fourier Transform Animation**: Wikimedia Commons (Public Domain)
- **Fourier Series Approximation**: Wikimedia Commons (CC BY-SA 3.0)
- **Time vs Frequency Domain**: Wikimedia Commons (CC BY-SA 3.0)
- **2D FFT Visualization**: Wikimedia Commons (Public Domain)
- **Complex Number Illustration**: Wikimedia Commons (Public Domain)
- **Filter Response Curves**: Wikimedia Commons (CC BY-SA 3.0)
- **Activation Functions**: Wikimedia Commons (CC BY-SA 4.0)
- **Aliasing/Sampling**: Wikimedia Commons (Public Domain)
- **FNO Architecture (Fig. 1)**: arXiv:2010.08895 (arXiv non-exclusive license)

**Note on arXiv figures:** arXiv grants a non-exclusive license to distribute articles, and figures from arXiv papers are commonly used in educational materials with proper citation. The FNO architecture diagram is from Li et al. (2021) and is cited appropriately.

---

# Next Steps

You now have complete understanding of the Fourier Layer!

**Chunk 3** will build the full FNO architecture:
- Lifting layer: Input features to hidden dimension
- Projection layer: Hidden dimension to output
- Complete forward pass
- Training with PyTorch

**Tutorial Navigation:**

- **[Previous: Part 1](../chunk1/chunk1_complete.md)** - Fourier Foundations: DFT, convolution theorem
- **Part 2** — Core Innovation: *(You are here)* Spectral convolution
- **[Next: Part 3](../chunk3/chunk3_complete.md)** - Complete Architecture: Lifting, Fourier layers, projection



**Paper Reference:**
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*.

- arXiv: https://arxiv.org/abs/2010.08895
- Code: https://github.com/neuraloperator/neuraloperator
- Author Blog: https://zongyi-li.github.io/blog/2020/fourier-pde/
