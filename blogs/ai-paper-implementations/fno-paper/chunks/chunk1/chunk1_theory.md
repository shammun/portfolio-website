# Fourier Neural Operator: Chunk 1
## Mathematical Foundations - Complete Expert-Level Tutorial

---

# Introduction

This tutorial covers **ONLY** the mathematical foundations you need before implementing FNO. By the end, you will have deep understanding of:

1. **What operators are** and why they differ from functions
2. **The Fourier Transform** from first principles
3. **Key properties** that make spectral methods powerful for PDEs

We will NOT implement FNO components yet—that's Chunk 2. Here we build the mathematical intuition that makes everything else make sense.

---

# Section 1: Functions vs Operators

## 1.1 What Neural Networks Traditionally Learn

A standard neural network learns a **function**:

$$f_\theta: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

This maps a finite-dimensional input to a finite-dimensional output.

**Examples:**

- **Image classification**
  - Input: 224×224×3 image
  - Output: 1000 class probabilities
  - Mapping: $\mathbb{R}^{150528} \rightarrow \mathbb{R}^{1000}$

- **House price prediction**
  - Input: 10 features
  - Output: 1 price
  - Mapping: $\mathbb{R}^{10} \rightarrow \mathbb{R}$

- **Random Forest LST model**
  - Input: 42 features at one pixel
  - Output: 1 temperature
  - Mapping: $\mathbb{R}^{42} \rightarrow \mathbb{R}$

**The critical limitation:** The dimensions are **fixed**. If you train on 64×64 grids, you cannot directly use the model on 128×128 grids.

## 1.2 What is an Operator?

An **operator** maps between **function spaces**:

$$\mathcal{G}: \mathcal{A} \rightarrow \mathcal{U}$$

Where $\mathcal{A}$ and $\mathcal{U}$ are spaces of functions (infinite-dimensional).

**The key difference:**
- Function: Takes a **vector** (finite list of numbers), outputs a **vector**
- Operator: Takes an **entire function** (infinite information), outputs an **entire function**

## 1.3 Concrete Example: The Heat Equation

Consider the 1D heat equation on a rod:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Where:
- $u(x, t)$ = temperature at position $x$ and time $t$
- $\alpha$ = thermal diffusivity (material property)

**The solution operator $\mathcal{G}_T$:**

Given initial temperature distribution $a(x) = u(x, 0)$, the solution at time $T$ is:

$$u(x, T) = \mathcal{G}_T[a](x)$$

This operator:
- **Input:** The function $a(x)$ — temperature everywhere along the rod at $t=0$
- **Output:** The function $u(x,T)$ — temperature everywhere along the rod at $t=T$

It's not mapping a few numbers to a few numbers. It's mapping an **entire temperature profile** to another **entire temperature profile**.

## 1.4 Why This Matters for Your Research

In your urban LST prediction:

**Traditional ML approach (what you did):**
- Input: 42 features at ONE pixel location
- Output: Temperature at that ONE pixel
- You process each pixel independently

**Operator learning approach:**
- Input: Entire spatial field of features over NYC (a function over 2D space)
- Output: Entire LST field over NYC (another function over 2D space)
- The model learns spatial relationships globally

The operator approach can capture how temperature at one location depends on features at distant locations (e.g., a park cooling nearby buildings).

## 1.5 Discretization and the Resolution Problem

In practice, we can't work with continuous functions. We discretize:

**Continuous function:** $a(x)$ for $x \in [0, 1]$

**Discretized representation:** Sample at $N$ points
$$\mathbf{a} = [a(x_0), a(x_1), ..., a(x_{N-1})]$$

**The problem with standard neural networks:**

If you train a CNN with input size 64×64:
- The first layer expects exactly 64×64 inputs
- Weight matrices have fixed dimensions tied to 64×64
- You **cannot** apply it to 128×128 without modification

The network learned:
$$f_\theta: \mathbb{R}^{64 \times 64} \rightarrow \mathbb{R}^{64 \times 64}$$

This is NOT the continuous operator $\mathcal{G}$. It's a resolution-specific approximation.

## 1.6 Discretization Invariance: The Holy Grail

A true neural operator should be **discretization invariant**:

> As the discretization becomes finer, the discrete approximation should converge to the same continuous operator, regardless of the specific discretization used.

**Practical implications:**
1. Train on 64×64 (fast, cheap) → Evaluate on 256×256 (accurate, detailed)
2. Train on regular grids → Apply to irregular sensor locations
3. The model learns **physics**, not **grid artifacts**

**This is what FNO achieves** through spectral methods, which we'll understand after learning Fourier transforms.

## 1.7 Summary: Functions vs Operators

**Function:**
- Input: Vector $\mathbf{x} \in \mathbb{R}^n$
- Output: Vector $\mathbf{y} \in \mathbb{R}^m$
- Dimensions: Fixed at training time
- What it learns: Pattern in specific discretization
- Example: CNN classifier

**Operator:**
- Input: Function $a: \Omega \rightarrow \mathbb{R}$
- Output: Function $u: \Omega \rightarrow \mathbb{R}$
- Dimensions: Resolution-independent
- What it learns: Underlying continuous mapping
- Example: PDE solution operator

---

# Section 2: The Fourier Transform

## 2.1 The Core Idea

**Any function can be written as a sum of sinusoids.**

This is one of the most powerful ideas in mathematics and physics.

A complex-looking function like temperature variation over a day can be decomposed into:
- A constant (average temperature)
- A slow oscillation (day/night cycle)
- Faster oscillations (hourly variations)
- Even faster oscillations (minute-by-minute fluctuations)

The Fourier transform tells us **how much** of each frequency is present.

## 2.2 Building Intuition: Fourier Series

Before the transform, let's understand Fourier series.

For a periodic function $f(x)$ with period $L$:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{2\pi n x}{L}\right) + b_n \sin\left(\frac{2\pi n x}{L}\right) \right]$$

Where:
- $a_0/2$ = average value (DC component)
- $a_n, b_n$ = amplitudes of the $n$-th harmonic
- $n$ = harmonic number (frequency = $n/L$ cycles per unit length)

**The coefficients are computed by:**

$$a_n = \frac{2}{L} \int_0^L f(x) \cos\left(\frac{2\pi n x}{L}\right) dx$$

$$b_n = \frac{2}{L} \int_0^L f(x) \sin\left(\frac{2\pi n x}{L}\right) dx$$

**Physical interpretation:** We're asking "how much does $f(x)$ look like a cosine/sine of frequency $n$?"

## 2.3 Complex Exponential Form

Using Euler's formula: $e^{i\theta} = \cos(\theta) + i\sin(\theta)$

We can write the Fourier series more elegantly:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i \frac{2\pi n x}{L}}$$

Where $c_n$ are complex coefficients containing both amplitude and phase information.

**Why complex exponentials?**
1. Mathematically cleaner (one formula instead of separate sin/cos)
2. Derivatives are simple: $\frac{d}{dx} e^{ikx} = ik \cdot e^{ikx}$
3. Products are simple: $e^{ik_1 x} \cdot e^{ik_2 x} = e^{i(k_1+k_2)x}$

## 2.4 The Continuous Fourier Transform

For non-periodic functions on the entire real line:

**Forward Transform:**
$$\hat{f}(k) = \mathcal{F}[f](k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} dx$$

**Inverse Transform:**
$$f(x) = \mathcal{F}^{-1}[\hat{f}](x) = \int_{-\infty}^{\infty} \hat{f}(k) e^{2\pi i k x} dk$$

**Notation:**
- $f(x)$ = function in **spatial/time domain**
- $\hat{f}(k)$ = function in **frequency/spectral domain**
- $k$ = frequency (or wavenumber for spatial signals)

**What $\hat{f}(k)$ tells us:**
- $|\hat{f}(k)|$ = amplitude of frequency component $k$
- $\arg(\hat{f}(k))$ = phase of frequency component $k$

## 2.5 The Discrete Fourier Transform (DFT)

For a discrete signal $f = [f_0, f_1, ..., f_{N-1}]$ sampled at $N$ points:

**Forward DFT:**
$$\hat{f}_k = \sum_{n=0}^{N-1} f_n \cdot e^{-2\pi i \frac{kn}{N}}, \quad k = 0, 1, ..., N-1$$

**Inverse DFT:**
$$f_n = \frac{1}{N} \sum_{k=0}^{N-1} \hat{f}_k \cdot e^{2\pi i \frac{kn}{N}}, \quad n = 0, 1, ..., N-1$$

**Understanding the formula:**

The term $e^{-2\pi i \frac{kn}{N}}$ is a complex sinusoid that completes exactly $k$ cycles over $N$ points.

- When $k=0$: $e^0 = 1$ for all $n$ → measures the sum (DC component)
- When $k=1$: One complete cycle → lowest non-zero frequency
- When $k=N/2$: Nyquist frequency → highest representable frequency

## 2.6 Frequency Interpretation

For a signal of length $N$ with sampling interval $\Delta x$:

**Frequency indices and their meaning:**

- **k = 0**: Frequency 0 — DC (mean value)
- **k = 1**: Frequency $\frac{1}{N\Delta x}$ — Lowest frequency, wavelength = domain size
- **k = 2**: Frequency $\frac{2}{N\Delta x}$ — Second harmonic
- **...**
- **k = N/2**: Frequency $\frac{1}{2\Delta x}$ — Nyquist frequency (highest)
- **k = N/2+1 to N-1**: Negative frequencies — Aliased with positive

**Nyquist theorem:** You can only accurately represent frequencies up to half the sampling rate.

## 2.7 Real FFT (RFFT)

For **real-valued** signals (like temperature), the Fourier transform has **conjugate symmetry**:

$$\hat{f}_{N-k} = \hat{f}_k^*$$

This means negative frequency components are redundant—they're just complex conjugates of positive frequencies.

**Real FFT** only computes the non-redundant half:
$$\hat{f}_0, \hat{f}_1, ..., \hat{f}_{N/2}$$

**Benefits:**
- Half the storage: $N/2 + 1$ complex values instead of $N$
- Half the computation
- No information lost

**This is what we use in FNO for real-valued fields like temperature.**

## 2.8 2D Fourier Transform

For 2D fields (like your NYC temperature maps):

$$\hat{f}(k_x, k_y) = \sum_{n_x=0}^{N_x-1} \sum_{n_y=0}^{N_y-1} f_{n_x, n_y} \cdot e^{-2\pi i \left(\frac{k_x n_x}{N_x} + \frac{k_y n_y}{N_y}\right)}$$

**Shape transformation:**
- Input: $(N_x, N_y)$ real values
- Output of `fft2`: $(N_x, N_y)$ complex values
- Output of `rfft2`: $(N_x, N_y//2 + 1)$ complex values

**Physical interpretation for 2D:**
- $k_x$ = spatial frequency in x-direction
- $k_y$ = spatial frequency in y-direction
- Low $(k_x, k_y)$ = smooth, large-scale patterns (like regional temperature gradients)
- High $(k_x, k_y)$ = fine details (like building-scale variations)

---

# Section 3: Key Properties of the Fourier Transform

These properties are **why FNO works**. Understand them deeply.

## 3.1 Linearity

$$\mathcal{F}[\alpha f + \beta g] = \alpha \mathcal{F}[f] + \beta \mathcal{F}[g]$$

The Fourier transform of a sum is the sum of Fourier transforms.

**Why it matters:** We can analyze complex signals by breaking them into simpler components.

## 3.2 Differentiation Property ⭐

**This is crucial for PDEs.**

$$\mathcal{F}\left[\frac{df}{dx}\right] = 2\pi i k \cdot \hat{f}(k)$$

**Differentiation in spatial domain = multiplication by $ik$ in frequency domain.**

For second derivatives:
$$\mathcal{F}\left[\frac{d^2f}{dx^2}\right] = -(2\pi k)^2 \cdot \hat{f}(k)$$

**Why this is powerful:**

The heat equation $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$ becomes:

$$\frac{\partial \hat{u}}{\partial t} = -\alpha (2\pi k)^2 \hat{u}$$

A **partial differential equation** (hard) becomes an **ordinary differential equation** (easy) for each frequency!

**Numerical implications:**
- Finite differences: Local, approximate, accumulates errors
- Spectral derivatives: Global, exact (for smooth functions), machine precision

## 3.3 Convolution Theorem ⭐⭐

**This is THE key insight for FNO.**

$$\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$$

**Convolution in spatial domain = multiplication in frequency domain.**

Where convolution is:
$$(f * g)(x) = \int f(y) g(x-y) dy$$

**Why this matters for neural networks:**

A convolution layer in a CNN:
- Spatial domain: Slide kernel over input, compute dot products → $O(N \cdot K)$ per output point, $O(N^2 \cdot K)$ total for large kernels
- Frequency domain: FFT → multiply → IFFT → $O(N \log N)$ regardless of kernel size

**For global convolutions (large receptive fields):**
- Spatial: Prohibitively expensive
- Spectral: Same cost as local convolutions

**FNO insight:** Instead of learning a spatial kernel and doing expensive convolution, learn weights directly in frequency domain and just multiply!

## 3.4 Parseval's Theorem (Energy Conservation)

$$\int |f(x)|^2 dx = \int |\hat{f}(k)|^2 dk$$

Energy in spatial domain equals energy in frequency domain.

**Implication:** Truncating high-frequency modes removes energy associated with fine-scale fluctuations but preserves energy in smooth structures.

## 3.5 Shift Property

$$\mathcal{F}[f(x - x_0)] = e^{-2\pi i k x_0} \cdot \hat{f}(k)$$

Shifting in space = phase shift in frequency.

**Why it matters:** The Fourier magnitude $|\hat{f}(k)|$ is translation-invariant. The same pattern shifted gives the same magnitude spectrum.

---

# Section 4: Why Spectral Methods Work for PDEs

## 4.1 PDEs Define Operators

Most PDEs can be written as:
$$\mathcal{L}[u] = f$$

Where $\mathcal{L}$ is a differential operator.

**Examples:**

- **Heat equation**: Operator $\mathcal{L} = \frac{\partial}{\partial t} - \alpha \nabla^2$
- **Wave equation**: Operator $\mathcal{L} = \frac{\partial^2}{\partial t^2} - c^2 \nabla^2$
- **Poisson equation**: Operator $\mathcal{L} = -\nabla^2$

## 4.2 Solving PDEs in Fourier Space

**Example: Heat equation**

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Take Fourier transform in $x$:
$$\frac{\partial \hat{u}}{\partial t} = \alpha \cdot (-(2\pi k)^2) \cdot \hat{u} = -\alpha (2\pi k)^2 \hat{u}$$

This is an ODE for each wavenumber $k$! Solution:
$$\hat{u}(k, t) = \hat{u}(k, 0) \cdot e^{-\alpha (2\pi k)^2 t}$$

**Key observation:** Each Fourier mode evolves **independently** and **exponentially decays** at a rate proportional to $k^2$.

- Low-$k$ modes (smooth patterns): Decay slowly
- High-$k$ modes (fine details): Decay rapidly

This is why heat smooths out sharp features—high frequencies die quickly.

## 4.3 The Spectral Solution Method

1. Start with initial condition $u(x, 0)$
2. Transform: $\hat{u}(k, 0) = \mathcal{F}[u(\cdot, 0)]$
3. Evolve each mode: $\hat{u}(k, t) = \hat{u}(k, 0) \cdot G(k, t)$
4. Transform back: $u(x, t) = \mathcal{F}^{-1}[\hat{u}(\cdot, t)]$

Where $G(k, t)$ is the **transfer function** or **Green's function** in Fourier space.

## 4.4 Connection to Neural Operators

**What if we don't know the physics?**

Instead of deriving $G(k, t)$ from the PDE, we can **learn it from data**.

The FNO learns weights $R(k)$ that act like a learned transfer function:
- Input: $\hat{v}(k)$ — Fourier coefficients of input function
- Operation: $\hat{w}(k) = R(k) \cdot \hat{v}(k)$ — learned transformation
- Output: $w(x) = \mathcal{F}^{-1}[\hat{w}]$ — back to spatial domain

**Why this works:**
1. Physical solutions are smooth → dominated by low frequencies
2. Global dependencies → captured by spectral (global) operations
3. Each frequency can be transformed independently → efficient

## 4.5 Why Truncate to Low Frequencies?

In FNO, we only keep the first $k_{max}$ modes. Why?

**Physical reasons:**
1. PDE solutions are typically smooth (high frequencies are small)
2. High-frequency components often represent noise, not signal
3. Heat/diffusion processes naturally damp high frequencies

**Computational reasons:**
1. Reduces parameters: $O(k_{max})$ instead of $O(N)$
2. Acts as regularization (prevents overfitting to noise)
3. Enables resolution invariance (low-freq modes exist at any resolution)

**For your urban temperature:**
- Low frequencies: Regional gradients, large park effects, water body influence
- High frequencies: Building-scale variations, sensor noise
- FNO focuses on the smooth, learnable patterns

---

# Section 5: Practical Numerical Considerations

## 5.1 FFT vs DFT

The **Fast Fourier Transform (FFT)** computes the DFT in $O(N \log N)$ instead of $O(N^2)$.

It's not a different transform—just a clever algorithm exploiting symmetries.

**Always use FFT libraries (numpy.fft, torch.fft), never implement DFT directly for real applications.**

## 5.2 Normalization Conventions

Different libraries use different normalizations:

**Normalization options:**

- **NumPy default**: Forward has no factor, Inverse divides by $N$
- **Symmetric**: Both Forward and Inverse divide by $\sqrt{N}$
- **Physics convention**: Both include $2\pi$ factors

**For FNO:** Consistency matters more than convention. NumPy/PyTorch defaults work fine.

## 5.3 Frequency Ordering

NumPy's `fft` returns frequencies in a specific order:

```
[0, 1, 2, ..., N/2-1, -N/2, -N/2+1, ..., -1]
```

- First half: Non-negative frequencies (0 to Nyquist)
- Second half: Negative frequencies

Use `fftfreq()` to get the actual frequency values.

## 5.4 Complex Numbers in Python

```python
# Creating complex numbers
z = 3 + 4j
z = complex(3, 4)

# Real and imaginary parts
z.real  # 3.0
z.imag  # 4.0

# Magnitude and phase
abs(z)         # 5.0 (magnitude)
np.angle(z)    # 0.927... (phase in radians)

# Complex conjugate
z.conjugate()  # 3 - 4j
```

---

# Summary: What You Now Understand

## Conceptual Understanding

1. **Operators vs Functions**
   - Functions map vectors to vectors (fixed dimensions)
   - Operators map functions to functions (infinite-dimensional)
   - Neural operators learn resolution-invariant mappings

2. **Fourier Transform**
   - Decomposes functions into frequency components
   - Any function = sum of sinusoids at different frequencies
   - Transforms between spatial and spectral domains

3. **Why Fourier for PDEs**
   - Derivatives become multiplications (huge simplification)
   - Convolutions become pointwise products (huge speedup)
   - PDE solutions are smooth (low frequencies dominate)

## Mathematical Tools

1. **Forward transform:** $\hat{f}_k = \sum_n f_n e^{-2\pi i kn/N}$
2. **Inverse transform:** $f_n = \frac{1}{N} \sum_k \hat{f}_k e^{2\pi i kn/N}$
3. **Differentiation:** $\mathcal{F}[f'] = 2\pi ik \cdot \hat{f}$
4. **Convolution:** $\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]$

## Connection to FNO (Preview)

FNO will use these ideas:
- Learn weights $R(k)$ in Fourier space
- Apply them: $\hat{w} = R \cdot \hat{v}$
- Only keep low frequencies (mode truncation)
- Combine with local operations for full spectral coverage

---

# Next Step

Now proceed to the code file: `chunk1_code.py`

This implements everything from this tutorial so you gain practical mastery alongside theoretical understanding.
