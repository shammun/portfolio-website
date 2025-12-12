# Fourier Neural Operator: Chunk 3
## The Complete FNO Architecture — Expert-Level Tutorial

---

# Introduction

In Chunks 1 and 2, you mastered:
- **Chunk 1:** Fourier transforms, convolution theorem, spectral methods for PDEs
- **Chunk 2:** Spectral convolution, mode truncation, the Fourier layer (R + W + b + GELU)

Now we assemble everything into the **complete Fourier Neural Operator**.

By the end of this chunk, you will deeply understand:
1. The full FNO architecture from input to output
2. The lifting layer P (input → hidden representation)
3. The projection layer Q (hidden → output)
4. Why we stack multiple Fourier layers
5. Training FNO: loss functions, optimization, and practical considerations
6. Hyperparameter selection for your urban temperature problem
7. Connection to the original Li et al. (2020) paper

**No code in this document** — pure conceptual mastery. Code implementation follows.

---

# Section 1: The Big Picture — What FNO Does

## 1.1 The Operator Learning Problem

Recall from Chunk 1: we want to learn an **operator** that maps functions to functions.

**Your problem:** Given input features (NDVI, building height, ERA5 temp, etc.) at every spatial location, predict the temperature field.

$$\mathcal{G}: a(x) \mapsto u(x)$$

Where:
- $a(x) \in \mathbb{R}^{d_a}$ is the input function (your 42 features at each point)
- $u(x) \in \mathbb{R}^{d_u}$ is the output function (temperature at each point)
- $\mathcal{G}$ is the operator we want to learn

## 1.2 The FNO Solution

FNO approximates this operator through a sequence of transformations:

$$a(x) \xrightarrow{P} v_0(x) \xrightarrow{\mathcal{L}_1} v_1(x) \xrightarrow{\mathcal{L}_2} v_2(x) \xrightarrow{\mathcal{L}_3} v_3(x) \xrightarrow{\mathcal{L}_4} v_4(x) \xrightarrow{Q} u(x)$$

Three stages:
1. **Lifting (P):** Project input to higher-dimensional hidden space
2. **Fourier Layers (𝓛₁...𝓛₄):** Iteratively transform through spectral convolutions
3. **Projection (Q):** Map back to output dimension

## 1.3 Why This Architecture?

**Lifting (P):** Your 42 features need to be combined and transformed into a richer representation. Think of it as creating "meta-features" that the Fourier layers can work with.

**Fourier Layers:** Each layer captures different aspects of the spatial relationships. Stacking allows learning complex, nonlinear mappings while maintaining global receptive field.

**Projection (Q):** The hidden representation needs to be decoded back to the physical quantity (temperature).

## 1.4 The Complete Architecture Diagram

```
INPUT: a(x) ∈ ℝ^(Nx × Ny × d_a)
       │
       │ (e.g., 42 features: NDVI, building height, ERA5, etc.)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  LIFTING LAYER P                                              │
│  ─────────────────                                            │
│  Linear transformation at each point                          │
│  P: ℝ^d_a → ℝ^d_v                                            │
│                                                               │
│  v₀(x) = P · a(x) + b_P                                       │
│                                                               │
│  (42 features → 64 hidden channels, for example)              │
└──────────────────────────────────────────────────────────────┘
       │
       │ v₀(x) ∈ ℝ^(Nx × Ny × d_v)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FOURIER LAYER 1                                              │
│  ───────────────                                              │
│  v₁ = GELU(W₁v₀ + K₁v₀ + b₁)                                 │
│                                                               │
│  where K₁v₀ = IFFT(R₁ · FFT(v₀))  [truncated to k_max modes] │
└──────────────────────────────────────────────────────────────┘
       │
       │ v₁(x) ∈ ℝ^(Nx × Ny × d_v)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FOURIER LAYER 2                                              │
│  ───────────────                                              │
│  v₂ = GELU(W₂v₁ + K₂v₁ + b₂)                                 │
└──────────────────────────────────────────────────────────────┘
       │
       │ v₂(x) ∈ ℝ^(Nx × Ny × d_v)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FOURIER LAYER 3                                              │
│  ───────────────                                              │
│  v₃ = GELU(W₃v₂ + K₃v₂ + b₃)                                 │
└──────────────────────────────────────────────────────────────┘
       │
       │ v₃(x) ∈ ℝ^(Nx × Ny × d_v)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FOURIER LAYER 4                                              │
│  ───────────────                                              │
│  v₄ = GELU(W₄v₃ + K₄v₃ + b₄)                                 │
└──────────────────────────────────────────────────────────────┘
       │
       │ v₄(x) ∈ ℝ^(Nx × Ny × d_v)
       ▼
┌──────────────────────────────────────────────────────────────┐
│  PROJECTION LAYER Q                                           │
│  ──────────────────                                           │
│  Two-layer MLP at each point:                                 │
│                                                               │
│  h(x) = GELU(Q₁ · v₄(x) + b_Q1)                              │
│  u(x) = Q₂ · h(x) + b_Q2                                      │
│                                                               │
│  (64 hidden → 128 intermediate → 1 output, for example)       │
└──────────────────────────────────────────────────────────────┘
       │
       │ u(x) ∈ ℝ^(Nx × Ny × d_u)
       ▼
OUTPUT: Predicted temperature field
```

---

# Section 2: The Lifting Layer (P)

## 2.1 Purpose of Lifting

The input $a(x)$ has $d_a$ channels (your 42 features). But Fourier layers work best with a richer, higher-dimensional representation.

**Lifting** transforms the input into a hidden space:
$$v_0(x) = P \cdot a(x) + b_P$$

Where:
- $P \in \mathbb{R}^{d_a \times d_v}$ is the lifting matrix
- $b_P \in \mathbb{R}^{d_v}$ is the bias
- $d_v$ is the hidden dimension (typically 32, 64, or 128)

## 2.2 What Lifting Does Conceptually

Think of lifting as **feature engineering learned by the network**:

1. **Combines related features:** NDVI and tree canopy might be combined into a "vegetation effect" channel
2. **Creates orthogonal representations:** Different aspects of the input get separated
3. **Expands dimensionality:** More channels = more capacity to represent complex patterns

## 2.3 Lifting is Pointwise

Critical: Lifting operates **independently at each spatial location**.

At point $(x, y)$:
$$v_0(x, y) = P \cdot a(x, y) + b_P$$

This is a simple matrix-vector multiplication. No spatial context yet — that comes from the Fourier layers.

## 2.4 Why Not Start with Fourier Layers Directly?

You could apply Fourier layers directly to the 42-channel input. But:

1. **Dimension mismatch:** Fourier layers typically use $d_v = d_{out}$ (same input/output channels). Your input (42) and output (1) don't match.

2. **Feature mixing:** Raw features may not be in the best form for spectral processing. Lifting creates a learned representation optimized for the Fourier layers.

3. **Computational efficiency:** It's cheaper to do spectral convolution on 64 channels than 42 → 64 → ... → 1 with changing dimensions.

## 2.5 Lifting for Your Problem

**Input:** 42 features at each 70m pixel
- NDVI, NDBI, NDWI
- Building height, density, SVF
- ERA5 temperature, humidity, wind
- Distance features, land cover, etc.

**Lifting:** 42 → 64 (or 32, depending on model size)

The network learns which combinations of these 42 features are useful for predicting temperature.

---

# Section 3: Stacking Fourier Layers

## 3.1 Why Multiple Layers?

A single Fourier layer applies:
$$v_1 = \sigma(Wv_0 + \mathcal{K}v_0 + b)$$

This is powerful but limited:
- Only one level of nonlinearity
- Limited compositional complexity
- Can't learn hierarchical features

Stacking layers enables:
- **Deeper nonlinear transformations**
- **Hierarchical feature learning**
- **More expressive mappings**

## 3.2 What Each Layer Contributes

Think of the layers as progressive refinement:

- **Layer 1**: Basic spatial patterns, initial feature combinations
- **Layer 2**: Interactions between Layer 1 patterns
- **Layer 3**: Higher-order relationships, complex spatial dependencies
- **Layer 4**: Final refinement, output-ready representation

## 3.3 How Many Layers?

**Standard choice:** 4 Fourier layers

**Why 4?**
- Empirically works well across many PDE problems
- Enough depth for complex mappings
- Not so deep that training becomes difficult
- Balances expressivity and computational cost

**When to use more/fewer:**
- Simpler problems (e.g., linear PDEs): 2-3 layers may suffice
- Very complex problems: 6-8 layers possible but diminishing returns
- Limited data: Fewer layers to prevent overfitting

## 3.4 Parameter Sharing vs. Independent Weights

**Standard FNO:** Each layer has its own weights $(R_l, W_l, b_l)$

**Alternative:** Share weights across layers (like RNNs)
- Fewer parameters
- Stronger regularization
- Less common in practice

For your problem, independent weights per layer is recommended.

## 3.5 Residual Connections (Optional but Beneficial)

Some FNO variants add skip connections:
$$v_{l+1} = \sigma(W_l v_l + \mathcal{K}_l v_l + b_l) + v_l$$

Benefits:
- Easier gradient flow during training
- Can learn identity mapping (if layer not needed)
- Often improves performance

The original FNO paper doesn't use residual connections in Fourier layers, but many implementations add them.

---

# Section 4: The Projection Layer (Q)

## 4.1 Purpose of Projection

After 4 Fourier layers, we have $v_4(x) \in \mathbb{R}^{d_v}$ at each point.

But we need output $u(x) \in \mathbb{R}^{d_u}$ (temperature = 1 channel).

**Projection** maps from hidden space to output:
$$u(x) = Q(v_4(x))$$

## 4.2 Two-Layer MLP Structure

The projection is typically a small MLP applied pointwise:

**Layer 1:**
$$h(x) = \text{GELU}(Q_1 \cdot v_4(x) + b_{Q1})$$

**Layer 2:**
$$u(x) = Q_2 \cdot h(x) + b_{Q2}$$

Where:
- $Q_1 \in \mathbb{R}^{d_v \times d_{mid}}$ — first projection matrix
- $Q_2 \in \mathbb{R}^{d_{mid} \times d_u}$ — second projection matrix
- $d_{mid}$ — intermediate dimension (often = $d_v$ or $d_v \times 2$)

## 4.3 Why Two Layers?

A single linear projection ($u = Q \cdot v_4$) would work but is limited:

1. **Nonlinearity:** The GELU between Q₁ and Q₂ adds final nonlinear transformation
2. **Expressivity:** Two layers can approximate more complex mappings
3. **Bottleneck/expansion:** Can have $d_{mid} > d_v$ or $d_{mid} < d_v$ for different effects

## 4.4 Projection is Pointwise

Like lifting, projection operates independently at each spatial location:

$$u(x, y) = Q_2 \cdot \text{GELU}(Q_1 \cdot v_4(x, y) + b_{Q1}) + b_{Q2}$$

All spatial mixing has already happened in the Fourier layers.

## 4.5 No Activation on Final Output

Note: The final output $u(x)$ has **no activation function**.

Why?
- Temperature can be any real value
- Activation would constrain the range (ReLU → positive only, sigmoid → [0,1])
- We want unconstrained prediction for regression

## 4.6 Projection for Your Problem

**Hidden dimension:** $d_v = 64$
**Intermediate:** $d_{mid} = 128$ (expansion)
**Output:** $d_u = 1$ (temperature)

$$64 \xrightarrow{Q_1} 128 \xrightarrow{\text{GELU}} 128 \xrightarrow{Q_2} 1$$

---

# Section 5: The Complete Forward Pass

## 5.1 Full Mathematical Specification

**Input:** $a \in \mathbb{R}^{N_x \times N_y \times d_a}$

**Lifting:**
$$v_0 = a \cdot P + b_P \quad \text{(pointwise)}$$

**Fourier Layers (for $l = 1, 2, 3, 4$):**
$$v_l = \text{GELU}\left( v_{l-1} \cdot W_l + \mathcal{F}^{-1}[R_l \cdot \mathcal{F}[v_{l-1}]]_{k < k_{max}} + b_l \right)$$

**Projection:**
$$h = \text{GELU}(v_4 \cdot Q_1 + b_{Q1})$$
$$u = h \cdot Q_2 + b_{Q2}$$

**Output:** $u \in \mathbb{R}^{N_x \times N_y \times d_u}$

## 5.2 Tensor Shape Flow

Let's trace shapes through the network for your problem:

**Shape progression:**

- **Input $a$**: (64, 64, 42) — Your features at each pixel
- **After Lifting**: (64, 64, 64) — Hidden representation
- **After Fourier L1**: (64, 64, 64) — Same shape
- **After Fourier L2**: (64, 64, 64) — Same shape
- **After Fourier L3**: (64, 64, 64) — Same shape
- **After Fourier L4**: (64, 64, 64) — Same shape
- **After Q₁ + GELU**: (64, 64, 128) — Intermediate
- **After Q₂ (Output)**: (64, 64, 1) — Temperature prediction

**Key insight:** Spatial dimensions (64, 64) never change. Only channel dimension changes at lifting and projection.

## 5.3 What Happens at Each Stage

**Lifting:** "What combinations of my 42 features matter?"
- Creates 64 learned combinations of input features
- No spatial mixing yet

**Fourier Layer 1:** "What are the basic spatial patterns?"
- Global spatial dependencies captured via FFT
- First nonlinear transformation

**Fourier Layer 2:** "How do these patterns interact?"
- Operates on Layer 1's output
- More complex spatial relationships

**Fourier Layer 3:** "What higher-order structures exist?"
- Even more abstract representations
- Approaching output-ready form

**Fourier Layer 4:** "Final spatial refinement"
- Last spectral processing
- Prepares for projection

**Projection:** "Convert to temperature"
- Decode hidden representation to physical quantity
- Final nonlinearity + linear output

## 5.4 Computational Flow

```
For each training sample:

1. Load input a(x,y) of shape (Nx, Ny, 42)

2. Lifting: 
   - Matrix multiply: (Nx×Ny, 42) @ (42, 64) → (Nx×Ny, 64)
   - Reshape to (Nx, Ny, 64)
   - Cost: O(Nx × Ny × 42 × 64)

3. Fourier Layer 1:
   - FFT: O(Nx × Ny × log(Nx × Ny) × 64)
   - Spectral multiply: O(k_max² × 64²)
   - IFFT: O(Nx × Ny × log(Nx × Ny) × 64)
   - Local path: O(Nx × Ny × 64²)
   - GELU: O(Nx × Ny × 64)

4. Fourier Layers 2-4: Same as Layer 1

5. Projection:
   - Q₁: O(Nx × Ny × 64 × 128)
   - GELU: O(Nx × Ny × 128)
   - Q₂: O(Nx × Ny × 128 × 1)

Total: Dominated by FFT operations and matrix multiplies
```

---

# Section 6: Loss Functions for Operator Learning

## 6.1 The Training Objective

We have training pairs: $\{(a^{(i)}, u^{(i)})\}_{i=1}^{N}$

Where:
- $a^{(i)}$ is the input function (your features)
- $u^{(i)}$ is the ground truth output (observed temperature)

Goal: Find network parameters $\theta$ that minimize:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \text{Loss}(\mathcal{G}_\theta(a^{(i)}), u^{(i)})$$

## 6.2 MSE Loss (Most Common)

**Mean Squared Error:**
$$\mathcal{L}_{MSE} = \frac{1}{N \cdot N_x \cdot N_y} \sum_{i=1}^{N} \sum_{x,y} \left( \hat{u}^{(i)}(x,y) - u^{(i)}(x,y) \right)^2$$

**When to use:**
- Default choice for regression
- When all errors are equally important
- When output distribution is roughly Gaussian

## 6.3 Relative L2 Loss (FNO Paper Default)

**Relative L2 Error:**
$$\mathcal{L}_{rel} = \frac{1}{N} \sum_{i=1}^{N} \frac{\| \hat{u}^{(i)} - u^{(i)} \|_2}{\| u^{(i)} \|_2}$$

Where $\| \cdot \|_2$ is the L2 norm over all spatial points.

**Why relative?**
- Scale-invariant: 10% error is 10% whether values are 0.1 or 1000
- Better for PDEs where solution magnitude varies
- Reported metric in FNO paper

**For temperature:**
- Values range ~280-320 K (or 0-50°C)
- Relative error gives consistent measure across this range

## 6.4 MAE Loss

**Mean Absolute Error:**
$$\mathcal{L}_{MAE} = \frac{1}{N \cdot N_x \cdot N_y} \sum_{i=1}^{N} \sum_{x,y} \left| \hat{u}^{(i)}(x,y) - u^{(i)}(x,y) \right|$$

**When to use:**
- When outliers are present (MAE is robust)
- When you care about median error rather than mean

## 6.5 Physics-Informed Loss (Optional)

For problems with known physics, add a PDE residual term:
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$$

**Example for heat equation:**
$$\mathcal{L}_{physics} = \| \nabla^2 \hat{u} - f \|^2$$

For your problem, this could encode:
- Spatial smoothness constraints
- Energy balance relationships
- Known temperature gradients near water bodies

## 6.6 Loss Choice for Your Problem

**Recommended:** Relative L2 or MSE

**Reasoning:**
- Temperature is continuous with no extreme outliers
- Relative L2 matches FNO paper methodology
- Easy to interpret (% error or °C error)

---

# Section 7: Training Considerations

## 7.1 Optimizer Choice

**Adam** is the standard choice:
- Adaptive learning rates per parameter
- Works well for neural networks
- Robust to hyperparameter choices

**Typical settings:**
- Learning rate: 1e-3 to 1e-4
- Betas: (0.9, 0.999) — defaults
- Weight decay: 1e-4 (L2 regularization)

## 7.2 Learning Rate Schedule

**Cosine annealing** or **step decay** commonly used:

**Cosine annealing:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t \cdot \pi}{T}))$$

Smoothly decreases learning rate from $\eta_{max}$ to $\eta_{min}$ over $T$ epochs.

**Step decay:**
- Reduce LR by factor 0.5 every 100 epochs
- Simple and effective

## 7.3 Batch Size

**Typical range:** 8-32 for FNO

**Trade-offs:**
- Larger batch: More stable gradients, but more memory
- Smaller batch: Regularization effect, less memory

**For 64×64 grid with d_v=64:**
- Batch size 16-20 fits in 16GB GPU
- Batch size 8-10 for 32×32×42 input

## 7.4 Number of Epochs

**Typical range:** 200-500 epochs

**Signs of convergence:**
- Validation loss plateaus
- Training/validation gap stabilizes
- Predictions visually accurate

**Early stopping:**
- Monitor validation loss
- Stop if no improvement for 50-100 epochs
- Restore best weights

## 7.5 Data Normalization

**Critical for stable training!**

**Input normalization:**
- Z-score: $(x - \mu) / \sigma$ for each channel
- Compute mean/std from training data only

**Output normalization:**
- Also normalize temperature
- Or normalize by typical range (e.g., divide by 50 for °C)

**At inference:**
- Apply same input normalization
- Denormalize output to get real temperature

## 7.6 Data Augmentation (Optional)

For spatial data, possible augmentations:
- **Random flips:** Horizontal/vertical
- **Random rotation:** 90°, 180°, 270°
- **Random crops:** Train on subregions

**Caution:** Ensure augmentations are physically meaningful
- Flipping temperature field: Usually OK
- Rotating: OK if domain is isotropic

## 7.7 Handling Multiple Time Steps

If predicting temporal evolution:
- Each time step = one training sample
- Or: Autoregressive (predict t+1, then t+2, ...)
- Or: Direct (separate model for each lead time)

For your single-time temperature prediction, this isn't needed.

---

# Section 8: Parameter Count and Model Sizing

## 8.1 Parameter Breakdown

**Lifting Layer P:**
- $P$: $d_a \times d_v$
- $b_P$: $d_v$
- Total: $d_a \times d_v + d_v$

**Each Fourier Layer:**
- $R$: $2 \times k_{max}^2 \times d_v^2$ (complex = 2× real)
- $W$: $d_v^2$
- $b$: $d_v$
- Total per layer: $2k_{max}^2 d_v^2 + d_v^2 + d_v$

**Projection Layer Q:**
- $Q_1$: $d_v \times d_{mid}$
- $b_{Q1}$: $d_{mid}$
- $Q_2$: $d_{mid} \times d_u$
- $b_{Q2}$: $d_u$
- Total: $d_v \times d_{mid} + d_{mid} + d_{mid} \times d_u + d_u$

## 8.2 Total for Standard Configuration

**Configuration:** $d_a=42$, $d_v=64$, $d_{mid}=128$, $d_u=1$, $k_{max}=12$, 4 layers

**Parameter breakdown:**

- **Lifting**: 42×64 + 64 = 2,752
- **Fourier Layer ×4**: 4 × (2×144×4096 + 4096 + 64) = 4,737,280
- **Projection**: 64×128 + 128 + 128×1 + 1 = 8,449
- **Total**: ~4.75M parameters

**Note:** The spectral weights R dominate! They scale as $O(k_{max}^2 \times d_v^2)$.

## 8.3 Smaller Model (Less Data)

If you have limited training data, reduce model size:

**Small configuration:** $d_v=32$, $k_{max}=8$

**Parameter breakdown (small):**

- **Lifting**: 42×32 + 32 = 1,376
- **Fourier Layer ×4**: 4 × (2×64×1024 + 1024 + 32) = 528,640
- **Projection**: 32×64 + 64 + 64×1 + 1 = 2,177
- **Total**: ~530K parameters

## 8.4 Model Size vs. Data Size

**Rule of thumb:**
- Parameters should be < 10× number of training samples × output size
- For 1000 training samples, 64×64 output: ~4M effective data points
- Can support 1-5M parameters

**Your data:**
- ~230 ECOSTRESS scenes (after quality filtering)
- Each scene: Many pixels, but spatially correlated
- Conservative: Start with smaller model

---

# Section 9: Hyperparameter Selection Guide

## 9.1 Key Hyperparameters

**Hidden dim ($d_v$)**: Typical 32, 64, 128 — For your problem: Start with 32-64

**Modes ($k_{max}$)**: Typical 8, 12, 16, 20 — For your problem: 12 for 64×64 grid

**Num layers**: Typical 2, 4, 6 — For your problem: 4 (standard)

**Learning rate**: Typical 1e-4 to 1e-3 — For your problem: 1e-3 initially

**Batch size**: Typical 4, 8, 16, 32 — For your problem: 8-16

**Weight decay**: Typical 0, 1e-4, 1e-3 — For your problem: 1e-4

## 9.2 How to Choose $d_v$

**Hidden dimension** controls model capacity.

**Guidelines:**
- $d_v \approx d_a$ to $2 \times d_a$ is common
- Larger $d_v$ = more capacity but more overfitting risk
- Start with $d_v = 32$ or $64$, increase if underfitting

**For your 42 features:**
- $d_v = 32$: Conservative, good for ~100 samples
- $d_v = 64$: Standard, good for ~500+ samples

## 9.3 How to Choose $k_{max}$

**Number of Fourier modes** controls spatial resolution of spectral processing.

**Guidelines:**
- $k_{max} \approx N/4$ to $N/8$ where N is grid size
- Larger $k_{max}$ = finer spatial features but more parameters
- For smooth fields (temperature), smaller $k_{max}$ often sufficient

**For 64×64 grid:**
- $k_{max} = 8$: Captures wavelengths down to 8 pixels
- $k_{max} = 12$: Finer detail, standard choice
- $k_{max} = 16$: Very fine, may overfit

## 9.4 How to Choose Number of Layers

**Depth** controls compositional complexity.

**Guidelines:**
- 4 layers is the standard
- Fewer layers (2-3) for simpler problems or less data
- More layers (6-8) rarely helps for typical PDEs

**For your problem:**
- Start with 4 layers
- If overfitting, try 3 layers
- If underfitting, try increasing $d_v$ first

## 9.5 Hyperparameter Search Strategy

**Step 1:** Start with defaults
- $d_v=64$, $k_{max}=12$, 4 layers, LR=1e-3

**Step 2:** Check for overfitting
- If train loss << val loss: Reduce $d_v$ or $k_{max}$, add dropout

**Step 3:** Check for underfitting
- If train loss high: Increase $d_v$, more epochs, check data normalization

**Step 4:** Fine-tune
- Grid search over LR: {1e-4, 3e-4, 1e-3}
- Try $d_v$ ∈ {32, 64}, $k_{max}$ ∈ {8, 12}

---

# Section 10: Connection to the Original Paper

## 10.1 Paper Reference

**"Fourier Neural Operator for Parametric Partial Differential Equations"**
Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar
arXiv:2010.08895 (NeurIPS 2020)

## 10.2 Key Claims from Paper

1. **Resolution invariance:** "FNO... can be used to transfer solutions between different grid resolutions"

2. **Efficiency:** "Our method achieves state-of-the-art accuracy while being 1000× faster than traditional PDE solvers"

3. **Benchmark results:**
   - Burgers equation: 0.4% relative error
   - Darcy flow: 1.08% relative error
   - Navier-Stokes: 0.7% relative error

## 10.3 Architecture Variations in Paper

The paper tests several variants:

- **FNO-2D**: Spatial dimensions only — Best for steady-state
- **FNO-3D**: Space + time as 3D — Best for time-dependent
- **FNO-2D+time**: 2D spatial, time as parameter — Alternative for time

**For your problem:** FNO-2D is appropriate (single-time temperature field)

## 10.4 What the Paper Doesn't Tell You

1. **Training stability:** Can be finicky; normalization is crucial
2. **Hyperparameter sensitivity:** $k_{max}$ and $d_v$ need tuning
3. **Memory usage:** FFT operations are memory-hungry
4. **Small data regime:** Paper uses 1000+ samples; fewer requires careful regularization

---

# Section 11: FNO for Your Urban Temperature Problem

## 11.1 Problem Mapping

- **Input $a(x)$**: 42 features (NDVI, building height, ERA5, etc.)
- **Output $u(x)$**: Land surface temperature (1 channel)
- **Domain**: NYC area at 70m resolution
- **Training data**: 230 ECOSTRESS scenes

## 11.2 Why FNO is Appropriate

1. **Global dependencies:** Central Park affects temperature kilometers away — spectral convolution captures this

2. **Smooth target:** Temperature varies smoothly — matches FNO's low-frequency bias

3. **Multi-scale physics:** Urban heat island operates at multiple scales — different $k$ modes capture different scales

4. **Resolution flexibility:** Train on coarser grid, evaluate on full 70m resolution

## 11.3 Expected Improvements Over Random Forest

**Random Forest:**
- Spatial context: None (pixel-independent)
- Feature engineering: Manual distance features
- Generalization: To similar pixels
- Resolution: Fixed

**FNO:**
- Spatial context: Global (full domain)
- Feature engineering: Learned spatial features
- Generalization: To similar fields
- Resolution: Flexible

## 11.4 Potential Challenges

1. **Limited data:** 230 scenes may require smaller model
2. **Temporal variation:** Different scenes have different conditions
3. **Domain shift:** Different seasons, weather patterns
4. **Missing data:** Cloud coverage, sensor issues

## 11.5 Recommended Configuration

**Conservative starting point:**
```
d_a = 42          # Your input features
d_v = 32          # Hidden dimension (small for limited data)
d_mid = 64        # Projection intermediate
d_u = 1           # Temperature output
k_max = 12        # Fourier modes
n_layers = 4      # Fourier layers
```

**Expected parameters:** ~600K (manageable for 230 samples)

---

# Section 12: Evaluation Metrics

## 12.1 Standard Metrics

**Root Mean Square Error (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_i (u^{(i)} - \hat{u}^{(i)})^2}$$

Units are same as temperature (°C or K)

**Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{N} \sum_i |u^{(i)} - \hat{u}^{(i)}|$$

More robust to outliers

**R² (Coefficient of Determination):**
$$R^2 = 1 - \frac{\sum_i (u^{(i)} - \hat{u}^{(i)})^2}{\sum_i (u^{(i)} - \bar{u})^2}$$

Fraction of variance explained (1.0 = perfect)

## 12.2 Spatial Metrics (Important for Your Work!)

Your thesis uses **spatial anomaly R²** — this is crucial.

**Spatial anomaly:**
$$\tilde{u}(x, y, t) = u(x, y, t) - \bar{u}(t)$$

Where $\bar{u}(t)$ is the mean temperature at time $t$.

**Why this matters:**
- Raw R² can be inflated by temporal patterns
- Spatial anomaly R² tests if you capture *where* is hot/cold
- This is the meaningful test for urban heat island prediction

## 12.3 Resolution-Specific Evaluation

Test resolution invariance:
1. Train on 64×64 downsampled grid
2. Evaluate on original 70m resolution
3. Compare predictions at different scales

---

# Summary: Complete FNO Architecture

## Key Components Checklist

**Architecture:**
- [ ] Lifting layer P: Input channels → hidden dimension
- [ ] 4 Fourier layers: Spectral conv + local path + GELU
- [ ] Projection layer Q: Hidden → output via 2-layer MLP

**Forward Pass:**
- [ ] $v_0 = Pa + b_P$ (lifting)
- [ ] $v_l = \text{GELU}(W_l v_{l-1} + \mathcal{K}_l v_{l-1} + b_l)$ for $l=1..4$
- [ ] $u = Q_2 \cdot \text{GELU}(Q_1 v_4 + b_{Q1}) + b_{Q2}$ (projection)

**Training:**
- [ ] Loss: Relative L2 or MSE
- [ ] Optimizer: Adam with LR ~1e-3
- [ ] Schedule: Cosine annealing or step decay
- [ ] Normalization: Z-score input and output

**Hyperparameters for Your Problem:**
- [ ] $d_v = 32-64$, $k_{max} = 12$, 4 layers
- [ ] ~500K-1M parameters for 230 training samples

---

# Next Step: Code Implementation

You now have complete theoretical understanding of FNO architecture.

**Chunk 3 Code** will implement:
1. Complete FNO class with all components
2. Training loop with proper loss and optimization
3. Evaluation metrics
4. Example on synthetic PDE data
5. Adaptation for your urban temperature problem

Let me know when you're ready for the code!
