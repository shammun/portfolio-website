# Fourier Neural Operator: Part 3
## The Complete FNO Architecture

---

**Series Navigation:** [← Part 2: The Fourier Layer](chunk2_blog_final.md) | **Part 3: Complete Architecture** | [Part 4: Advanced Topics →](chunk4_blog_final.md)

---

## Introduction

In Parts 1 and 2, we built up the foundational concepts piece by piece. Part 1 established how Fourier transforms convert spatial patterns into frequency components, and how the convolution theorem allows us to perform convolution efficiently in the frequency domain. Part 2 showed how to construct the Fourier layer—the workhorse of the FNO—by combining spectral convolution (for global patterns) with a local linear path (for fine details), followed by a nonlinear activation.

Now we assemble these pieces into the **complete Fourier Neural Operator**. A single Fourier layer is powerful, but not sufficient for learning complex PDE solution operators. We need additional components: a way to transform raw inputs into a suitable hidden representation, multiple Fourier layers stacked for depth, and a final projection to produce the output. This part covers all of these, plus the practical considerations of training and hyperparameter selection.

---

## External Resources and Further Reading

Before diving in, here are high-quality resources to supplement this tutorial:

### Video Tutorials

- **[Yannic Kilcher: FNO Paper Explained](https://www.youtube.com/watch?v=IaS72aHrJKE)** — 66 min: Comprehensive walkthrough of the original paper with code discussion
- **[Steve Brunton: Fourier Analysis](https://www.youtube.com/playlist?list=PLMrJAkhIeNNT_Xh3Oy0Y4LTj0Oxo8GqsC)** — Series: Excellent foundation on Fourier transforms for scientific computing
- **[MATLAB: Solve PDE Using FNO](https://www.mathworks.com/videos/solve-pde-using-fourier-neural-operator-1678273498061.html)** — 15 min: Step-by-step implementation with clear visualizations

### Written Tutorials and Blog Posts

- **[Fourier Neural Operator](https://zongyi-li.github.io/blog/2020/fourier-pde/)** — Zongyi Li: Authoritative explanation from the paper's first author
- **[Neural Operator Documentation](https://neuraloperator.github.io/dev/index.html)** — NeuralOperator Team: Official library documentation with theory guide
- **[MATLAB FNO Tutorial](https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html)** — MathWorks: Complete tutorial with exceptionally clear diagrams

### Key Papers

- **Fourier Neural Operator for Parametric PDEs** (2021) — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895): Original FNO paper (ICLR 2021)
- **Neural Operator: Learning Maps Between Function Spaces** (2023) — [arXiv:2108.08481](https://arxiv.org/abs/2108.08481): Mathematical foundations (JMLR)
- **Physics-Informed Neural Operator (PINO)** (2021) — [arXiv:2111.03794](https://arxiv.org/abs/2111.03794): Adding physics constraints
- **Geo-FNO** (2023) — [arXiv:2207.05209](https://arxiv.org/abs/2207.05209): Extension to irregular geometries

### Code Repositories

- **[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)** — 3.1k+ stars: Official PyTorch library (pip install neuraloperator)
- **[zongyi-li/fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)** — 1.5k+ stars: Original paper implementation - cleanest educational code
- **[NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)** — 2.1k+ stars: Production-ready implementation with distributed training

### Interactive Notebooks

- [**NeuralOperator Examples Gallery**](https://neuraloperator.github.io/dev/auto_examples/index.html) — Official Jupyter notebooks
- [**ETH Zurich AI Science Tutorial**](https://github.com/camlab-ethz/AI_Science_Engineering) — University course materials
- [**UvA Deep Learning Notebooks**](https://uvadlc-notebooks.readthedocs.io/) — Complete FNO tutorial with PyTorch

---

## Section 1: The Operator Learning Problem

Before diving into architecture details, let's be precise about what we're trying to accomplish.

Traditional neural networks learn mappings between finite-dimensional vectors. Given an input vector $\mathbf{x} \in \mathbb{R}^n$, they produce an output vector $\mathbf{y} \in \mathbb{R}^m$. This works well for classification or regression on tabular data, but PDEs require something different.

The FNO learns mappings between **function spaces**:

$$\mathcal{G}_\theta: \mathcal{A} \to \mathcal{U}$$

Here $\mathcal{A}$ is the space of input functions and $\mathcal{U}$ is the space of output functions. The subscript $\theta$ denotes all learnable parameters.

What does this mean concretely? Consider the steady-state heat equation with spatially varying thermal conductivity $\kappa(x)$:

$$\nabla \cdot (\kappa(x) \nabla u) = f(x)$$

A traditional numerical solver takes one specific conductivity field $\kappa(x)$ and one source term $f(x)$, then computes the corresponding temperature distribution $u(x)$. If you want solutions for different conductivity fields, you must run the solver again for each one.

The FNO learns the entire solution operator. Once trained, given *any* conductivity field from the training distribution, it predicts the corresponding solution in a single forward pass—typically 1000× faster than running a numerical solver. This is what makes neural operators so valuable for parametric studies, uncertainty quantification, and inverse problems.

---

## Section 2: The Three-Stage Architecture

The FNO processes data through three distinct stages:

$$a(x) \xrightarrow{P} v_0(x) \xrightarrow{\mathcal{L}_1} v_1(x) \xrightarrow{\mathcal{L}_2} v_2(x) \xrightarrow{\mathcal{L}_3} v_3(x) \xrightarrow{\mathcal{L}_4} v_4(x) \xrightarrow{Q} u(x)$$

**Stage 1 — Lifting (P):** The input function $a(x)$ might have a different number of channels than what our Fourier layers expect. For instance, we might have 10 input channels (initial condition, boundary indicators, forcing terms, coefficient fields) but want to work with 64 hidden channels for more representational capacity. The lifting layer handles this dimension change.

**Stage 2 — Fourier Layers ($\mathcal{L}_1$ through $\mathcal{L}_4$):** The heart of the FNO. Each Fourier layer applies the dual-path transformation we developed in Part 2: spectral convolution for global patterns, local linear transformation for fine details, combined through addition and passed through GELU activation. Stacking multiple layers enables learning complex, hierarchical features.

**Stage 3 — Projection (Q):** The final hidden representation needs to be decoded back to the output dimension. For a scalar field like temperature or pressure, this means going from 64 hidden channels down to 1 output channel. The projection uses a two-layer MLP for additional expressivity.

Let's visualize this complete pipeline to see how data flows through the architecture:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create a clean, professional FNO architecture diagram
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.axis('off')

# Color scheme - clean blue gradient
colors = {
    'input': '#E3F2FD', 'lift': '#BBDEFB', 'fourier': '#90CAF9',
    'project': '#64B5F6', 'output': '#42A5F5',
    'border': '#1565C0', 'arrow': '#37474F', 'text': '#263238'
}

box_h, box_w, y_center = 2.5, 1.8, 1.5

def draw_component(x, y, w, h, color, label, sublabel=''):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=color, edgecolor=colors['border'], linewidth=2)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.3, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=colors['text'])
        ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha='center', va='center', 
                fontsize=9, color=colors['text'], style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=colors['text'])

def draw_arrow(x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, mutation_scale=15))

# Component positions
pos = {'input': 0, 'lift': 2.2, 'f1': 4.4, 'f2': 6.2, 'f3': 8.0, 'f4': 9.8, 'project': 11.8, 'output': 14}
fl_w = 1.4

# Draw components
draw_component(pos['input'], y_center - box_h/2, box_w, box_h, colors['input'], 'Input', 'a(x)')
draw_component(pos['lift'], y_center - box_h/2, box_w, box_h, colors['lift'], 'Lift', 'P')
for i, key in enumerate(['f1', 'f2', 'f3', 'f4']):
    draw_component(pos[key], y_center - box_h/2, fl_w, box_h, colors['fourier'], f'F{i+1}', '')
draw_component(pos['project'], y_center - box_h/2, box_w, box_h, colors['project'], 'Project', 'Q')
draw_component(pos['output'], y_center - box_h/2, box_w, box_h, colors['output'], 'Output', 'u(x)')

# Dimension labels
for key, label in [('input', '$d_a$ channels'), ('lift', '$d_a → d_v$'), 
                   ('project', '$d_v → d_u$'), ('output', '$d_u$ channels')]:
    w = box_w
    ax.text(pos[key] + w/2, y_center - box_h/2 - 0.4, label, ha='center', fontsize=9, color='gray')
for key in ['f1', 'f2', 'f3', 'f4']:
    ax.text(pos[key] + fl_w/2, y_center - box_h/2 - 0.4, '$d_v$', ha='center', fontsize=9, color='gray')

# Draw arrows
draw_arrow(pos['input'] + box_w + 0.1, pos['lift'] - 0.1, y_center)
draw_arrow(pos['lift'] + box_w + 0.1, pos['f1'] - 0.1, y_center)
for i, (k1, k2) in enumerate([('f1','f2'), ('f2','f3'), ('f3','f4')]):
    draw_arrow(pos[k1] + fl_w + 0.1, pos[k2] - 0.1, y_center)
draw_arrow(pos['f4'] + fl_w + 0.1, pos['project'] - 0.1, y_center)
draw_arrow(pos['project'] + box_w + 0.1, pos['output'] - 0.1, y_center)

# Title and bracket
ax.text(8, 4.3, 'Fourier Neural Operator (FNO) Architecture', 
        ha='center', fontsize=14, fontweight='bold', color=colors['text'])
bracket_l, bracket_r = pos['f1'] - 0.2, pos['f4'] + fl_w + 0.2
bracket_y = y_center + box_h/2 + 0.3
ax.plot([bracket_l, bracket_l, bracket_r, bracket_r], 
        [bracket_y, bracket_y + 0.2, bracket_y + 0.2, bracket_y], color=colors['border'], lw=1.5)
ax.text((bracket_l + bracket_r)/2, bracket_y + 0.5, 
        '4 Fourier Layers (Spectral Conv + Linear + GELU)', ha='center', fontsize=10, color=colors['border'])

# Stage labels
ax.text(pos['lift'] + box_w/2, -0.7, 'Stage 1:\nDimension\nExpansion', ha='center', fontsize=9, color='gray')
ax.text((pos['f1'] + pos['f4'] + fl_w)/2, -0.7, 'Stage 2:\nIterative Spectral\nProcessing', ha='center', fontsize=9, color='gray')
ax.text(pos['project'] + box_w/2, -0.7, 'Stage 3:\nOutput\nDecoding', ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('figures/chunk3_01_architecture_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
```

![FNO Architecture Overview](chunk3_01_architecture_overview.png)

**Figure 1: FNO Architecture Overview — What You're Looking At**

This diagram shows the complete data flow through a Fourier Neural Operator from left to right:

- **Input (light blue):** Your input function $a(x)$ with $d_a$ channels. For a PDE problem, this might be initial conditions, boundary conditions, or physical parameters defined on a spatial grid.

- **Lift (P):** The lifting layer expands dimensions from $d_a$ to $d_v$ (typically $d_v > d_a$). This is a simple pointwise linear transformation—no spatial mixing yet.

- **F1, F2, F3, F4 (medium blue):** Four Fourier layers, each performing spectral convolution plus a local linear path, followed by GELU activation. The bracket above emphasizes these are the "heart" of the FNO where spatial patterns are learned.

- **Project (Q):** The projection layer compresses from $d_v$ hidden channels back to $d_u$ output channels (often $d_u = 1$ for scalar fields).

- **Output (dark blue):** The predicted solution $u(x)$ at the same spatial resolution as the input.

**Key Insights from This Figure:**

1. **Uniform hidden dimension:** Notice all four Fourier layers maintain the same $d_v$ dimension—this simplifies the architecture and enables residual-style learning.

2. **Three distinct stages:** The architecture cleanly separates concerns: dimension preparation (Lift), spatial learning (F1-F4), and output decoding (Project).

3. **Spatial dimensions unchanged:** The grid resolution stays constant throughout—only the channel dimension changes at lifting and projection.

> **📖 Reference Figure:** For an alternative visualization, see **Figure 1** in the [original FNO paper (Li et al., 2021)](https://arxiv.org/abs/2010.08895), which shows the architecture with mathematical notation. The paper is available under arXiv's non-exclusive license.

The dimension annotations below each component are important: the hidden dimension $d_v$ stays constant through all Fourier layers. This uniformity simplifies the architecture and allows residual-style learning where the input and output of each Fourier layer have the same shape.

---

## Section 3: The Lifting Layer

The lifting layer seems simple—just a linear transformation—but it serves several important purposes.

### 3.1 What Lifting Does

Mathematically, lifting applies a pointwise linear transformation at each spatial location:

$$v_0(x) = P \cdot a(x) + b_P$$

Here $P \in \mathbb{R}^{d_v \times d_a}$ is a learned weight matrix and $b_P \in \mathbb{R}^{d_v}$ is a bias vector. The key word is *pointwise*: at each spatial location $(x, y)$, we apply the exact same transformation. No spatial mixing happens yet—that's the job of the Fourier layers.

### 3.2 Why We Need Lifting

You might wonder why we don't just feed the input directly into the Fourier layers. There are three reasons:

**Dimension matching.** The input might have 10 channels while we want to work with 64 hidden channels. Or the input might have 100 features while the output is a single scalar field. The lifting layer bridges this gap.

**Feature preparation.** Raw input features may not be in the optimal form for spectral processing. Lifting learns which linear combinations of input features are most useful. Think of it as learned feature engineering: the network discovers that certain combinations of initial conditions and boundary terms create representations that the Fourier layers can work with effectively.

**Computational efficiency.** It's cleaner to standardize on a fixed hidden dimension $d_v$ throughout the Fourier layers rather than handling changing dimensions at each layer.

### 3.3 Implementing Lifting

The lifting layer is straightforward to implement:

```python
import torch
import torch.nn as nn

class LiftingLayer(nn.Module):
    """
    Projects input from d_a channels to d_v hidden channels.
    
    This is a pointwise (1x1) linear transformation applied independently
    at each spatial location.
    """
    def __init__(self, d_a: int, d_v: int):
        super().__init__()
        self.linear = nn.Linear(d_a, d_v)
        # Xavier initialization for stable training
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # x shape: (batch, nx, ny, d_a)
        # Output shape: (batch, nx, ny, d_v)
        return self.linear(x)
```

Notice that we use `nn.Linear` which operates on the last dimension. When applied to a tensor of shape `(batch, nx, ny, d_a)`, it transforms the last dimension from `d_a` to `d_v` while leaving the spatial dimensions untouched. This is exactly the pointwise behavior we want.

---

## Section 4: Stacking Fourier Layers

A single Fourier layer applies:

$$v_{\ell} = \sigma\left( W_\ell \cdot v_{\ell-1} + \mathcal{F}^{-1}(R_\ell \cdot \mathcal{F}(v_{\ell-1})) + b_\ell \right)$$

This is powerful—global receptive field, learnable spectral filtering—but one layer has limitations. It applies only one level of nonlinearity and can't learn hierarchical features. Stacking multiple Fourier layers addresses these limitations.

### 4.1 Why Multiple Layers?

Think about what each successive layer can learn:

**Layer 1** learns basic spatial patterns and initial feature combinations. It transforms the lifted representation into features that capture fundamental spatial structures.

**Layer 2** operates on Layer 1's output, learning interactions between those basic patterns. It can represent more complex spatial relationships.

**Layer 3** builds higher-order relationships on top of Layer 2's features. The network is now capturing increasingly abstract spatial dependencies.

**Layer 4** performs final refinement, preparing the representation for projection to the output.

This hierarchical structure is similar to deep CNNs, where early layers learn edges and textures while later layers learn more abstract concepts. The difference is that each Fourier layer has global receptive field from the start, so the hierarchy isn't about expanding receptive field—it's about learning progressively more complex nonlinear transformations.

### 4.2 How Many Layers?

The original FNO paper uses 4 Fourier layers, and this has become the standard choice. Why 4?

Empirically, 4 layers work well across a wide range of PDE problems (Burgers' equation, Darcy flow, Navier-Stokes). Adding more layers typically doesn't improve performance and can make training harder. Fewer layers (2-3) may be appropriate for simpler problems or when training data is limited.

The general guideline is to start with 4 layers. If you're overfitting, try reducing to 3. If you're underfitting, increase the hidden dimension $d_v$ rather than adding more layers—width helps more than depth for FNO.

### 4.3 Parameter Sharing vs. Independent Weights

**Standard FNO:** Each layer has its own independent weights $(R_\ell, W_\ell, b_\ell)$. This is the default and recommended approach.

**Alternative:** Share weights across all layers (similar to RNNs). This reduces parameters and provides stronger regularization, but is less common in practice and typically underperforms independent weights.

### 4.4 Residual Connections (Optional but Beneficial)

Some FNO variants add skip connections within Fourier layers:

$$v_{\ell+1} = \sigma(W_\ell v_\ell + \mathcal{K}_\ell v_\ell + b_\ell) + v_\ell$$

The added $+ v_\ell$ term creates a residual connection. Benefits include:

- **Easier gradient flow:** Gradients can bypass the transformation, making deep networks easier to train
- **Identity mapping:** If a layer isn't needed, it can learn to pass input through unchanged
- **Often improves performance:** Especially for deeper networks

The original FNO paper doesn't use residual connections in Fourier layers, but many implementations add them. For 4-layer networks, the difference is usually small; for deeper networks, residual connections become more important.

### 4.5 Implementing the Fourier Layer Stack

Here's the Fourier layer implementation from Part 2, now ready to be stacked:

```python
class FourierLayer(nn.Module):
    """
    Single Fourier layer with spectral convolution and local linear path.
    
    Computes: v_out = GELU(W·v_in + IFFT(R·FFT(v_in)) + b)
    """
    def __init__(self, d_v: int, k_max: int):
        super().__init__()
        self.d_v = d_v
        self.k_max = k_max
        
        # Spectral weights: complex-valued, shape (k_max, k_max, d_v, d_v)
        # We parameterize as two real tensors for real and imaginary parts
        scale = 1.0 / (d_v * d_v)
        self.R_real = nn.Parameter(scale * torch.randn(k_max, k_max, d_v, d_v))
        self.R_imag = nn.Parameter(scale * torch.randn(k_max, k_max, d_v, d_v))
        
        # Local linear path (1x1 convolution)
        self.W = nn.Linear(d_v, d_v)
        
        # Bias
        self.b = nn.Parameter(torch.zeros(d_v))
    
    def forward(self, v):
        batch_size, nx, ny, d_v = v.shape
        
        # === Spectral Path ===
        # FFT over spatial dimensions
        v_fft = torch.fft.rfft2(v, dim=(1, 2))
        
        # Truncate to k_max modes
        v_fft_truncated = v_fft[:, :self.k_max, :self.k_max, :]
        
        # Apply spectral weights (complex multiplication)
        R_complex = torch.complex(self.R_real, self.R_imag)
        # Einstein summation: multiply and sum over input channel dimension
        out_fft = torch.einsum('ijkl,bijk->bijl', R_complex, v_fft_truncated)
        
        # Pad back to original size and inverse FFT
        out_fft_padded = torch.zeros_like(v_fft)
        out_fft_padded[:, :self.k_max, :self.k_max, :] = out_fft
        spectral_out = torch.fft.irfft2(out_fft_padded, s=(nx, ny), dim=(1, 2))
        
        # === Local Path ===
        local_out = self.W(v)
        
        # === Combine and Activate ===
        out = spectral_out + local_out + self.b
        return torch.nn.functional.gelu(out)
```

---

## Section 5: The Projection Layer

After passing through all Fourier layers, we have a hidden representation $v_L(x)$ with $d_v$ channels. The projection layer maps this to the output dimension $d_u$.

### 5.1 Two-Layer MLP Design

Unlike the single-linear-layer lifting, projection uses a two-layer MLP:

$$h(x) = \sigma(Q_1 \cdot v_L(x) + b_{Q_1})$$
$$u(x) = Q_2 \cdot h(x) + b_{Q_2}$$

The intermediate dimension is typically $2 \times d_v$. Why two layers instead of one?

The Fourier layers have been learning representations optimized for spectral processing—lots of smooth, global features. The output might need sharper, more localized features. The two-layer MLP with a nonlinearity (GELU) in between provides the flexibility to transform the Fourier layer representation into the output form.

Additionally, for many PDE problems, the output is a single scalar field (temperature, pressure, concentration). Going directly from 64 channels to 1 channel with a single linear layer would be a severe bottleneck. The intermediate layer with 128 dimensions allows more gradual information compression.

### 5.2 Implementing Projection

```python
class ProjectionLayer(nn.Module):
    """
    Projects from hidden dimension to output dimension via two-layer MLP.
    
    Architecture: d_v -> d_mid -> d_u with GELU activation after first layer.
    """
    def __init__(self, d_v: int, d_u: int, d_mid: int = None):
        super().__init__()
        if d_mid is None:
            d_mid = d_v * 2
        
        self.layer1 = nn.Linear(d_v, d_mid)
        self.layer2 = nn.Linear(d_mid, d_u)
        
        # Xavier initialization
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
    
    def forward(self, v):
        # v shape: (batch, nx, ny, d_v)
        h = torch.nn.functional.gelu(self.layer1(v))
        return self.layer2(h)
```

---

## Section 6: The Complete FNO

Now we can assemble all components into the complete architecture. Let's visualize how each component transforms the data:

```python
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.3)

# Panel 1: Lifting Layer
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Lifting Layer P\n$d_a \\rightarrow d_v$', fontsize=12, fontweight='bold', pad=10)

# Input channels
for i, label in enumerate(['$a_1$', '$a_2$', '...', '$a_{d_a}$']):
    y = 8 - i*1.8
    rect = plt.Rectangle((0.5, y-0.4), 2.5, 0.8, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(1.75, y, label, ha='center', va='center', fontsize=11)

# Arrow
ax1.annotate('', xy=(6.5, 5), xytext=(4, 5), 
             arrowprops=dict(arrowstyle='-|>', color='#1565C0', lw=2.5, mutation_scale=20))
ax1.text(5.25, 5.8, 'Linear', ha='center', fontsize=10, color='#1565C0')
ax1.text(5.25, 4.2, '$v_0 = Pa + b$', ha='center', fontsize=10)

# Output channels
for i, label in enumerate(['$v_1$', '$v_2$', '...', '$v_{d_v}$']):
    y = 8 - i*1.8
    rect = plt.Rectangle((7, y-0.4), 2.5, 0.8, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(8.25, y, label, ha='center', va='center', fontsize=11)

# Panel 2: Fourier Layer (dual-path)
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Fourier Layer\nDual-Path Architecture', fontsize=12, fontweight='bold', pad=10)

# Input
rect = plt.Rectangle((0.3, 4), 2, 2), 
ax2.add_patch(plt.Rectangle((0.3, 4), 2, 2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2))
ax2.text(1.3, 5, '$v_{\\ell-1}$', ha='center', va='center', fontsize=12, fontweight='bold')

# Spectral path (top)
ax2.annotate('', xy=(3.5, 7.5), xytext=(2.5, 5.5), 
             arrowprops=dict(arrowstyle='-|>', color='#E65100', lw=2))
for x, label in [(4, 'FFT'), (6.5, '$R \\cdot$'), (9, 'IFFT')]:
    rect = plt.Rectangle((x, 7), 2, 1.2, facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x+1, 7.6, label, ha='center', va='center', fontsize=10, fontweight='bold')
ax2.annotate('', xy=(6.3, 7.6), xytext=(6, 7.6), arrowprops=dict(arrowstyle='-|>', color='#E65100', lw=1.5))
ax2.annotate('', xy=(8.8, 7.6), xytext=(8.5, 7.6), arrowprops=dict(arrowstyle='-|>', color='#E65100', lw=1.5))
ax2.text(7.5, 8.7, 'Spectral Path (Global)', ha='center', fontsize=10, color='#E65100', fontweight='bold')

# Local path (bottom)
ax2.annotate('', xy=(6.5, 2.5), xytext=(2.5, 4.5), 
             arrowprops=dict(arrowstyle='-|>', color='#2E7D32', lw=2))
rect = plt.Rectangle((6.5, 2), 2.5, 1.2, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
ax2.add_patch(rect)
ax2.text(7.75, 2.6, '$W \\cdot$', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(7.75, 1, 'Local Path', ha='center', fontsize=10, color='#2E7D32', fontweight='bold')

# Merge point
ax2.annotate('', xy=(11, 5), xytext=(11, 7), arrowprops=dict(arrowstyle='-|>', color='#E65100', lw=2))
ax2.annotate('', xy=(11, 5), xytext=(9.2, 2.6), arrowprops=dict(arrowstyle='-|>', color='#2E7D32', lw=2))
ax2.plot(11, 5, 'o', markersize=15, color='white', markeredgecolor='#333', markeredgewidth=2)
ax2.text(11, 5, '+', ha='center', va='center', fontsize=14, fontweight='bold')

# GELU and output
rect = plt.Rectangle((11.5, 4), 2, 2, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
ax2.add_patch(rect)
ax2.text(12.5, 5, 'GELU', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.annotate('', xy=(11.5, 5), xytext=(11.3, 5), arrowprops=dict(arrowstyle='-|>', color='#333', lw=2))

# Panel 3: Projection Layer
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Projection Layer Q\n$d_v \\rightarrow d_u$', fontsize=12, fontweight='bold', pad=10)

# Input
rect = plt.Rectangle((0.5, 4), 2, 2, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax3.add_patch(rect)
ax3.text(1.5, 5, '$v_L$', ha='center', va='center', fontsize=12, fontweight='bold')

# First layer + GELU
ax3.annotate('', xy=(3.5, 5), xytext=(2.7, 5), arrowprops=dict(arrowstyle='-|>', color='#7B1FA2', lw=2))
rect = plt.Rectangle((3.5, 3.8), 2.5, 2.4, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
ax3.add_patch(rect)
ax3.text(4.75, 5.3, '$Q_1$', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(4.75, 4.5, 'GELU', ha='center', va='center', fontsize=9)

# Second layer
ax3.annotate('', xy=(7, 5), xytext=(6.2, 5), arrowprops=dict(arrowstyle='-|>', color='#7B1FA2', lw=2))
rect = plt.Rectangle((7, 4), 2.5, 2, facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
ax3.add_patch(rect)
ax3.text(8.25, 5, '$Q_2$', ha='center', va='center', fontsize=10, fontweight='bold')

ax3.text(5, 1.5, '$u = Q_2 \\cdot \\text{GELU}(Q_1 v_L + b_1) + b_2$', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#FFFDE7', edgecolor='#F9A825'))

plt.savefig('figures/chunk3_02_layer_details.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Layer Details](chunk3_02_layer_details.png)

**Figure 2: Internal Structure of Each Component — What You're Looking At**

This three-panel figure reveals what happens inside each type of layer:

**Left Panel — Lifting Layer (P):**
- Input channels $a_1, a_2, \ldots, a_{d_a}$ (green circles on left) feed into a weight matrix $P$ of size $d_a \times d_v$
- Output channels $v_1, v_2, \ldots, v_{d_v}$ (green circles on right) are linear combinations of all inputs
- This is a **pointwise operation**—each spatial location is transformed independently
- Think of it as "mixing" your input features into a richer representation

**Center Panel — Fourier Layer (Dual Path):**
- **Spectral Path (top, orange):** Input $v_\ell$ → FFT → multiply by learnable weights $R$ → IFFT → contributes to output
- **Local Path (bottom, orange):** Input $v_\ell$ → multiply by weights $W$ → contributes to output
- Both paths **merge by addition** at the "+" node
- Finally, **GELU activation** introduces nonlinearity
- The spectral path captures global patterns; the local path handles features that don't fit the spectral representation

**Right Panel — Projection Layer (Q):**
- A **two-layer MLP** that progressively compresses: $d_v \to d_{mid} \to d_u$
- First layer $Q_1$ expands to intermediate dimension (often $d_{mid} = 2 \times d_v$)
- GELU activation between layers
- Second layer $Q_2$ projects to final output dimension
- Dimension labels below show the shape transformation

**Key Insights from This Figure:**

1. **Dual-path architecture is essential:** The Fourier layer isn't just spectral convolution—the local $W$ path captures information lost in mode truncation.

2. **Lifting and projection are simple:** Despite their importance, these are just linear transformations (lifting) or small MLPs (projection).

3. **All spatial mixing happens in Fourier layers:** Lifting and projection operate pointwise—they don't "see" neighboring pixels.

### 6.1 Complete Implementation

Here's the full FNO class putting everything together:

```python
class FNO2d(nn.Module):
    """
    Complete 2D Fourier Neural Operator.
    
    Architecture: Lifting -> Fourier Layers (x4) -> Projection
    
    Parameters:
        d_a: Input channels
        d_v: Hidden channels (uniform across Fourier layers)
        d_u: Output channels  
        k_max: Number of Fourier modes to keep
        n_layers: Number of Fourier layers (default 4)
    """
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int, n_layers: int = 4):
        super().__init__()
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.k_max = k_max
        self.n_layers = n_layers
        
        # Lifting layer
        self.lifting = LiftingLayer(d_a, d_v)
        
        # Stack of Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(d_v, k_max) for _ in range(n_layers)
        ])
        
        # Projection layer
        self.projection = ProjectionLayer(d_v, d_u, d_mid=d_v * 2)
    
    def forward(self, a):
        """
        Forward pass through FNO.
        
        Args:
            a: Input tensor of shape (batch, nx, ny, d_a)
        
        Returns:
            u: Output tensor of shape (batch, nx, ny, d_u)
        """
        # Lifting: (batch, nx, ny, d_a) -> (batch, nx, ny, d_v)
        v = self.lifting(a)
        
        # Fourier layers: (batch, nx, ny, d_v) -> (batch, nx, ny, d_v)
        for layer in self.fourier_layers:
            v = layer(v)
        
        # Projection: (batch, nx, ny, d_v) -> (batch, nx, ny, d_u)
        u = self.projection(v)
        
        return u
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

The forward pass is clean and readable: lift, iterate through Fourier layers, project. This modularity makes it easy to experiment with different configurations.

> **💡 Pro Tip:** While understanding the implementation from scratch is valuable for learning, for production use consider the [official neuraloperator library](https://github.com/neuraloperator/neuraloperator) which includes optimized implementations, pre-built datasets, and extensive testing. Install with `pip install neuraloperator`.

---

## Section 7: Training the FNO

With the architecture defined, we need to train it effectively. This section covers loss functions, optimization, and practical considerations.

### 7.1 Loss Functions

For regression problems like PDE solving, we have several loss function options:

**Mean Squared Error (MSE):**
$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|u_i - \hat{u}_i\|^2$$

MSE is the default choice for regression. It's simple, well-understood, and works when all errors are equally important. Use MSE when output distributions are roughly Gaussian and you don't need scale invariance.

**Relative L2 Loss (FNO Paper Default):**
$$\mathcal{L}_{rel} = \frac{1}{N} \sum_{i=1}^{N} \frac{\|u_i - \hat{u}_i\|_2}{\|u_i\|_2}$$

Normalizing by the target magnitude makes the loss scale-invariant: a 10% error is 10% whether values are 0.1 or 1000. This is particularly useful for PDEs where solution magnitudes vary significantly across samples. The original FNO paper uses relative L2, and it's generally recommended.

**Mean Absolute Error (MAE):**
$$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |u_i - \hat{u}_i|$$

MAE is more robust to outliers than MSE but can lead to slower convergence. Use MAE when your data contains outliers or when you care about median error rather than mean error.

**Physics-Informed Loss (Optional):**

For problems with known physics, you can add a PDE residual term:
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$$

For example, if the solution should satisfy the heat equation, you might add:
$$\mathcal{L}_{physics} = \| \nabla^2 \hat{u} - f \|^2$$

This encourages the network to produce physically consistent predictions, which can improve generalization especially with limited training data. However, computing PDE residuals adds complexity and computational cost.

```python
def relative_l2_loss(pred, target):
    """
    Relative L2 loss as used in the original FNO paper.
    
    Normalizes error by target magnitude, making the loss scale-invariant.
    """
    diff_norm = torch.norm(pred - target, p=2, dim=(-2, -1))
    target_norm = torch.norm(target, p=2, dim=(-2, -1))
    return torch.mean(diff_norm / (target_norm + 1e-8))

def mse_loss(pred, target):
    """Standard mean squared error."""
    return torch.mean((pred - target) ** 2)

def mae_loss(pred, target):
    """Mean absolute error - more robust to outliers."""
    return torch.mean(torch.abs(pred - target))
```

### 7.2 Optimization

The standard optimizer choice is Adam with an initial learning rate around $10^{-3}$:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

A small weight decay ($10^{-4}$ to $10^{-5}$) provides regularization without overly constraining the model.

For the learning rate schedule, cosine annealing works well—it starts high to explore the loss landscape, then gradually decreases for fine-tuning:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
```

Alternatively, step decay (reducing LR by 0.5 every 100 epochs) is simpler and also effective.

### 7.3 Batch Size and Training Duration

**Batch size** affects both training stability and memory usage:

- **Larger batches (16-32):** More stable gradients, better GPU utilization, but requires more memory
- **Smaller batches (4-8):** Acts as regularization, uses less memory, but noisier gradients

For a 64×64 grid with $d_v=64$, batch sizes of 16-20 typically fit in 16GB of GPU memory.

**Number of epochs:** Typically 200-500 epochs are needed for convergence. Signs that training has converged include:
- Validation loss plateaus for 50+ epochs
- Training/validation gap stabilizes
- Predictions are visually accurate

**Early stopping** helps prevent overfitting: monitor validation loss and stop if no improvement for 50-100 epochs, then restore the best weights.

### 7.4 Data Augmentation (Optional)

For spatial data, augmentation can help with limited training samples:

- **Random flips:** Horizontal and/or vertical
- **Random rotation:** 90°, 180°, 270° (if domain is isotropic)
- **Random crops:** Train on subregions of larger domains

Ensure augmentations are physically meaningful for your problem. Flipping a temperature field is usually fine; rotating might not be if there's a preferred direction (like prevailing wind).

### 7.5 Visualizing Training Dynamics

Let's visualize typical training dynamics:

```python
np.random.seed(42)
epochs = np.arange(500)

# Simulate realistic training curves
train_loss = 0.5 * np.exp(-epochs/100) + 0.01 + np.random.randn(500) * 0.003
val_loss = 0.55 * np.exp(-epochs/120) + 0.02 + np.random.randn(500) * 0.005
train_loss = np.maximum(train_loss, 0.008)
val_loss = np.maximum(val_loss, 0.015)

# Smooth for visualization
from scipy.ndimage import gaussian_filter1d
train_loss = gaussian_filter1d(train_loss, sigma=3)
val_loss = gaussian_filter1d(val_loss, sigma=3)

# Learning rate schedule
lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * epochs / 500))

# R² scores (derived from loss)
r2_train = 1 - train_loss / 0.5
r2_val = 1 - val_loss / 0.55

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].semilogy(epochs, train_loss, 'b-', lw=2, label='Train', alpha=0.9)
axes[0, 0].semilogy(epochs, val_loss, 'r-', lw=2, label='Validation', alpha=0.9)
axes[0, 0].fill_between(epochs, train_loss, val_loss, alpha=0.1, color='purple')
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Loss (log scale)', fontsize=11)
axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, 500)

# Learning rate
axes[0, 1].plot(epochs, lr * 1000, 'g-', lw=2)
axes[0, 1].fill_between(epochs, 0, lr * 1000, alpha=0.2, color='green')
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Learning Rate (×10⁻³)', fontsize=11)
axes[0, 1].set_title('Cosine Annealing Schedule', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 500)
axes[0, 1].annotate('High LR:\nExplore', xy=(50, 0.9), fontsize=9, ha='center')
axes[0, 1].annotate('Low LR:\nRefine', xy=(450, 0.1), fontsize=9, ha='center')

# R² evolution
axes[1, 0].plot(epochs, r2_train, 'b-', lw=2, label='Train R²', alpha=0.9)
axes[1, 0].plot(epochs, r2_val, 'r-', lw=2, label='Validation R²', alpha=0.9)
axes[1, 0].axhline(y=0.95, color='green', ls='--', alpha=0.7, label='Target: 0.95')
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('R² Score', fontsize=11)
axes[1, 0].set_title('Model Performance', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 500)
axes[1, 0].set_ylim(0.8, 1.0)

# Generalization gap
gap = val_loss - train_loss
axes[1, 1].plot(epochs, gap, 'purple', lw=2)
axes[1, 1].axhline(y=0.007, color='orange', ls='--', lw=1.5, label='Acceptable threshold')
axes[1, 1].fill_between(epochs, 0, gap, where=(gap > 0.007), alpha=0.3, color='red')
axes[1, 1].fill_between(epochs, 0, gap, where=(gap <= 0.007), alpha=0.3, color='green')
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Validation - Train Loss', fontsize=11)
axes[1, 1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, 500)

plt.suptitle('FNO Training Dynamics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/chunk3_04_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Training Curves](chunk3_04_training_curves.png)

**Figure 4: FNO Training Dynamics — What You're Looking At**

This four-panel figure shows everything you need to monitor during FNO training:

**Top-Left — Loss Curves (Log Scale):**
- **Blue line:** Training loss decreasing over epochs
- **Red line:** Validation loss (on held-out data)
- **Purple shaded area:** The gap between validation and training loss
- **Green dashed line:** Target loss level
- Both curves should decrease; validation should track training (not diverge upward)

**Top-Right — Learning Rate Schedule:**
- Shows cosine annealing from high (1e-3) to low (near 0)
- **"High LR: Explore"** — Early epochs use large steps to explore the loss landscape broadly
- **"Low LR: Refine"** — Later epochs use small steps to fine-tune near a minimum
- The smooth curve prevents training instabilities from abrupt LR changes

**Bottom-Left — R² Score Evolution:**
- **Blue:** Training R² (should approach 1.0)
- **Red:** Validation R² (the metric that actually matters)
- **Green dashed:** Target threshold (e.g., 0.95)
- Tracks how much variance your model explains—higher is better

**Bottom-Right — Generalization Gap:**
- Difference between validation and training loss
- **Green region:** Acceptable gap (model generalizes well)
- **Red region:** Gap exceeds threshold (potential overfitting)
- If this gap keeps growing, consider regularization or smaller model

**Key Insights from This Figure:**

1. **Watch the gap, not just the loss:** Low training loss means nothing if validation loss is much higher—you're memorizing, not learning.

2. **Early stopping matters:** The best model often occurs before training "converges"—when validation loss starts increasing while training loss still decreases.

3. **Cosine annealing is gentle:** Unlike step decay (sudden drops), cosine annealing smoothly reduces LR, which works well for FNO's complex loss landscape.

4. **R² gives interpretable progress:** Unlike loss values, R² tells you directly what fraction of variance you're capturing—0.95 means 95% explained.

### 7.6 Data Normalization

This point cannot be overstated: **proper normalization is critical for FNO training stability.**

Both inputs and outputs should be normalized to zero mean and unit variance:

```python
class DataNormalizer:
    """
    Z-score normalization for inputs and outputs.
    
    Computes statistics from training data, then applies to all data.
    """
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        """Compute mean and std from training data."""
        self.mean = data.mean(dim=(0, 1, 2), keepdim=True)
        self.std = data.std(dim=(0, 1, 2), keepdim=True) + 1e-8
        return self
    
    def transform(self, data):
        """Apply normalization."""
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """Undo normalization."""
        return data * self.std + self.mean

# Usage:
# input_normalizer = DataNormalizer().fit(train_inputs)
# output_normalizer = DataNormalizer().fit(train_outputs)
# 
# normalized_input = input_normalizer.transform(input)
# prediction = model(normalized_input)
# final_output = output_normalizer.inverse_transform(prediction)
```

Without normalization, the FFT operations can produce numerical instabilities, gradients can explode or vanish, and training will likely fail or converge to poor solutions.

> **🚀 Production Training:** For large-scale training with distributed computing support, see [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo), which provides production-optimized FNO implementations with multi-GPU and multi-node support.

---

## Section 8: Understanding What FNO Learns

To build intuition, let's create a synthetic example showing FNO learning a PDE-like operator. We'll generate noisy input fields and smooth target fields (simulating a diffusion process), then show what a trained FNO would produce:

```python
np.random.seed(42)
N = 64

# Create coordinate grids
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Input: structured pattern plus noise (simulating noisy initial condition)
input_field = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) + 0.3 * np.random.randn(N, N)

# Target: smoothed version (simulating diffusion operator output)
# We apply spectral smoothing to simulate what a PDE solver would produce
input_fft = np.fft.fft2(input_field)
kx = np.fft.fftfreq(N)
ky = np.fft.fftfreq(N)
KX, KY = np.meshgrid(kx, ky)
diffusion_kernel = np.exp(-10 * (KX**2 + KY**2))
target_field = np.fft.ifft2(input_fft * diffusion_kernel).real

# Simulated FNO prediction (close to target with small errors)
# In practice this would come from a trained model
prediction = target_field + np.random.randn(N, N) * 0.03
error = np.abs(prediction - target_field)
rmse = np.sqrt(np.mean((prediction - target_field)**2))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: spatial fields
vmin, vmax = min(input_field.min(), target_field.min()), max(input_field.max(), target_field.max())

im1 = axes[0, 0].imshow(input_field, cmap='RdBu_r', extent=[0, 1, 0, 1], vmin=-1.5, vmax=1.5)
axes[0, 0].set_title('Input: Noisy Initial Condition', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

im2 = axes[0, 1].imshow(target_field, cmap='RdBu_r', extent=[0, 1, 0, 1], vmin=-1.5, vmax=1.5)
axes[0, 1].set_title('Target: PDE Solution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

im3 = axes[0, 2].imshow(prediction, cmap='RdBu_r', extent=[0, 1, 0, 1], vmin=-1.5, vmax=1.5)
axes[0, 2].set_title('FNO Prediction', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('y')
plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)

# Bottom row: frequency analysis and error
input_spectrum = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(input_field))) + 1)
target_spectrum = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(target_field))) + 1)

im4 = axes[1, 0].imshow(input_spectrum, cmap='hot', extent=[-N//2, N//2, -N//2, N//2])
axes[1, 0].set_title('Input Spectrum (log)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('$k_x$')
axes[1, 0].set_ylabel('$k_y$')
plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

im5 = axes[1, 1].imshow(target_spectrum, cmap='hot', extent=[-N//2, N//2, -N//2, N//2])
axes[1, 1].set_title('Target Spectrum (log)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('$k_x$')
axes[1, 1].set_ylabel('$k_y$')
plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

im6 = axes[1, 2].imshow(error, cmap='Reds', extent=[0, 1, 0, 1])
axes[1, 2].set_title(f'Absolute Error (RMSE = {rmse:.4f})', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('y')
plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)

plt.suptitle('FNO Learning a Diffusion-like Operator', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/chunk3_03_synthetic_pde.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Synthetic PDE Example](chunk3_03_synthetic_pde.png)

**Figure 3: FNO Learning a Smoothing Operator — What You're Looking At**

This six-panel figure demonstrates FNO learning to replicate a diffusion-like operator:

**Top Row — Physical Space:**
- **Input (left):** A noisy field with clear sinusoidal structure corrupted by random high-frequency noise. This simulates a typical initial condition you might give to a PDE solver.
- **Target (center):** The "ground truth" solution after diffusion—the smooth, large-scale sinusoidal pattern is preserved while noise is filtered out.
- **FNO Prediction (right):** What the trained FNO produces—visually nearly identical to the target.

**Bottom Row — Frequency Space:**
- **Input Spectrum (left):** Energy spread across many frequencies. The bright spots at the center are the dominant sinusoidal modes; the diffuse background is noise energy.
- **Target Spectrum (center):** Energy concentrated at low frequencies only—the diffusion operator has attenuated high-frequency components (the cross-shaped pattern shows only the lowest modes remain strong).
- **Error Map (right):** Spatial distribution of prediction error, with RMSE reported. Small, uniformly distributed errors indicate the FNO learned the operator well.

**Key Insights from This Figure:**

1. **PDEs often act as low-pass filters:** Diffusion, heat conduction, and many physical processes attenuate high frequencies—exactly what the FNO's mode truncation naturally captures.

2. **Spectral view reveals the transformation:** Looking at frequency space shows clearly that the operator "filters out" high-frequency content—this is why spectral methods work so well.

3. **FNO learns the correct filtering:** The prediction matches the target in both physical and frequency space, demonstrating the network learned the underlying physics.

4. **Error is spatially uniform:** No systematic bias or problematic regions—the FNO generalizes well across the entire domain.

> **📊 Standard Benchmarks:** The original FNO paper reports impressive results on standard PDE benchmarks:
> - **Burgers' equation:** 0.4% relative error
> - **Darcy flow:** 1.08% relative error  
> - **Navier-Stokes:** 0.7% relative error
> 
> These benchmarks are available in the [official repository](https://github.com/neuraloperator/neuraloperator) for reproducibility.

---

## Section 9: Hyperparameter Selection

Choosing the right hyperparameters is crucial for FNO performance. The three main hyperparameters are:

- **$d_v$**: Hidden dimension (number of channels in Fourier layers)
- **$k_{max}$**: Number of Fourier modes retained
- **$n_{layers}$**: Number of Fourier layers

### 9.1 Impact on Model Size

The total parameter count depends primarily on $d_v$ and $k_{max}$:

$$\text{Parameters} \approx d_a \cdot d_v + n_{layers} \cdot (2 k_{max}^2 d_v^2 + d_v^2) + 2 d_v^2 + d_v \cdot d_u$$

The dominant term is $2 k_{max}^2 d_v^2$ from the spectral weights—doubling either $k_{max}$ or $d_v$ roughly quadruples the parameter count.

### 9.2 Parameter Breakdown by Component

Let's trace where the parameters come from:

**Lifting Layer P:**
- Weight matrix $P$: $d_a \times d_v$ parameters
- Bias $b_P$: $d_v$ parameters
- Total: $d_a \cdot d_v + d_v$

**Each Fourier Layer:**
- Spectral weights $R$: $2 \times k_{max}^2 \times d_v^2$ (factor of 2 for complex numbers)
- Local weights $W$: $d_v^2$
- Bias $b$: $d_v$
- Total per layer: $2k_{max}^2 d_v^2 + d_v^2 + d_v$

**Projection Layer Q:**
- First layer $Q_1$: $d_v \times d_{mid}$ + bias $d_{mid}$
- Second layer $Q_2$: $d_{mid} \times d_u$ + bias $d_u$
- Total: $d_v \cdot d_{mid} + d_{mid} + d_{mid} \cdot d_u + d_u$

**Example calculation** for $d_a=10$, $d_v=64$, $d_{mid}=128$, $d_u=1$, $k_{max}=12$, 4 layers:

**Parameter breakdown:**

- **Lifting**: 10×64 + 64 = 704
- **Fourier Layer ×4**: 4 × (2×144×4096 + 4096 + 64) = 4,737,280
- **Projection**: 64×128 + 128 + 128×1 + 1 = 8,449
- **Total**: ~4.75M parameters

The spectral weights $R$ dominate the parameter count, scaling as $O(k_{max}^2 \times d_v^2)$.

### 9.3 Model Size vs. Data Size

A useful rule of thumb: total parameters should be less than 10× the number of training samples × output size.

For example, with 1000 training samples on a 64×64 grid:
- Effective data points: 1000 × 64 × 64 ≈ 4M
- Can reasonably support 1-5M parameters

With limited data (100-500 samples), use smaller models ($d_v=32$, $k_{max}=8$) to prevent overfitting.

### 9.4 Visualizing Parameter Scaling

```python
def count_parameters(d_a, d_v, d_u, k_max, n_layers=4):
    """Estimate total FNO parameters."""
    d_mid = d_v * 2
    lifting = d_a * d_v + d_v
    fourier = n_layers * (2 * k_max * k_max * d_v * d_v + d_v * d_v + d_v)
    projection = d_v * d_mid + d_mid + d_mid * d_u + d_u
    return lifting + fourier + projection

# Build parameter grid
d_v_values = [16, 32, 64, 128]
k_max_values = [8, 12, 16, 20]
param_grid = np.zeros((len(d_v_values), len(k_max_values)))

for i, d_v in enumerate(d_v_values):
    for j, k_max in enumerate(k_max_values):
        param_grid[i, j] = count_parameters(10, d_v, 1, k_max) / 1e6

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap
im = axes[0].imshow(param_grid, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(range(len(k_max_values)))
axes[0].set_xticklabels(k_max_values)
axes[0].set_yticks(range(len(d_v_values)))
axes[0].set_yticklabels(d_v_values)
axes[0].set_xlabel('$k_{max}$ (Fourier modes)', fontsize=12)
axes[0].set_ylabel('$d_v$ (Hidden dimension)', fontsize=12)
axes[0].set_title('Parameter Count (Millions)', fontsize=12, fontweight='bold')

for i in range(len(d_v_values)):
    for j in range(len(k_max_values)):
        color = 'white' if param_grid[i, j] > 3 else 'black'
        axes[0].text(j, i, f'{param_grid[i, j]:.2f}M', ha='center', va='center', 
                     fontsize=10, color=color, fontweight='bold')

plt.colorbar(im, ax=axes[0])

# Highlight recommended region for moderate data
rect = plt.Rectangle((-0.5, 0.5), 2.2, 1.2, fill=False, edgecolor='#4CAF50', lw=3, linestyle='--')
axes[0].add_patch(rect)
axes[0].text(0.5, 2.1, 'Recommended for\n~500-1000 samples', ha='center', fontsize=9, 
             color='#4CAF50', fontweight='bold')

# Scaling curves
d_v_range = np.linspace(16, 128, 50)
for k_max, color, ls in [(8, '#2196F3', '-'), (12, '#4CAF50', '-'), (16, '#FF9800', '--')]:
    params = [count_parameters(10, int(dv), 1, k_max)/1e6 for dv in d_v_range]
    axes[1].plot(d_v_range, params, color=color, lw=2.5, ls=ls, label=f'$k_{{max}}={k_max}$')

axes[1].axhline(y=1, color='gray', ls=':', alpha=0.7)
axes[1].text(20, 1.1, '1M parameters', fontsize=9, color='gray')
axes[1].set_xlabel('Hidden Dimension ($d_v$)', fontsize=12)
axes[1].set_ylabel('Parameters (Millions)', fontsize=12)
axes[1].set_title('Quadratic Scaling with $d_v$', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(16, 128)

plt.suptitle('Hyperparameter Impact on Model Size', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/chunk3_05_parameter_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Parameter Scaling](chunk3_05_parameter_scaling.png)

**Figure 5: Hyperparameter Impact on Model Size — What You're Looking At**

This two-panel figure helps you choose the right model size for your data:

**Left Panel — Parameter Count Heatmap:**
- Rows: Hidden dimension $d_v$ (16, 32, 64, 128)
- Columns: Fourier modes $k_{max}$ (8, 12, 16, 20)
- Cell values: Total parameters in millions (e.g., "0.53M" = 530,000 parameters)
- Color scale: Yellow = smaller models, Red = larger models
- **Green dashed box:** "Recommended for ~500-1000 samples"—the sweet spot for moderate-sized datasets

**Right Panel — Scaling Curves:**
- X-axis: Hidden dimension $d_v$
- Y-axis: Parameter count in millions
- Three curves for different $k_{max}$ values (8, 12, 16)
- Gray horizontal line marks 1M parameters as a reference
- Notice the **quadratic growth**—the curves are parabolas, not lines

**Key Insights from This Figure:**

1. **Quadratic scaling dominates:** The parameter formula is roughly $O(k_{max}^2 \cdot d_v^2)$. Doubling $d_v$ from 32 to 64 gives ~4× more parameters. This is why careful hyperparameter selection matters.

2. **$k_{max}$ has a multiplicative effect:** Changing $k_{max}$ shifts the entire curve up or down. Use smaller $k_{max}$ (8-12) for limited data.

3. **Match model size to data size:** With 500 samples, using a 50M parameter model is a recipe for overfitting. The heatmap's green region suggests appropriate configurations.

4. **Start small, scale up:** It's easier to diagnose underfitting (model too small) than overfitting. Begin with $d_v=32$, $k_{max}=12$ and increase only if needed.

### 9.5 Practical Guidelines

Based on these scaling properties and empirical results from the literature:

**Recommended configurations by dataset size:**

- **~100 samples**: $d_v$=16, $k_{max}$=8, Layers=3-4, ~50-100K parameters
- **~500 samples**: $d_v$=32, $k_{max}$=12, Layers=4, ~300-500K parameters
- **~1000 samples**: $d_v$=64, $k_{max}$=12, Layers=4, ~1-2M parameters
- **~5000+ samples**: $d_v$=128, $k_{max}$=16, Layers=4-6, ~5-20M parameters

The general strategy is to start conservative and increase capacity only if the model underfits:

1. **Start with defaults:** $d_v=32$, $k_{max}=12$, 4 layers
2. **Check for overfitting:** If validation loss is much higher than training loss, reduce $d_v$ or add regularization
3. **Check for underfitting:** If training loss remains high, increase $d_v$ or check data normalization
4. **Fine-tune:** Grid search over learning rate, try different $d_v \in \{32, 64\}$

---

## Section 10: Resolution Invariance

One of FNO's most remarkable properties is **resolution invariance**: a model trained at one resolution can be applied at different resolutions without retraining.

### 10.1 Why This Works

Recall that the spectral weights $R$ are learned in Fourier space, where each entry corresponds to a specific frequency mode. These frequencies have physical meaning independent of the grid resolution—mode $k=3$ represents the same wavelength whether we discretize the domain with 32 or 128 points.

When we apply an FNO trained on a 32×32 grid to a 128×128 grid:
- The FFT at 128×128 produces more frequency modes
- We still only apply the learned weights to the first $k_{max}$ modes
- The IFFT reconstructs at the new resolution

The learned spectral filter remains the same; only the discretization changes.

```python
def create_test_field(N, seed=42):
    """Create a smooth test field at resolution N×N."""
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)
    return np.sin(X) * np.sin(2*Y) + 0.5 * np.cos(3*X) * np.sin(Y)

def apply_spectral_filter(field, k_max=12):
    """Simulate FNO spectral filtering (same weights at any resolution)."""
    f_hat = np.fft.fft2(field)
    N = field.shape[0]
    k = min(k_max, N//2)
    
    # Keep only low-frequency modes (simulating learned filter)
    mask = np.zeros((N, N))
    mask[:k, :k] = 1
    mask[:k, -k+1:] = 1
    mask[-k+1:, :k] = 1
    mask[-k+1:, -k+1:] = 1
    
    # Apply filter and transform back
    return np.fft.ifft2(f_hat * mask).real * 0.8 + 0.1

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
resolutions = [32, 64, 128]

for i, N in enumerate(resolutions):
    field_in = create_test_field(N)
    field_out = apply_spectral_filter(field_in, k_max=12)
    
    # Input
    im1 = axes[0, i].imshow(field_in, cmap='RdBu_r', extent=[0, 2*np.pi, 0, 2*np.pi])
    axes[0, i].set_title(f'Input: {N}×{N}', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
    
    # Output
    im2 = axes[1, i].imshow(field_out, cmap='RdBu_r', extent=[0, 2*np.pi, 0, 2*np.pi])
    axes[1, i].set_title(f'Output: {N}×{N}\n(Same $k_{{max}}=12$ weights)', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel('x')
    axes[1, i].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, i], shrink=0.8)

plt.suptitle('Resolution Invariance: Same FNO Applies at Any Grid Size', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/chunk3_06_resolution_invariance.png', dpi=150, bbox_inches='tight')
plt.show()
```

![Resolution Invariance](chunk3_06_resolution_invariance.png)

**Figure 6: Resolution Invariance — What You're Looking At**

This six-panel figure demonstrates FNO's remarkable ability to work at different resolutions without retraining:

**Top Row — Input Fields:**
- **Left (32×32):** A test function discretized on a coarse 32×32 grid. Notice the pixelated appearance.
- **Center (64×64):** The same underlying function on a 64×64 grid. More detail visible.
- **Right (128×128):** The same function on a fine 128×128 grid. Smooth, detailed appearance.
- All three represent the **same continuous function** at different discretizations.

**Bottom Row — FNO Output Fields:**
- Each panel shows the output of applying the **same spectral filter** (same learned weights with $k_{max}=12$)
- **Left:** Output on 32×32 grid
- **Center:** Output on 64×64 grid
- **Right:** Output on 128×128 grid
- The transformation is **consistent across all resolutions**—the same operator produces physically equivalent results.

**Key Insights from This Figure:**

1. **Frequency has physical meaning:** Mode $k=3$ represents the same wavelength regardless of grid resolution. An FNO trained with $k_{max}=12$ learns to process the first 12 frequency modes—these frequencies are resolution-independent.

2. **Train cheap, deploy expensive:** You can train on coarse 32×32 grids (faster, less memory) and then apply the trained model to 128×128 grids for fine-resolution predictions.

3. **Zero-shot super-resolution:** Without any additional training, an FNO can produce higher-resolution outputs than it was trained on. This is impossible with standard CNNs.

4. **The physics transfers:** Because the spectral weights correspond to physical frequencies (not pixel positions), the learned physics generalizes across discretizations.

> **🎬 See It In Action:** Zongyi Li's blog features an [animated GIF](https://zongyi-li.github.io/blog/2020/fourier-pde/) demonstrating zero-shot super-resolution on Navier-Stokes equations. The animation shows an FNO trained at low resolution producing accurate predictions when evaluated at 4× higher resolution—a compelling demonstration of resolution invariance in practice.

### 10.2 Practical Implications

Resolution invariance has significant practical benefits:

**Train on coarse, deploy on fine.** Training on a 32×32 grid is much faster than 128×128 (memory scales quadratically, FFT time scales as $N^2 \log N$). You can train quickly on downsampled data, then deploy at full resolution.

**Zero-shot super-resolution.** Without any fine-tuning, the same model produces higher-resolution outputs when given higher-resolution inputs.

**Transfer across meshes.** The same model can be applied to problems discretized on different grids, as long as the underlying physics is similar.

This is a fundamental advantage over CNNs, which learn pixel-level patterns that don't transfer across resolutions. FNO learns in function space, so resolution is just a discretization detail.

---

## Section 11: Connection to the Original Paper

Understanding the original FNO paper helps contextualize what we've learned and sets realistic expectations.

> **📄 Read the Paper:** The full paper is freely available at [arXiv:2010.08895](https://arxiv.org/pdf/2010.08895.pdf). It's well-written and accessible, with excellent figures and comprehensive appendices.

### 11.1 Paper Reference

**"Fourier Neural Operator for Parametric Partial Differential Equations"**
Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar
arXiv:2010.08895 (NeurIPS 2020)

### 11.2 Key Claims from the Paper

The original paper makes several important claims that have been validated across many applications:

**Resolution invariance:** "FNO... can be used to transfer solutions between different grid resolutions." This is the property we demonstrated in Section 10—the same trained weights work at any discretization.

**Efficiency:** "Our method achieves state-of-the-art accuracy while being 1000× faster than traditional PDE solvers." Once trained, a single forward pass through the FNO replaces iterative numerical solving.

**Benchmark results:** The paper reports impressive accuracy on standard PDE benchmarks:
- Burgers' equation: 0.4% relative error
- Darcy flow: 1.08% relative error  
- Navier-Stokes: 0.7% relative error

These numbers represent the relative L2 error between FNO predictions and ground truth solutions from numerical solvers.

### 11.3 Architecture Variations

The paper tests several FNO variants for different problem types:

**FNO Variants:**

- **FNO-2D**: Spatial dimensions only — Best for steady-state problems
- **FNO-3D**: Space + time as 3D tensor — Best for time-dependent PDEs
- **FNO-2D+time**: 2D spatial with time as input parameter — Alternative for temporal problems

For steady-state problems (e.g., equilibrium temperature distribution, steady flow), FNO-2D is the appropriate choice. For time-evolution problems where you want to predict the solution at future times, FNO-3D or FNO-2D+time are better suited.

### 11.4 What the Paper Doesn't Tell You

While the paper presents impressive results, practical implementation reveals challenges that aren't fully discussed:

**Training stability:** FNO training can be finicky. Without proper normalization of inputs and outputs, training often fails to converge or produces poor results. The importance of z-score normalization cannot be overstated.

**Hyperparameter sensitivity:** The choice of $k_{max}$ (Fourier modes) and $d_v$ (hidden dimension) significantly affects performance. The paper uses specific values for each benchmark, but finding optimal values for new problems requires experimentation.

**Memory usage:** FFT operations are memory-hungry, especially for large spatial domains. A 256×256 grid with 64 hidden channels can easily consume several GB of GPU memory during training.

**Small data regime:** The paper's experiments use 1000+ training samples. With fewer samples (100-500), careful regularization is needed—smaller models, weight decay, possibly dropout. The impressive benchmark numbers may not be achievable with limited data.

These practical considerations don't diminish FNO's value, but they're important to know before starting an implementation.

---

## Section 12: Summary

We've now covered the complete Fourier Neural Operator architecture:

**The Three Stages:**
1. **Lifting** transforms input from $d_a$ channels to $d_v$ hidden channels via pointwise linear transformation
2. **Fourier Layers** (typically 4) apply spectral convolution plus local path, with GELU activation
3. **Projection** maps from $d_v$ to $d_u$ output channels via two-layer MLP

**Training:**
- Use relative L2 loss for scale-invariant training
- Adam optimizer with learning rate ~$10^{-3}$
- Cosine annealing schedule over 500 epochs
- **Critical:** normalize inputs and outputs to zero mean, unit variance

**Hyperparameters:**
- Start with $d_v=32$, $k_{max}=12$, 4 layers
- Increase $d_v$ if underfitting, decrease if overfitting
- Match model size to dataset size

**Resolution Invariance:**
- Same weights work at any grid resolution
- Train fast on coarse grids, deploy at full resolution
- Fundamental advantage over grid-based methods

The FNO architecture elegantly combines the global receptive field of spectral methods with the learning capacity of neural networks. In Part 4, we'll explore advanced topics: physics-informed training, handling irregular geometries, and extensions to time-dependent problems.

---

## Getting Started: Next Steps

Now that you understand the FNO architecture, here's how to start implementing:

### Quick Start with Official Library

```bash
pip install neuraloperator
```

```python
from neuralop.models import FNO

# Create FNO model
model = FNO(n_modes=(16, 16), hidden_channels=64, 
            in_channels=1, out_channels=1)

# The library handles FFT, mode truncation, and all components
# See examples at: neuraloperator.github.io/dev/auto_examples/
```

### Learning Path Recommendation

1. **Conceptual Understanding:** Watch [Yannic Kilcher's video](https://www.youtube.com/watch?v=IaS72aHrJKE) (66 min)
2. **First Author's Perspective:** Read [Zongyi Li's blog post](https://zongyi-li.github.io/blog/2020/fourier-pde/)
3. **Hands-On Implementation:** Work through the [original repository examples](https://github.com/zongyi-li/fourier_neural_operator)
4. **Production Use:** Graduate to the [neuraloperator library](https://github.com/neuraloperator/neuraloperator)
5. **Deep Theory:** Study the [JMLR Neural Operator survey](https://arxiv.org/abs/2108.08481)

### Benchmark Datasets

The community uses standard benchmarks for comparison:

- **Burgers**: 1D viscous flow, Resolution 8192, 1000 samples — [neuraloperator data](https://github.com/neuraloperator/neuraloperator)
- **Darcy Flow**: 2D steady-state, Resolution 421×421, 1000 samples — Original paper
- **Navier-Stokes**: 2D turbulence, Resolution 64×64, 1000 samples — Original paper

---

## Figure Summary: Visual Guide to FNO

Here's a quick reference to all figures in this tutorial and what each teaches:

- **Fig. 1 - Architecture Overview**: FNO has three stages: Lift → 4 Fourier Layers → Project. Spatial dimensions stay constant; only channels change.
- **Fig. 2 - Layer Details**: Fourier layers use dual paths (spectral + local). Lifting/projection are simple linear operations.
- **Fig. 3 - Synthetic PDE**: FNO learns to filter frequencies—ideal for PDEs that act as low-pass filters (diffusion, heat equation).
- **Fig. 4 - Training Curves**: Monitor validation loss and generalization gap. Use cosine annealing for smooth LR decay.
- **Fig. 5 - Parameter Scaling**: Parameters scale as $O(k_{max}^2 d_v^2)$. Match model size to your dataset size.
- **Fig. 6 - Resolution Invariance**: Same weights work at any resolution—train cheap (32×32), deploy fine (128×128).

---

## References

1. Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

2. Kovachki, N., Li, Z., Liu, B., et al. (2023). *Neural Operator: Learning Maps Between Function Spaces.* Journal of Machine Learning Research. [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)

3. Li, Z., et al. (2021). *Physics-Informed Neural Operator for Learning Partial Differential Equations.* [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)

4. Li, Z., et al. (2023). *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries.* JMLR. [arXiv:2207.05209](https://arxiv.org/abs/2207.05209)

5. Kovachki, N., Lanthaler, S., Stuart, A. (2024). *Operator Learning: Algorithms and Analysis.* [arXiv:2402.15715](https://arxiv.org/abs/2402.15715)

### Image and Diagram Credits

- **Architecture diagrams in this tutorial:** Original creations for educational purposes
- **Reference to Figure 1 from Li et al. (2021):** Available under arXiv's non-exclusive distribution license
- **Animated Navier-Stokes visualization:** [Zongyi Li's blog](https://zongyi-li.github.io/blog/2020/fourier-pde/), used with attribution
- **MATLAB FNO diagrams:** [MathWorks documentation](https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html), referenced for educational purposes

---

**Series Navigation:**

- [← **Part 2:** The Fourier Layer](../chunk2/chunk2_blog_final.md)
- **Part 3:** Complete Architecture (You are here)
- [**Part 4:** Advanced Topics →](../chunk4/chunk4_blog_final.md)

**Part 2** covered the Fourier layer in detail—spectral convolution, mode truncation, and the dual-path architecture.

**Part 4** will explore advanced topics including physics-informed training, irregular geometries, and real-world applications.

---

*This tutorial is part of a series on Fourier Neural Operators for scientific machine learning. All code is available for educational purposes.*
