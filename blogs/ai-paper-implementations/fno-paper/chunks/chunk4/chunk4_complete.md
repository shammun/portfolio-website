# Fourier Neural Operator: Part 4
## Advanced Topics: Physics-Informed Learning, Architectural Variants, and Real-World Applications

---

**Series Navigation:** [← Part 3: Complete Architecture](chunk3_blog_final.md) | **Part 4: Advanced Topics** | [Part 5: Implementation →](chunk5_blog_final.md)

---

## Introduction

You have now mastered the fundamentals:

- **Part 1:** Fourier transforms, convolution theorem, spectral methods
- **Part 2:** Spectral convolution, mode truncation, the Fourier layer
- **Part 3:** Complete FNO architecture (lifting → Fourier layers → projection)

Now we venture into **advanced territory**:

1. Physics-Informed Neural Operators (PINO) — combining data with physical laws
2. Advanced architectures (U-FNO, Factorized FNO, AFNO)
3. Weather and climate foundation models
4. Comparison with other neural operator methods
5. Model interpretation and debugging
6. Common pitfalls and how to avoid them

**This part transforms you from "I understand FNO" to "I can successfully apply FNO to real problems."**

---

## External Resources and Further Reading

Before diving into advanced topics, here are authoritative resources to supplement this tutorial:

### Video Tutorials

- **[Yannic Kilcher: FNO Paper Explained](https://www.youtube.com/watch?v=IaS72aHrJKE)** (66 min): Comprehensive walkthrough of the original FNO paper with mathematical details
- **[ICML 2024: Neural Operators Tutorial](https://slideslive.com/icml-2024)** (~2 hours): Academic foundations covering FNO, DeepONet, and PINO
- **[Steve Brunton: Fourier Series](https://www.youtube.com/watch?v=r6sGWTCMz2k)** (15 min): Excellent prerequisite on Fourier fundamentals

### Key Papers

- **Physics-Informed Neural Operator (PINO)** (2021): [arXiv:2111.03794](https://arxiv.org/abs/2111.03794) - Adding physics constraints to FNO
- **Adaptive Fourier Neural Operator (AFNO)** (2022): [arXiv:2111.13587](https://arxiv.org/abs/2111.13587) - Attention-weighted frequency selection
- **Factorized Fourier Neural Operators** (2023): [arXiv:2111.13802](https://arxiv.org/abs/2111.13802) - Parameter-efficient FNO variants
- **U-FNO for Multiphase Flow** (2022): [arXiv:2109.03697](https://arxiv.org/abs/2109.03697) - U-Net style skip connections
- **FourCastNet** (2022): [arXiv:2202.11214](https://arxiv.org/abs/2202.11214) - Global weather prediction with AFNO
- **Neural Operator Survey** (2024): [arXiv:2309.15325](https://arxiv.org/abs/2309.15325) - Comprehensive review in Nature Reviews Physics

### Code Repositories

- **[neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)** (3.1k+ stars): Official PyTorch library for FNO, PINO, TFNO
- **[neuraloperator/physics_informed](https://github.com/neuraloperator/physics_informed)** (500+ stars): PINO implementation for Burgers, Darcy, Navier-Stokes
- **[NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet)** (640+ stars): NVIDIA's weather prediction model
- **[google-deepmind/graphcast](https://github.com/google-deepmind/graphcast)** (6.4k+ stars): DeepMind's graph-based weather model

### Authoritative Blogs

- [Zongyi Li's FNO Blog](https://zongyi-li.github.io/blog/2020/fourier-pde/) — First author's introduction with animations
- [NVIDIA Modulus Documentation](https://docs.nvidia.com/deeplearning/modulus/) — Industrial-strength neural operator implementations
- [NeuralOperator Theory Guide](https://neuraloperator.github.io/dev/theory_guide/fno.html) — Official library documentation

---

## Section 1: Training FNO Effectively

### 1.1 Data Preparation — The Foundation of Success

Unlike image classification (where networks are robust to variations), neural operators are sensitive to:

- **Scale mismatches** between input features
- **Coordinate inconsistencies** in spatial data
- **Missing data patterns** that differ between train/test
- **Domain boundaries** that affect Fourier transforms

#### Step-by-Step Data Pipeline

**Step 1: Spatial Alignment**

All input features must be on the **same grid**:

```
Feature 1: Resolution R, Projection P, Extent E
Feature 2: Must match EXACTLY — same R, P, E
...
Feature N: Same grid as Feature 1
```

For climate and remote sensing applications, this typically means reprojecting all data to a common grid (e.g., 0.25° for global weather, or a fixed-resolution raster for regional problems).

**Step 2: Feature Normalization**

**Critical:** Normalize EACH feature channel independently.

```
For channel c in 1...N:
    μ_c = mean of channel c over all training samples
    σ_c = std of channel c over all training samples
    channel_c_normalized = (channel_c - μ_c) / σ_c
```

**Why per-channel?** Different physical variables have vastly different scales:
- Temperature might range [250, 320] K
- Pressure might range [950, 1050] hPa  
- Normalized indices might range [-1, 1]

Without normalization, large-valued features dominate gradient updates.

**Step 3: Output Normalization**

Also normalize the target:
```
y_normalized = (y - μ_y) / σ_y
```

Store μ_y and σ_y for denormalization at inference.

**Step 4: Train/Validation/Test Split**

For spatial data, consider different splitting strategies:

**Random samples:**
- Pros: Simple, more data per split
- Cons: May overfit to specific conditions

**Temporal split:**
- Pros: Tests generalization to new times
- Cons: May miss seasonal patterns

**Spatial split:**
- Pros: Tests generalization to new regions
- Cons: Harder to implement

**Recommended:** 70% training, 15% validation (for hyperparameter tuning), 15% test (held out until final evaluation).

### 1.2 Batch Construction for FNO

#### Full-Field vs. Patch-Based Training

**Option 1: Full Field Training**
- Input: Entire spatial domain (e.g., 256×256 or 512×512)
- Batch: Multiple samples (different times or conditions)
- Pros: Captures full global context
- Cons: Memory intensive, limited batch size

**Option 2: Patch-Based Training**
- Extract patches (e.g., 64×64) from larger domain
- More samples per epoch
- Pros: Larger batches, enables data augmentation
- Cons: Loses some long-range context at patch boundaries

**Recommendation:** Start with full-field at reduced resolution (64×64), scale up for final training.

#### Leveraging Resolution Invariance

FNO's resolution invariance enables a powerful training strategy:

```
Epoch 1-100: Train on 64×64 (fast iteration, large batches)
Epoch 101-200: Fine-tune on 128×128 (refine details)
Final: Evaluate on full resolution (256×256 or higher)
```

This exploits the fact that learned spectral weights transfer across resolutions.

### 1.3 Optimization Settings

#### Learning Rate

**Recommended starting point:** 1e-3

**Learning rate schedule options:**

1. **Cosine Annealing** (recommended):
   $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{\pi t}{T}))$$
   - Smooth decay from 1e-3 to 1e-6
   - Works well for most problems

2. **Step Decay:**
   - Reduce by 0.5 every 100 epochs
   - Simple but effective

3. **ReduceLROnPlateau:**
   - Reduce when validation loss plateaus
   - Adaptive to training dynamics

#### Weight Decay (L2 Regularization)

**Recommended:** 1e-4

Helps prevent overfitting, especially important with limited training data.

#### Gradient Clipping

**Recommended:** Clip gradients to max norm 1.0

Prevents exploding gradients during early training when FFT operations can amplify gradients.

#### Optimizer

**AdamW** (decoupled weight decay) is the standard choice:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### 1.4 Regularization Techniques

#### Dropout in FNO

Apply dropout to the **local path** (W), not the spectral path:

```python
v_out = activation(spectral_conv(v) + dropout(W @ v) + b)
```

**Why not spectral?** Dropping Fourier coefficients randomly disrupts the smoothness prior that makes FNO effective.

**Recommended dropout rate:** 0.1-0.2

#### Spectral Regularization

Add penalty on spectral weight magnitudes:
$$\mathcal{L}_{spectral} = \lambda \sum_k |R(k)|^2$$

This encourages the network to use fewer/smaller spectral weights, acting as a form of frequency-domain regularization.

#### Data Augmentation for Spatial Data

**Valid augmentations:**
- Random horizontal/vertical flips (if physics is symmetric)
- 90°, 180°, 270° rotations (if physics is isotropic)
- Small random crops + resize

**Invalid augmentations:**
- Arbitrary rotations (breaks grid alignment for FFT)
- Large translations (shifts boundary conditions)
- Color jittering (not applicable to physical fields)

---

## Section 2: Physics-Informed Neural Operators (PINO)

> **📖 Reference:** The foundational PINO paper is Li et al. (2021), "Physics-Informed Neural Operator for Learning Partial Differential Equations" — [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)

### 2.1 The Idea: Combining Data and Physics

Standard FNO learns purely from data:
$$\mathcal{L} = \mathcal{L}_{data} = \frac{1}{N}\sum_i \|\mathcal{G}_\theta(a^{(i)}) - u^{(i)}\|^2$$

**Physics-Informed FNO** adds physical constraints:
$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{physics} \mathcal{L}_{physics}$$

This hybrid approach combines the flexibility of data-driven learning with the generalization power of physics.

![PINO Concept](figures/chunk4_01_pino_concept.png)

**Figure 1: Physics-Informed Neural Operator Concept — Understanding the Diagram**

Take a moment to study this three-panel comparison—it captures the essential difference between "learning from data alone" and "learning from data plus physics."

**Left Panel — Standard FNO (Data-Driven Only):**
Think of this as a student who only learns from example problems. Input $a(x)$ flows into the FNO "black box," producing prediction $\hat{u}(x)$. The only feedback the model receives is: "How far was your prediction from the correct answer?" This is the data loss $\mathcal{L}_{data}$. The model has no idea *why* certain answers are correct—it just memorizes patterns.

**Center Panel — PINO (Data + Physics):**
Now imagine that same student also knows the underlying rules of physics. The architecture is identical, but the learning signal is richer. The orange arrow labeled "PDE Residual" represents a second question the model must answer: "Does your prediction actually satisfy the laws of physics?" The combined loss $\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$ forces the model to produce predictions that are both accurate AND physically consistent.

**Right Panel — The Physics Toolbox:**
These four colored boxes show the different types of physics knowledge you can encode:
1. **PDE Residual (orange):** "Does my prediction satisfy the heat equation / wave equation / Navier-Stokes?"
2. **Conservation Laws (yellow):** "Is mass/energy conserved in my prediction?"
3. **Boundary Conditions (purple):** "Does my prediction match known values at the edges?"
4. **Smoothness Prior (blue):** "Is my prediction reasonably smooth, as physical fields should be?"

> **💡 Key Takeaway:** PINO is like giving an AI both a textbook of examples AND the underlying theory. With limited training data, physics constraints act as powerful regularizers—they eliminate the vast space of "predictions that fit the data but violate physical laws." This is especially valuable when data collection is expensive or dangerous (turbulence simulations, nuclear reactor modeling, etc.).

### 2.2 Types of Physics Constraints

#### Type 1: PDE Residual Loss

If you know the governing PDE, penalize violations:

**Example: Heat equation**
$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T + Q$$

**PDE residual loss:**
$$\mathcal{L}_{PDE} = \left\| \frac{\partial \hat{T}}{\partial t} - \alpha \nabla^2 \hat{T} - Q \right\|^2$$

The derivatives can be computed using automatic differentiation or spectral methods.

#### Type 2: Conservation Laws

**Energy conservation:**
$$\mathcal{L}_{conservation} = \left| \int_\Omega \hat{u} \, dA - \int_\Omega u_{true} \, dA \right|^2$$

This ensures predicted fields conserve relevant quantities (mass, energy, momentum).

#### Type 3: Boundary Conditions

Enforce known behavior at boundaries:
$$\mathcal{L}_{BC} = \| \hat{u}|_{\partial\Omega} - u_{BC} \|^2$$

For example, in heat transfer problems, you might know the temperature at certain boundaries.

#### Type 4: Smoothness Prior

Physical fields should typically be smooth:
$$\mathcal{L}_{smooth} = \| \nabla \hat{u} \|^2$$

This is somewhat redundant with FNO's spectral bias but can help stabilize training.

### 2.3 Implementing PINO

#### Computing Spatial Derivatives

**Finite differences (simple):**
$$\nabla^2 u \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}$$

**Spectral method (elegant):**
$$\nabla^2 u = \mathcal{F}^{-1}[-(k_x^2 + k_y^2) \cdot \mathcal{F}[u]]$$

The Laplacian in Fourier space is just multiplication by $-|k|^2$—this is both more accurate and computationally efficient.

#### Training Loop with Physics Loss

```python
for batch in dataloader:
    pred = model(batch.input)
    
    # Data loss
    loss_data = F.mse_loss(pred, batch.target)
    
    # Physics loss (example: smoothness via Laplacian)
    laplacian = compute_spectral_laplacian(pred)
    loss_physics = torch.mean(laplacian**2)
    
    # Combined loss
    loss = loss_data + lambda_physics * loss_physics
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Choosing λ_physics

**Too small:** Physics has no effect
**Too large:** Model focuses on physics, ignores data

**Strategy:** Start small, increase gradually:
- Epochs 1-50: λ = 0.01
- Epochs 51-100: λ = 0.1
- Epochs 101+: λ = 1.0

Or use automatic balancing based on loss magnitudes (GradNorm or similar).

### 2.4 Benefits of Physics-Informed Training

- **Better generalization**: Physics constraints guide extrapolation beyond training data
- **Data efficiency**: Learn more from fewer samples
- **Physical consistency**: Predictions obey known laws
- **Reduced overfitting**: Physics acts as regularization
- **Interpretability**: Can verify predictions satisfy physics

> **💡 When PINO Shines:** Physics constraints are particularly valuable with limited training data. If you have abundant data, pure data-driven FNO may suffice. If data is scarce, PINO can dramatically improve generalization.

---

## Section 3: Advanced FNO Architectures

The original FNO architecture works well for many problems, but several variants address specific limitations.

![Advanced Architectures](figures/chunk4_02_advanced_architectures.png)

**Figure 2: Advanced FNO Architecture Variants — Understanding the Diagram**

This four-panel comparison is your roadmap for choosing the right FNO variant. Each architecture addresses a specific limitation of the standard FNO.

**Top-Left — Standard FNO:**
The baseline: four sequential Fourier layers (FL1 → FL2 → FL3 → FL4). Simple, elegant, and surprisingly powerful. The label "Fixed $k_{max}$ truncation, sequential processing" tells you both its strength (simplicity) and limitation (fixed frequency resolution). If this works for your problem, there's no need to complicate things.

**Top-Right — U-FNO (U-Net + FNO):**
Notice the hourglass shape with green dashed "skip connections" arcing over the top. This borrows the brilliant insight from U-Net in image segmentation: when you downsample (compress), you lose fine details; skip connections let you recover them later. The label "Multi-scale features + sharp detail preservation" captures the benefit. Use U-FNO when your output has *both* smooth large-scale patterns AND sharp discontinuities (think: fluid interfaces, shock waves, terrain boundaries).

**Bottom-Left — Factorized FNO (F-FNO):**
This panel reveals a mathematical trick that saves enormous memory. The red box shows the standard approach: spectral weights $R \in \mathbb{C}^{k^2 \times d^2}$—roughly 26 million parameters! The blue box shows the factorization: $R(k_x, k_y) = R_x(k_x) \cdot R_y(k_y)$—just 1.3 million parameters (20× reduction). The green checklist confirms: deeper networks, less memory, same accuracy. Use F-FNO when you're memory-constrained or need to scale to very high resolutions.

**Bottom-Right — Adaptive FNO (AFNO):**
This architecture powers NVIDIA's FourCastNet weather model. The flow diagram shows: FFT → "Attention + Soft Threshold" → IFFT → MLP. The key innovation is that learnable attention weights $\alpha(k)$ let the model *decide* which frequencies matter, rather than using a fixed $k_{max}$ cutoff. The soft thresholding adds sparsity—irrelevant frequencies get zeroed out entirely.

> **💡 Decision Guide:**
> - **Limited data or memory?** → Factorized FNO (10-20× fewer parameters)
> - **Sharp gradients or discontinuities?** → U-FNO (skip connections preserve details)
> - **Very high resolution or global scale?** → AFNO (adaptive frequency selection)
> - **None of the above?** → Standard FNO (simpler is often better)

### 3.1 U-FNO: U-Net + FNO

> **📖 Reference:** Wen et al. (2022), "U-FNO—An enhanced Fourier neural operator for multiphase flow" — [arXiv:2109.03697](https://arxiv.org/abs/2109.03697)

#### Motivation

Standard FNO uses mode truncation, losing high-frequency information. U-Net's skip connections preserve details across scales.

#### Architecture

```
Encoder (FNO path):
    Input → FL1 → Downsample → FL2 → Downsample → FL3 (bottleneck)

Decoder (with skip connections):
    FL3 → Upsample → Concat(FL2) → FL4 → Upsample → Concat(FL1) → FL5 → Output
```

The key innovation is adding a parallel U-Net path that captures spatial features lost in spectral truncation.

#### When to Use U-FNO

- When output has both smooth trends AND sharp features
- Problems with discontinuities or steep gradients
- Multi-scale phenomena requiring both local and global context

### 3.2 Factorized FNO (F-FNO)

> **📖 Reference:** Tran et al. (2023), "Factorized Fourier Neural Operators" — [arXiv:2111.13802](https://arxiv.org/abs/2111.13802)

#### Motivation

Standard FNO spectral weights: $R \in \mathbb{C}^{k_{max}^2 \times d_v^2}$

This scales poorly with $k_{max}$ and $d_v$.

#### Factorization

Instead of full tensor R, use separable form:
$$R(k_x, k_y) = R_x(k_x) \cdot R_y(k_y)$$

Or with mixing matrices:
$$R(k_x, k_y) = U \cdot \text{diag}(R_x(k_x)) \cdot V \cdot \text{diag}(R_y(k_y)) \cdot W$$

#### Parameter Reduction

**k_max=12, d_v=64:**
- Standard FNO: 2.4M parameters
- Factorized FNO: 0.2M parameters
- Reduction: 12×

**k_max=20, d_v=128:**
- Standard FNO: 26M parameters
- Factorized FNO: 1.3M parameters
- Reduction: 20×

**Trade-off:** Fewer parameters means reduced expressivity, but F-FNO often matches standard FNO performance while enabling much deeper networks.

### 3.3 Adaptive Fourier Neural Operator (AFNO)

> **📖 Reference:** Guibas et al. (2022), "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers" — [arXiv:2111.13587](https://arxiv.org/abs/2111.13587)

#### Motivation

FNO uses fixed mode truncation—the same frequencies are kept everywhere. What if different spatial regions need different frequencies?

#### Approach

Learn attention weights for Fourier modes:
$$\hat{w}(k) = \text{softmax}(\alpha(k)) \cdot R(k) \cdot \hat{v}(k)$$

Where $\alpha(k)$ is a learned attention weight over frequencies.

Additionally, AFNO uses **soft thresholding** to sparsify frequency modes:
$$\hat{v}_{sparse}(k) = \text{sign}(\hat{v}(k)) \cdot \max(|\hat{v}(k)| - \tau, 0)$$

#### Benefits

- Adapts to problem structure automatically
- Can focus on relevant frequencies per region
- Scales to very high resolution (used in weather models)
- Powers NVIDIA's FourCastNet weather prediction system

### 3.4 Multi-Scale FNO

#### Motivation

Many physical problems have patterns at multiple scales:
- Global scale: Planetary waves, large-scale circulation
- Regional scale: Weather systems, fronts
- Local scale: Convection, turbulence

#### Architecture

```
Branch 1 (coarse): k_max = 4, captures large-scale patterns
Branch 2 (medium): k_max = 12, captures regional patterns  
Branch 3 (fine): k_max = 24, captures local patterns

Output = Combine(Branch1, Branch2, Branch3)
```

**Implementation options:**
- Parallel FNO branches with different $k_{max}$, concatenate features
- Hierarchical processing (coarse → fine refinement)
- Multi-grid inspired approaches (MG-TFNO)

---

## Section 4: Weather and Climate Foundation Models

Neural operators have achieved breakthrough results in weather prediction, demonstrating their power for real-world applications.

> **📊 Key Achievement:** Modern AI weather models achieve **1000-10,000× speedup** over traditional numerical weather prediction while matching or exceeding accuracy.

![Weather Foundation Models](figures/chunk4_03_weather_models.png)

**Figure 3: AI Weather Foundation Models — Understanding the Landscape**

This comparison table tells one of the most exciting stories in modern AI: in just three years (2022-2025), neural networks went from "interesting research" to "beating the world's best weather forecasting systems."

**Reading the Table:**
Each row represents a major AI weather model. The columns show: organization, year of release, underlying architecture, inference speed (time to generate a 24-hour forecast), and model size. The color-coding helps distinguish different approaches.

**What Each Model Represents:**

**FourCastNet (Green, NVIDIA 2022):** The pioneer. Built on AFNO (Adaptive Fourier Neural Operator) architecture—directly relevant to this tutorial. Generates forecasts in 0.001 seconds (yes, one millisecond). With ~100M parameters, it proved that AI weather prediction was not just possible but practical.

**Pangu-Weather (Red, Huawei 2023):** The game-changer. First AI model to *officially outperform* the European Centre for Medium-Range Weather Forecasts (ECMWF)—the gold standard of weather prediction. Uses 3D Transformers to capture atmospheric vertical structure.

**GraphCast (Blue, DeepMind 2023):** The elegant approach. Uses graph neural networks on an icosahedral mesh (like a subdivided 20-sided die)—no rectangular grid distortions near the poles. With only 37M parameters, it's remarkably efficient.

**ClimaX (Cyan, Microsoft 2023):** The foundation model approach. Pre-trained on decades of climate simulations (CMIP6), then fine-tuned for weather. The "Variable ViT" architecture handles different variables (temperature, pressure, humidity) as separate tokens.

**Aurora (Purple, Microsoft 2025):** The behemoth. At 1.3 billion parameters, it's a true "foundation model" spanning weather, climate, and air quality prediction. Represents where the field is heading.

**The Yellow Box at the Bottom:**
This is the headline: ALL of these models achieve 1,000-10,000× speedup over traditional numerical weather prediction (which requires massive supercomputers running for hours). This isn't just an incremental improvement—it's a paradigm shift.

> **💡 Key Takeaways:**
> 1. **Speed matters:** Traditional weather prediction takes hours on supercomputers. AI does it in seconds on a single GPU.
> 2. **Architectures are diverse:** FNO, Transformers, and Graph Networks all work—the "best" approach is still being discovered.
> 3. **This is your opportunity:** Weather/climate AI is still young. The techniques in this tutorial (FNO, PINO) are directly used in production models.

### 4.1 FourCastNet (NVIDIA)

> **📖 Reference:** Pathak et al. (2022), "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators" — [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)

**Architecture:** AFNO-based Vision Transformer
- Processes 0.25° global ERA5 data (720×1440 grid)
- 26 atmospheric variables at multiple pressure levels
- Produces 7-day forecasts in under 2 seconds

**Key Innovation:** Uses AFNO layers as efficient token mixers, enabling transformer-scale models for high-resolution weather data.

**Code:** [github.com/NVlabs/FourCastNet](https://github.com/NVlabs/FourCastNet)

### 4.2 Pangu-Weather (Huawei)

> **📖 Reference:** Bi et al. (2023), "Accurate medium-range global weather forecasting with 3D neural networks" — *Nature* (2023)

**Architecture:** 3D Earth-Specific Transformer
- First AI model to outperform operational numerical weather prediction
- 10,000× faster than traditional methods (1.4 seconds for 24-hour forecast)

**Key Innovation:** 3D attention across latitude, longitude, and pressure levels, with Earth-specific positional encodings.

**Code:** [github.com/198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather)

### 4.3 GraphCast (DeepMind)

> **📖 Reference:** Lam et al. (2023), "Learning skillful medium-range global weather forecasting" — *Science* (2023)

**Architecture:** Graph Neural Network on icosahedral mesh
- Outperforms ECMWF HRES on 90%+ of verification targets
- ~1000× more energy-efficient than traditional NWP

**Key Innovation:** Multi-mesh graph structure that naturally handles spherical geometry without projection distortions.

**Code:** [github.com/google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) (6.4k+ stars)

### 4.4 ClimaX (Microsoft)

> **📖 Reference:** Nguyen et al. (2023), "ClimaX: A foundation model for weather and climate" — ICML 2023

**Architecture:** Variable-tokenizing transformer
- Pre-trains on CMIP6 climate simulation data
- Fine-tunes for weather forecasting, downscaling, climate projections

**Key Innovation:** Variable tokenization allows handling different input/output configurations without architecture changes.

**Code:** [github.com/microsoft/ClimaX](https://github.com/microsoft/ClimaX)

### 4.5 Aurora (Microsoft)

> **📖 Reference:** Bodnar et al. (2025), "Aurora: A Foundation Model for the Earth System" — *Nature* (2025)

**Architecture:** 1.3B parameter foundation model
- Trained on 1+ million hours of diverse atmospheric data
- Produces 5-day forecasts including air pollution
- 5,000× speedup over traditional methods

**Key Innovation:** Multi-task learning across weather, climate, and atmospheric chemistry.

**Code:** [github.com/microsoft/aurora](https://github.com/microsoft/aurora)

---

## Section 5: Comparison with Other Methods

![Neural Operator Comparison](figures/chunk4_04_operator_comparison.png)

**Figure 4: Neural Operator Methods — Choosing the Right Tool**

This three-panel diagram answers a crucial question: "FNO sounds great, but when should I use something else?" Each panel shows a different neural operator architecture optimized for different problem types.

**Left Panel — FNO (Fourier Neural Operator):**
The familiar friend from this tutorial. Notice the regular grid of blue squares—FNO *requires* rectangular, uniformly-spaced data. The flow: Regular Grid → FFT → Spectral weights $R \cdot \hat{v}$ → Output. The checkmarks and X summarize perfectly:
- ✓ **O(N log N)** — blazingly fast due to FFT
- ✓ **Resolution invariant** — train at one resolution, test at another
- ✗ **Regular grids only** — can't handle irregular boundaries or adaptive meshes

**Center Panel — DeepONet (Deep Operator Network):**
A fundamentally different approach. Two separate neural networks: a green "Branch" network that encodes the input function $a$, and a purple "Trunk" network that encodes the query location $x$. Their outputs combine via dot product: $u(x) = \sum_k b_k \cdot t_k$. The key insight:
- ✓ **Any domain shape** — no rectangular restriction
- ✓ **Point evaluation** — query the solution at any continuous location
- ✗ **Separate location queries** — predicting a full field requires many forward passes

Think of DeepONet like GPS: give it any coordinates, get an answer. Think of FNO like a satellite image: get the whole picture at once, but only on the grid.

**Right Panel — GNO (Graph Neural Operator):**
For when your domain looks like a mesh of connected nodes (finite elements, molecular structures, road networks). The formula $v_i \leftarrow \sum_j \kappa(x_i, x_j) v_j$ is message passing—each node aggregates information from neighbors through learned kernels.
- ✓ **Irregular meshes** — naturally handles unstructured grids
- ✓ **Complex geometry** — aircraft wings, turbine blades, anatomical structures
- ✗ **O(N × neighbors)** — scales with connectivity, not just grid size

> **💡 Decision Framework:**
> | Your Data Looks Like... | Use This |
> |------------------------|----------|
> | Regular rectangular grid (satellite imagery, weather data, simulation output) | **FNO** |
> | Need to query arbitrary continuous points | **DeepONet** |
> | Unstructured mesh or complex geometry (CFD meshes, molecular graphs) | **GNO** |
> | Very large scale with flexible architecture | **Transformer** variants |

### 5.1 FNO vs. DeepONet

**DeepONet** (Deep Operator Network) uses a different architecture:
- Branch network encodes input function
- Trunk network encodes output location
- Output: $u(x) = \sum_k b_k(a) \cdot t_k(x)$

**Architecture:**
- FNO: FFT-based spectral
- DeepONet: Basis function expansion

**Resolution invariance:**
- FNO: Yes
- DeepONet: Yes

**Global receptive field:**
- FNO: Yes (FFT)
- DeepONet: Depends on branch design

**Domain flexibility:**
- FNO: Regular grids
- DeepONet: Any domain shape

**Implementation:**
- FNO: Moderate complexity
- DeepONet: Simpler

**When to prefer DeepONet:**
- Irregular domains where FFT doesn't apply
- Need point-wise evaluation at arbitrary locations
- Simpler implementation is preferred

> **📖 Reference:** Lu et al. (2021), "Learning nonlinear operators via DeepONet" — *Nature Machine Intelligence*

### 5.2 FNO vs. Graph Neural Operators (GNO)

**Graph Neural Operators:**
- Work on unstructured meshes
- Message passing between nodes
- Natural for complex geometries

**Domain:**
- FNO: Regular grids
- GNO: Any mesh

**Complexity:**
- FNO: O(N log N)
- GNO: O(N × neighbors)

**Resolution:**
- FNO: Must be grid-based
- GNO: Adaptive mesh refinement

**Implementation:**
- FNO: FFT libraries
- GNO: Graph neural network libraries

**When to prefer GNO:**
- Complex domain geometries (aircraft, turbines)
- Adaptive mesh refinement needed
- Inherently graph-structured data

### 5.3 FNO vs. Vision Transformers

**Vision Transformers for PDEs:**
- Self-attention over spatial patches
- Can model long-range dependencies

**Global context:**
- FNO: Yes (FFT)
- Transformer: Yes (attention)

**Complexity:**
- FNO: O(N log N)
- Transformer: O(N²) or O(N log N) with optimizations

**Inductive bias:**
- FNO: Smoothness (spectral)
- Transformer: None (fully data-driven)

**Data efficiency:**
- FNO: Better
- Transformer: Needs more data

**When to prefer Transformers:**
- Very large datasets available
- Complex, multi-modal inputs
- Pre-trained models available for fine-tuning

### 5.4 FNO vs. CNNs

**Receptive field:**
- FNO: Global (immediately)
- CNN: Local (grows with depth)

**Resolution:**
- FNO: Invariant
- CNN: Fixed

**Parameters for global:**
- FNO: O(k_max² d²)
- CNN: O(N² d²) for equivalent kernel

**Inductive bias:**
- FNO: Spectral/smooth
- CNN: Translation equivariance

**For PDE problems:** FNO is more natural—many PDEs have spectral structure that FNO captures directly.

---

## Section 6: Model Interpretation — What Does FNO Learn?

One of the beautiful aspects of FNO is that it's more interpretable than many deep learning models. Because it operates in the frequency domain, we can actually peek inside and understand *what* the model learned. This section shows you how.

![Interpretation Methods](chunk4_05_interpretation.png)

**Figure 5: FNO Interpretation Methods — Opening the Black Box**

This four-panel figure provides a complete toolkit for understanding your trained FNO. Each panel answers a different question about what the model learned.

**Top-Left — Spectral Weight Magnitude $|R(k)|$:**
This heatmap shows the magnitude of learned spectral weights across frequency modes $(k_x, k_y)$. The axes represent frequency in each spatial direction. The color scale goes from dark (low) to bright yellow (high).

*What to look for:* Notice the bright yellow region in the bottom-left corner (low $k_x$, low $k_y$). The annotation "Low freq. = important" points this out. This model learned to pay most attention to large-scale, slowly-varying patterns—exactly what we'd expect for smooth PDE solutions like heat diffusion or steady flows.

*Red flag:* If you see high weights scattered randomly at high frequencies, your model might be fitting noise rather than physics.

**Top-Right — Frequency Importance Analysis:**
This bar chart shows what happens when you zero out high-frequency modes. The x-axis shows the cutoff frequency $k_{cut}$ (modes above this are zeroed). The y-axis shows relative error increase.

*Reading the plot:* When $k_{cut} = 2$ (only keeping the lowest 2 modes), error increases by ~80%—losing mid-frequencies hurts a lot! As $k_{cut}$ increases, the bars get shorter—zeroing out very high frequencies (k > 10) barely affects accuracy. The annotation "Low frequencies are critical!" captures the key message.

*This confirms:* The model genuinely relies on large-scale patterns, not memorizing high-frequency details.

**Bottom-Left — Channel Importance:**
This horizontal bar chart shows feature importance via permutation testing. For each input channel (Temperature, Pressure, Humidity, Wind U/V, Precipitation, Cloud, Radiation), we shuffle that channel's values and measure how much prediction error increases.

*Reading the results:* "Temp" (temperature) has the longest bar (~0.45 error increase)—it's the most important input. "Press" (pressure) is second (~0.25), followed by "Humid." This matches physical intuition: for weather prediction, temperature and pressure are fundamental drivers.

*Scientific validation:* If your model claimed "Radiation" was more important than "Temperature" for predicting weather, you'd know something was wrong!

**Bottom-Right — Gradient Saliency Map:**
This spatial heatmap shows which input locations affect the prediction at a specific point (marked with a star). Brighter colors = higher influence.

*What you're seeing:* The influence is concentrated in a blob around the target point, with intensity falling off with distance. The annotation "Target Point" shows where we're predicting. This confirms FNO has a global receptive field (the entire domain contributes) but with locality bias (nearby regions matter more).

> **💡 Key Takeaways for Model Interpretation:**
> 1. **Spectral weights should be smooth and concentrated at low frequencies** for most physical problems
> 2. **Frequency importance should decay** — low modes critical, high modes expendable
> 3. **Channel importance should match physical intuition** — this validates your model learned real physics
> 4. **Saliency maps should show spatial locality** with global context — pure noise suggests a broken model

### 6.1 Examining Spectral Weights

The spectral weights $R(k)$ reveal which frequencies matter:

```python
# Visualize spectral weight magnitudes
for layer_idx, layer in enumerate(model.fourier_layers):
    R = layer.spectral_weights  # Shape: (k_max, k_max, d_v, d_v)
    R_magnitude = torch.abs(R).mean(dim=(-1, -2))  # Average over channels
    
    plt.subplot(2, 2, layer_idx + 1)
    plt.imshow(R_magnitude.cpu().numpy(), cmap='hot')
    plt.title(f'Layer {layer_idx + 1}: |R(k)|')
    plt.colorbar()
```

**Expected patterns:**
- High weights at low k → large-scale patterns matter
- Lower weights at high k → fine details less important
- Peaks at specific frequencies → periodic structure in the problem

### 6.2 Frequency Importance Analysis

**Method:** Zero out modes at different frequencies, measure prediction degradation.

```python
def frequency_importance(model, test_data, k_cutoffs):
    """Measure importance of different frequency ranges."""
    baseline_error = evaluate(model, test_data)
    importance = {}
    
    for k_cut in k_cutoffs:
        # Zero out high frequencies
        model_modified = modify_spectral_weights(model, k_max=k_cut)
        error_k = evaluate(model_modified, test_data)
        importance[k_cut] = error_k - baseline_error
    
    return importance
```

**Interpretation:**
- Large error increase when zeroing k=1-4 → global patterns critical
- Large error increase when zeroing k=8-12 → regional patterns critical

### 6.3 Channel Importance via Permutation

**Method:** Shuffle one input channel, measure prediction degradation.

```python
def channel_importance(model, test_data, n_channels):
    """Permutation importance for input channels."""
    baseline_error = evaluate(model, test_data)
    importance = {}
    
    for c in range(n_channels):
        # Shuffle channel c across batch
        shuffled_data = shuffle_channel(test_data, channel=c)
        error_c = evaluate(model, shuffled_data)
        importance[c] = error_c - baseline_error
    
    return importance
```

High importance indicates the model relies heavily on that feature.

### 6.4 Gradient-Based Saliency Maps

**Method:** Compute gradient of output with respect to input to see which input locations affect predictions.

```python
def compute_saliency(model, input_field, target_point):
    """Which input locations affect prediction at target_point?"""
    input_field.requires_grad_(True)
    output = model(input_field)
    
    # Gradient of prediction at target_point
    target_value = output[0, target_point[0], target_point[1], 0]
    target_value.backward()
    
    saliency = input_field.grad.abs()
    return saliency
```

**Expected for FNO:** High saliency across the entire domain (global receptive field), with peaks near relevant features.

### 6.5 Physics Consistency Checks

Verify that FNO learns physically meaningful relationships:

**Test 1: Perturbation response**
```python
# Does increasing feature X have expected effect on output?
perturbed_input = input.clone()
perturbed_input[:, :, :, feature_idx] += delta
perturbed_output = model(perturbed_input)

# Check if output changes in expected direction
```

**Test 2: Conservation properties**
```python
# Does total quantity remain conserved?
input_total = input.sum(dim=(1, 2))
output_total = model(input).sum(dim=(1, 2))
conservation_error = (output_total - expected_total).abs()
```

---

## Section 7: Common Pitfalls and Solutions

Before you rush off to train your first FNO, let me save you hours of debugging. These six pitfalls trip up almost everyone—including experienced researchers. Study this figure carefully; it might be the most practically valuable part of this entire tutorial.

![Common Pitfalls](figures/chunk4_06_pitfalls.png)

**Figure 6: Common FNO Training Pitfalls — Learn from Others' Mistakes**

This six-panel figure is a rogues' gallery of training failures. Each panel shows what goes wrong and hints at the solution.

**Top-Left — Forgetting Normalization:**
The red line shows what happens when you feed raw, unnormalized data to an FNO: the loss explodes to $10^{11}$ and then crashes to NaN. The green line shows healthy training with proper normalization—loss decreases smoothly to ~1.

*Why this happens:* Neural networks (including FNO) are sensitive to input scale. If your temperature data ranges from 250-320 Kelvin while your pressure data ranges from 900-1100 hPa, the model will struggle to balance gradients across such different scales.

*The fix:* Normalize each input channel to zero mean and unit variance. Also normalize your targets. Denormalize predictions at the end.

**Top-Middle — Overfitting on Small Data:**
The classic train-val divergence plot. Blue line (training loss) drops to near zero while red line (validation loss) rises after epoch ~100. The orange dashed line marks where you should have stopped. The red shaded "Gap = Overfitting!" region shows wasted training.

*Why this happens:* FNO has many parameters. With limited training samples, it memorizes rather than generalizes.

*The fix:* Early stopping (the orange line), dropout, weight decay, data augmentation, or—best of all—add physics constraints (PINO).

**Top-Right — Mode Count > N/2:**
This bar chart shows a subtle but devastating bug. Green bars show safe $k_{max}$ values; red bars show dangerous ones. The blue dashed lines at $N/2 = 16, 32, 64$ show the maximum valid $k_{max}$ for each grid size.

*Why this matters:* The Nyquist theorem says a signal sampled at $N$ points can only represent frequencies up to $N/2$. If you set $k_{max} = 40$ on a 64×64 grid (where max safe value is 32), you're asking for frequencies that don't exist.

*The fix:* Always ensure $k_{max} \leq \min(N_x, N_y) // 2$. Add an assertion in your code!

**Bottom-Left — Boundary Artifacts:**
The orange-bordered rectangle shows edge artifacts. The annotation "Artifacts at edges!" points to the problem regions.

*Why this happens:* FFT assumes your data is *periodic*—that the right edge connects smoothly to the left edge. Real data rarely satisfies this assumption. The discontinuity at boundaries creates high-frequency artifacts.

*The fix:* Use a larger domain than you need and crop predictions. Or use zero-padding. Or train on periodic boundary conditions when possible.

**Bottom-Middle — Evaluating with the Wrong Metric:**
This is perhaps the most insidious pitfall. The bar chart compares three models on two metrics: Global R² (tall blue bars, ~0.9 for all models) and Spatial R² (shorter green bars, varying from ~0.45 to ~0.75).

*The trap:* Model A has the highest Global R² (0.98!) but the *lowest* Spatial R² (0.45). The annotation says it all: "Model A looks best but is actually worst!"

*Why this happens:* Global R² can be dominated by correctly predicting the mean. A model that predicts "everything is 300K" scores high on Global R² if the average temperature is 300K, even if it captures no spatial patterns at all.

*The fix:* Use Spatial Anomaly R² (predict patterns after removing the spatial mean) for any task where spatial structure matters.

**Bottom-Right — Data Leakage:**
The Venn diagram shows Training Data (green) overlapping with Test Data (red). The overlapping region has a warning triangle and the message "Overlapping patches or same samples at different resolutions = LEAKAGE!"

*Why this happens:* When extracting patches from large domains, it's easy to accidentally include overlapping regions in both splits. Or when training on 64×64 data and testing on 128×128, you might use the same underlying scenes.

*The fix:* Split by *scene* or *time* before extracting patches. Never let the same physical location appear in both train and test.

> **💡 The Master Lesson:**
> Most FNO failures come from **data issues** (normalization, leakage, wrong metrics) or **configuration errors** (mode count), NOT architecture problems. Before blaming the model, triple-check your data pipeline!

### 7.1 Pitfall: Forgetting Data Normalization

**Symptom:** Training doesn't converge, NaN losses

**Solution:**
```python
# WRONG
prediction = model(raw_input)

# RIGHT
input_normalized = (raw_input - mean) / (std + 1e-8)
prediction_normalized = model(input_normalized)
prediction = prediction_normalized * target_std + target_mean
```

### 7.2 Pitfall: Wrong Tensor Shapes

**Symptom:** Runtime errors, weird results

**Expected shapes:**
- Input: `(batch, Nx, Ny, channels)` or `(batch, channels, Nx, Ny)` depending on convention
- FFT output: `(batch, Nx, Ny//2+1, channels)` for `rfft2`
- FNO output: Same spatial dimensions as input

**Solution:** Add shape assertions throughout your pipeline:
```python
assert input.shape == (batch_size, Nx, Ny, n_channels), f"Got {input.shape}"
```

### 7.3 Pitfall: Overfitting on Small Data

**Symptom:** Training loss near zero, validation loss high

**Solutions:**
1. Reduce model size ($d_v$, $k_{max}$)
2. Add dropout (0.1-0.2)
3. Increase weight decay (1e-3)
4. Add data augmentation
5. Add physics constraints (PINO)

### 7.4 Pitfall: Ignoring Boundary Effects

**FFT assumes periodic boundaries!**

If your domain has non-periodic boundaries:
- Predictions may have artifacts at edges
- Solution: Use larger domain and crop predictions
- Or: Add zero-padding before FFT

### 7.5 Pitfall: Mode Count Mismatch

**If $k_{max} > N//2$:**
- Trying to use more Fourier modes than exist
- Will cause errors or silent bugs

**Solution:** Always ensure $k_{max} \leq \min(N_x, N_y) // 2$

### 7.6 Pitfall: Evaluating Wrong Metric

**Symptom:** Good training loss, poor real-world performance

**Problem:** Optimizing for wrong objective
- MSE might not capture spatial patterns
- Global R² can be inflated by mean prediction

**Solution:** Monitor task-relevant metrics:
- Spatial anomaly correlation for weather/climate
- Peak error for extreme events
- Structure similarity for image-like outputs

### 7.7 Pitfall: Data Leakage

**Symptom:** Test performance suspiciously better than expected

**Causes:**
- Overlapping patches between train/test
- Temporal ordering not respected
- Same sample at different resolutions in train and test

**Solution:** Strict separation by sample ID, temporal ordering.

---

## Section 8: Research Frontiers and Future Directions

### 8.1 Foundation Models for Physics

Large-scale pre-trained models for physical simulation:

- **FourCastNet/Aurora:** Weather and climate
- **Emerging work:** Materials science, fluid dynamics, molecular simulation

The key idea: pre-train on diverse physics simulations, fine-tune for specific applications.

### 8.2 Neural Operator + Numerical Solver Hybrids

Combining learned operators with traditional methods:
- Use FNO for fast coarse approximation
- Refine with traditional PDE solver
- Best of both: speed + guaranteed accuracy

### 8.3 Uncertainty Quantification

FNO gives point predictions. Research directions include:
- **Ensemble FNO:** Train multiple models, use spread as uncertainty
- **Bayesian FNO:** Uncertainty in weights propagates to predictions
- **MC Dropout:** Use dropout at inference for uncertainty estimates

### 8.4 Multi-Fidelity Learning

Combine data at different resolutions/qualities:
- High-fidelity: Few expensive simulations
- Low-fidelity: Many cheap simulations

FNO's resolution invariance enables this naturally—train on mixed-resolution data.

### 8.5 Geometry-Aware Operators

Extending beyond regular grids:
- **Geo-FNO:** Learned coordinate deformations for general geometries
- **GINO:** Combines graph neural operators with FNO for arbitrary meshes

---

## Summary: Key Takeaways

### Training Essentials
- ✓ Normalize inputs per-channel and outputs
- ✓ Use appropriate train/val/test split
- ✓ AdamW optimizer, LR=1e-3, cosine annealing
- ✓ Early stopping with patience=50
- ✓ Monitor task-relevant metrics (not just loss)

### Physics-Informed FNO
- ✓ Add PDE residuals, conservation, smoothness constraints
- ✓ λ_physics typically 0.01-0.1, increase gradually
- ✓ Particularly valuable with limited data
- ✓ Encode domain knowledge as soft constraints

### Advanced Architectures
- ✓ **U-FNO** for multi-scale + skip connections
- ✓ **Factorized FNO** for parameter efficiency
- ✓ **AFNO** for adaptive frequency attention
- ✓ **Multi-scale FNO** for hierarchical patterns

### Interpretation
- ✓ Examine spectral weights for frequency importance
- ✓ Permutation importance for channel relevance
- ✓ Gradient saliency for spatial attention
- ✓ Verify learned patterns match physics expectations

### Common Pitfalls
- ✓ Always normalize data
- ✓ Watch for overfitting with limited data
- ✓ Ensure $k_{max} \leq N//2$
- ✓ Use correct evaluation metrics
- ✓ Prevent data leakage

---

## Figure Summary: Visual Guide to Advanced FNO Topics

Here's a quick reference to all figures in this tutorial:

- **Fig. 1 - PINO Concept**: Combine data loss + physics loss. Physics constraints improve generalization with limited data.
- **Fig. 2 - Advanced Architectures**: U-FNO for multi-scale, F-FNO for efficiency, AFNO for attention. Choose based on your problem.
- **Fig. 3 - Weather Foundation Models**: AI weather models achieve 1000-10,000× speedup. Active research area with multiple competing approaches.
- **Fig. 4 - Operator Comparison**: FNO for grids, DeepONet for point queries, GNO for irregular meshes. Match method to data structure.
- **Fig. 5 - Interpretation Methods**: Spectral weights, frequency importance, channel importance, saliency maps. Verify physics is learned.
- **Fig. 6 - Common Pitfalls**: Normalization, overfitting, mode count, boundaries, metrics, leakage. Most failures are data issues.

---

## Wrapping Up: Your Journey So Far

Congratulations! You've now covered the advanced topics that separate FNO beginners from practitioners. Let's recap what you learned:

**Physics-Informed Learning:** You can now combine data-driven learning with physical constraints. This is especially powerful when data is scarce but physics is well understood—the constraints guide your model toward physically plausible solutions.

**Advanced Architectures:** You know when to reach for U-FNO (sharp features), Factorized FNO (memory constraints), or AFNO (very high resolution). You understand the tradeoffs and can make informed decisions.

**Weather Foundation Models:** You've seen how these same techniques power production AI weather forecasting systems that are revolutionizing meteorology. The gap between "research technique" and "deployed system" is surprisingly small.

**Model Interpretation:** You can look inside your trained FNO, understand which frequencies and features matter, and validate that the model learned physics rather than spurious correlations.

**Common Pitfalls:** Most importantly, you know the six deadly sins of FNO training. This knowledge alone will save you countless hours of debugging.

---

## References

### Core Papers

1. Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations.* ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

2. Li, Z., Zheng, H., Kovachki, N., et al. (2021). *Physics-Informed Neural Operator for Learning Partial Differential Equations.* [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)

3. Guibas, J., Mardani, M., Li, Z., et al. (2022). *Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers.* ICLR 2022. [arXiv:2111.13587](https://arxiv.org/abs/2111.13587)

4. Tran, A., Mathews, A., Xie, L., Ong, C.S. (2023). *Factorized Fourier Neural Operators.* ICLR 2023. [arXiv:2111.13802](https://arxiv.org/abs/2111.13802)

5. Wen, G., Li, Z., Azizzadenesheli, K., et al. (2022). *U-FNO—An enhanced Fourier neural operator for multiphase flow.* Advances in Water Resources. [arXiv:2109.03697](https://arxiv.org/abs/2109.03697)

### Weather/Climate Foundation Models

6. Pathak, J., Subramanian, S., Harrington, P., et al. (2022). *FourCastNet: A Global Data-driven High-resolution Weather Model.* [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)

7. Bi, K., Xie, L., Zhang, H., et al. (2023). *Accurate medium-range global weather forecasting with 3D neural networks.* Nature 619, 533–538.

8. Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). *Learning skillful medium-range global weather forecasting.* Science 382, 1416–1421.

### Survey Papers

9. Kovachki, N., Li, Z., Liu, B., et al. (2024). *Neural operators for accelerating scientific simulations and design.* Nature Reviews Physics. [arXiv:2309.15325](https://arxiv.org/abs/2309.15325)

10. Kovachki, N., Li, Z., Liu, B., et al. (2023). *Neural Operator: Learning Maps Between Function Spaces.* JMLR 24(89):1-97. [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)

---

## Continue Your Learning

- **[← Part 3: Complete Architecture](../chunk3/chunk3_final.md)**: Full FNO implementation from lifting to projection
- **Part 4: Advanced Topics** *(You are here)*: Physics-informed learning, variants, interpretation
- **[Part 5 →: Practical Implementation](../chunk5/chunk5_final.md)**: Complete training code, real datasets, production tips

**What's Next?**

Part 5 brings everything together with complete, runnable code. You'll implement a full training pipeline, work with real data, and learn the practical tips that make the difference between a paper implementation and a production system. See you there!

---

*This tutorial is part of a comprehensive series on Fourier Neural Operators for scientific machine learning. Whether you're predicting weather, simulating physics, or solving PDEs, FNO provides a powerful, efficient approach that's revolutionizing computational science.*
