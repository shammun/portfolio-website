# Fourier Neural Operator: Chunk 4
## Advanced Topics: Training, Physics-Informed Learning, and Application to Urban Temperature

---

# Introduction

You have mastered:
- **Chunk 1:** Fourier transforms, convolution theorem, spectral methods
- **Chunk 2:** Spectral convolution, mode truncation, the Fourier layer
- **Chunk 3:** Complete FNO architecture (lifting → Fourier layers → projection)

Now we go deeper into **practical mastery**:

1. Training FNO effectively (data preparation, optimization, debugging)
2. Physics-Informed Neural Operators (PINO) — adding physical constraints
3. Advanced architectures (U-FNO, Factorized FNO, Adaptive FNO)
4. Applying FNO to YOUR urban temperature problem
5. Interpretation: What does FNO learn?
6. Comparison with other neural operator methods
7. Common pitfalls and how to avoid them

**This chunk transforms you from "I understand FNO" to "I can successfully apply FNO to real problems."**

---

# Section 1: Training FNO Effectively

## 1.1 Data Preparation — The Foundation of Success

### Why Data Preparation Matters More for Neural Operators

Unlike image classification (where networks are robust to variations), neural operators are sensitive to:
- **Scale mismatches** between input features
- **Coordinate inconsistencies** in spatial data
- **Missing data patterns** that differ between train/test
- **Domain boundaries** that affect Fourier transforms

### Step-by-Step Data Pipeline

**Step 1: Spatial Alignment**

All input features must be on the **same grid**:
```
Feature 1: 70m resolution, WGS84, covering NYC
Feature 2: Must match EXACTLY — same resolution, projection, extent
...
Feature 42: Same grid as Feature 1
```

**Your ECOSTRESS data:** You've already aligned to a 70m master grid. This is correct.

**Step 2: Temporal Alignment**

For multi-temporal data, ensure consistency:
- Same time of day (or encode time as a feature)
- Account for seasonal variations
- Handle missing observations (cloud cover)

**Step 3: Feature Normalization**

**Critical:** Normalize EACH feature channel independently.

```
For channel c in 1...42:
    μ_c = mean of channel c over all training samples
    σ_c = std of channel c over all training samples
    channel_c_normalized = (channel_c - μ_c) / σ_c
```

**Why per-channel?**
- NDVI ranges [-1, 1]
- Building height ranges [0, 500] meters
- ERA5 temperature ranges [250, 320] K
- Without normalization, ERA5 temperature dominates due to large values

**Step 4: Output Normalization**

Also normalize the target (temperature):
```
T_normalized = (T - μ_T) / σ_T
```

Store μ_T and σ_T for denormalization at inference.

**Step 5: Train/Validation/Test Split**

For spatial data, consider:

**Random scenes:**
- Pros: Simple, more data
- Cons: May overfit to specific locations

**Temporal split:**
- Pros: Tests generalization to new times
- Cons: May miss seasonal patterns

**Spatial split:**
- Pros: Tests generalization to new areas
- Cons: Harder to implement

**Recommended for your problem:**
- 70% training (~160 scenes)
- 15% validation (~35 scenes) — for hyperparameter tuning
- 15% test (~35 scenes) — held out until final evaluation

## 1.2 Batch Construction for FNO

### Full-Field vs. Patch-Based Training

**Option 1: Full Field Training**
- Input: Entire spatial domain (e.g., 256×256 or 512×512)
- Batch: Multiple time steps (scenes)
- Pros: Captures full global context
- Cons: Memory intensive, limited batch size

**Option 2: Patch-Based Training**
- Extract patches (e.g., 64×64) from larger domain
- More samples per epoch
- Pros: Larger batches, data augmentation
- Cons: Loses some long-range context at patch boundaries

**For your urban temperature:**
- If memory allows: Full field at reduced resolution (128×128 or 64×64)
- If memory limited: 64×64 patches with overlap

### Handling Variable Grid Sizes

FNO's resolution invariance means you can:
1. Train on downsampled 64×64 grids (faster)
2. Evaluate on full 256×256 or larger (higher resolution)

**Training strategy:**
```
Epoch 1-100: Train on 64×64 (fast iteration)
Epoch 101-200: Fine-tune on 128×128 (refine details)
Final: Evaluate on full resolution
```

## 1.3 Optimization Settings

### Learning Rate

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

### Weight Decay (L2 Regularization)

**Recommended:** 1e-4

Helps prevent overfitting, especially with limited data (your 230 scenes).

### Gradient Clipping

**Recommended:** Clip gradients to max norm 1.0

Prevents exploding gradients during early training when FFT operations can amplify gradients.

### Optimizer

**Adam** is the standard choice:
```
optimizer = Adam(lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
```

**AdamW** (decoupled weight decay) is slightly better:
```
optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
```

## 1.4 Regularization Techniques

### Dropout in FNO

Apply dropout to the **local path** (W), not the spectral path:

```
v_out = GELU(spectral_conv(v) + dropout(W @ v) + b)
```

**Why not spectral?** Dropping Fourier coefficients randomly disrupts the smoothness prior.

**Recommended dropout rate:** 0.1-0.2

### Spectral Regularization

Add penalty on spectral weight magnitudes:
$$\mathcal{L}_{spectral} = \lambda \sum_k |R(k)|^2$$

This encourages the network to use fewer/smaller spectral weights.

### Data Augmentation for Spatial Data

**Valid augmentations:**
- Random horizontal/vertical flips
- 90°, 180°, 270° rotations
- Small random crops + resize

**Invalid augmentations:**
- Arbitrary rotations (breaks grid alignment)
- Large translations (shifts boundary conditions)
- Color jittering (not applicable to physical fields)

## 1.5 Early Stopping and Checkpointing

### Early Stopping Protocol

```
best_val_loss = infinity
patience_counter = 0
patience = 50  # epochs

for epoch in training:
    val_loss = evaluate(validation_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Load best model for final evaluation
model.load_state_dict(best_checkpoint)
```

### What to Monitor

1. **Training loss** — should decrease steadily
2. **Validation loss** — should decrease then plateau
3. **Validation metrics** (R², RMSE) — more interpretable than loss
4. **Spatial anomaly R²** — YOUR key metric

---

# Section 2: Physics-Informed Neural Operators (PINO)

## 2.1 The Idea: Combining Data and Physics

Standard FNO: Learn from data alone
$$\mathcal{L} = \mathcal{L}_{data} = \frac{1}{N}\sum_i \|\mathcal{G}_\theta(a^{(i)}) - u^{(i)}\|^2$$

**Physics-Informed FNO:** Add physical constraints
$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{physics} \mathcal{L}_{physics}$$

## 2.2 Types of Physics Constraints

### Type 1: PDE Residual Loss

If you know the governing PDE, penalize violations:

**Example: Heat equation**
$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T + Q$$

**PDE residual:**
$$\mathcal{L}_{PDE} = \left\| \frac{\partial \hat{T}}{\partial t} - \alpha \nabla^2 \hat{T} - Q \right\|^2$$

Computed using automatic differentiation.

### Type 2: Conservation Laws

**Energy conservation:**
$$\mathcal{L}_{conservation} = \left| \int_\Omega \hat{T} \, dA - \int_\Omega T_{true} \, dA \right|^2$$

**For urban temperature:** Total heat content should be conserved (approximately).

### Type 3: Boundary Conditions

Enforce known behavior at boundaries:
$$\mathcal{L}_{BC} = \| \hat{T}|_{\partial\Omega} - T_{BC} \|^2$$

**For your problem:** Water bodies maintain relatively stable temperature.

### Type 4: Smoothness Prior

Temperature fields should be smooth (no unphysical discontinuities):
$$\mathcal{L}_{smooth} = \| \nabla \hat{T} \|^2$$

This is somewhat redundant with FNO's spectral bias but can help.

## 2.3 Physics Constraints for Urban Temperature

**Relevant physical knowledge for your problem:**

1. **Thermal diffusion:** Temperature varies smoothly in space
   - Already captured by FNO's low-frequency bias
   - Can add explicit smoothness penalty

2. **Energy balance:** 
   $$R_{net} = H + LE + G$$
   Where: R_net = net radiation, H = sensible heat, LE = latent heat, G = ground heat flux
   - Hard to implement without full energy balance data

3. **Known relationships:**
   - Vegetation cools (higher NDVI → lower temperature)
   - Urban surfaces warm (higher NDBI → higher temperature)
   - Water bodies stabilize temperature

4. **Spatial constraints:**
   - Parks create cooling "halos" extending beyond their boundaries
   - Urban heat island intensity peaks in city center

### Soft Physics: Encoding Known Relationships

Instead of hard constraints, encode as regularization:

**Vegetation cooling prior:**
$$\mathcal{L}_{veg} = \text{ReLU}\left(\text{corr}(\hat{T}, \text{NDVI})\right)^2$$

This penalizes positive correlation (warm vegetation), allowing negative (cool vegetation).

**Urban heating prior:**
$$\mathcal{L}_{urban} = \text{ReLU}\left(-\text{corr}(\hat{T}, \text{NDBI})\right)^2$$

Penalizes negative correlation (cool urban), allowing positive (warm urban).

## 2.4 Implementing PINO for FNO

### Gradient Computation

To compute ∇²T, use finite differences or automatic differentiation:

**Finite differences (simple):**
$$\nabla^2 T \approx \frac{T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j}}{\Delta x^2}$$

**Spectral (elegant):**
$$\nabla^2 T = \mathcal{F}^{-1}[-(k_x^2 + k_y^2) \cdot \mathcal{F}[T]]$$

The Laplacian in Fourier space is just multiplication by $-|k|^2$.

### Training with Physics Loss

```
for batch in data:
    pred = model(batch.input)
    
    # Data loss
    loss_data = MSE(pred, batch.target)
    
    # Physics loss (example: smoothness)
    laplacian = compute_laplacian_spectral(pred)
    loss_physics = torch.mean(laplacian**2)
    
    # Combined loss
    loss = loss_data + lambda_physics * loss_physics
    
    loss.backward()
    optimizer.step()
```

### Choosing λ_physics

**Too small:** Physics has no effect
**Too large:** Model focuses on physics, ignores data

**Strategy:** Start with λ=0.01, increase gradually:
- Epochs 1-50: λ = 0.01
- Epochs 51-100: λ = 0.1
- Epochs 101+: λ = 1.0

Or use automatic balancing based on loss magnitudes.

## 2.5 Benefits of Physics-Informed Training

- **Better generalization**: Physics constraints guide extrapolation
- **Data efficiency**: Learn more from fewer samples
- **Physical consistency**: Predictions obey known laws
- **Reduced overfitting**: Physics acts as regularization
- **Interpretability**: Can verify predictions satisfy physics

**For your limited data (230 scenes):** Physics constraints could significantly help!

---

# Section 3: Advanced FNO Architectures

## 3.1 U-FNO: U-Net + FNO

### Motivation

Standard FNO uses mode truncation, losing high-frequency information. U-Net's skip connections preserve details.

### Architecture

```
Encoder (FNO path):
    Input → FL1 → Downsample → FL2 → Downsample → FL3 (bottleneck)

Decoder (with skip connections):
    FL3 → Upsample → Concat(FL2) → FL4 → Upsample → Concat(FL1) → FL5 → Output
```

### Benefits
- Better high-frequency reconstruction
- Multi-scale feature learning
- Particularly good for problems with sharp gradients

### When to Use
- When output has both smooth trends AND sharp features
- For image-like outputs requiring fine detail
- Urban temperature: Could help capture building-scale variations

## 3.2 Factorized FNO (F-FNO)

### Motivation

Standard FNO spectral weights: $R \in \mathbb{C}^{k_{max}^2 \times d_v^2}$

This scales poorly with k_max and d_v.

### Factorization

Instead of full tensor R, use factorized form:
$$R(k_x, k_y) = R_x(k_x) \cdot R_y(k_y)$$

Or with mixing:
$$R(k_x, k_y) = U \cdot \text{diag}(R_x(k_x)) \cdot V \cdot \text{diag}(R_y(k_y)) \cdot W$$

### Parameter Reduction

**k_max=12, d_v=64:**
- Standard: 2.4M parameters
- Factorized: 0.2M parameters
- Reduction: 12×

**k_max=20, d_v=128:**
- Standard: 26M parameters
- Factorized: 1.3M parameters
- Reduction: 20×

### Trade-off
- Fewer parameters = less expressivity
- But often works just as well in practice
- Better for limited data situations (like yours)

## 3.3 Adaptive Fourier Neural Operator (AFNO)

### Motivation

FNO uses fixed mode truncation. What if different spatial regions need different frequencies?

### Approach

Learn attention weights for Fourier modes:
$$\hat{w}(k) = \text{softmax}(\alpha(k)) \cdot R(k) \cdot \hat{v}(k)$$

Where α(k) is learned attention over frequencies.

### Benefits
- Adapts to problem structure
- Can focus on relevant frequencies per region
- State-of-the-art for weather prediction (Pathways/AFNO models)

## 3.4 Multi-Scale FNO (MS-FNO)

### Motivation

Urban temperature has patterns at multiple scales:
- City-wide (10+ km): Regional gradients
- Neighborhood (1-5 km): Parks, districts
- Block (100m-1km): Buildings, streets

### Architecture

```
Branch 1 (coarse): k_max = 4, captures city-wide patterns
Branch 2 (medium): k_max = 12, captures neighborhood patterns  
Branch 3 (fine): k_max = 24, captures block-scale patterns

Output = Combine(Branch1, Branch2, Branch3)
```

### Implementation
- Run parallel FNO branches with different k_max
- Concatenate features before projection
- Or: Use hierarchical processing (coarse → fine)

## 3.5 Temporal FNO (for Time Series)

### For Time-Dependent Problems

If predicting temperature evolution over time:

**Option 1: FNO-3D**
- Treat time as third dimension
- Input: (Nx, Ny, Nt, d_a)
- 3D Fourier transform
- Computational expensive

**Option 2: Autoregressive FNO**
- Train: predict T(t+1) from T(t) and features
- Inference: Roll out step by step
- Simpler, but error accumulation

**Option 3: Direct Multi-Step**
- Separate models for each lead time
- T(t+1), T(t+2), ..., T(t+n)
- No error accumulation but more models

### For Your Problem

Since you predict single-time temperature (not evolution), standard FNO-2D is appropriate.

But if you want to predict "temperature at 3pm given 9am observations," consider temporal encoding.

---

# Section 4: Applying FNO to Your Urban Temperature Problem

## 4.1 Problem Formulation

**Input:** $a(x, y) \in \mathbb{R}^{42}$ — Your 42 features at each pixel

**Output:** $T(x, y) \in \mathbb{R}$ — Land surface temperature

**Operator:** $\mathcal{G}: a \mapsto T$

**Training data:** ~230 ECOSTRESS scenes (after quality filtering)

## 4.2 Data Preparation Pipeline

### Step 1: Create Master Grid

You have this: 70m resolution grid covering NYC

### Step 2: Prepare Feature Tensors

For each scene:
```
Input tensor: shape (Nx, Ny, 42)
    Channel 0: NDVI
    Channel 1: NDBI
    Channel 2: NDWI
    Channel 3-5: Building height, density, SVF
    Channel 6-10: ERA5 (T, RH, U, V, radiation)
    Channel 11-20: Land cover one-hot
    Channel 21-30: Distance features
    Channel 31-41: Interaction terms
    Channel 42: Maybe solar geometry (zenith, azimuth)

Target tensor: shape (Nx, Ny, 1)
    Channel 0: ECOSTRESS LST
```

### Step 3: Handle Missing Data

**Option 1: Mask and exclude**
- Create valid pixel mask
- Only compute loss on valid pixels
- Requires careful implementation

**Option 2: Imputation**
- Fill missing with spatial interpolation
- Or fill with climatological mean
- Simpler but introduces artifacts

**Option 3: Separate by quality**
- Only use scenes with >90% valid coverage
- May reduce dataset size

**Recommended:** Option 3 initially (simpler), Option 1 if you need more data.

### Step 4: Temporal Stratification

Your 230 scenes span different:
- Times of day (ECOSTRESS varies)
- Seasons
- Weather conditions

**Encode temporal information:**
- Add solar zenith angle as feature
- Add day-of-year as feature (cyclical encoding)
- Or: Train separate models per season

### Step 5: Resolution Strategy

**Training resolution:**
- If 230 scenes at 256×256: batch_size ≈ 4-8 (memory limited)
- If 230 scenes at 64×64: batch_size ≈ 16-32 (comfortable)

**Recommendation:** 
1. Downsample to 64×64 for development
2. Use 128×128 for final training
3. Evaluate at full resolution (resolution invariance!)

## 4.3 Model Configuration

### Conservative Configuration (Start Here)

```python
config = {
    'd_a': 42,        # Your features
    'd_v': 32,        # Hidden dimension
    'd_u': 1,         # Temperature
    'k_max': 12,      # Fourier modes
    'n_layers': 4,    # Depth
    'd_mid': 64,      # Projection
}
# ~1.2M parameters
```

### Why Conservative?

- 230 samples is limited for neural networks
- Better to underfit slightly than overfit severely
- Can always increase capacity if needed

### Scaling Up (If Underfitting)

```python
config_larger = {
    'd_a': 42,
    'd_v': 64,        # Increase hidden
    'd_u': 1,
    'k_max': 16,      # More modes
    'n_layers': 4,
    'd_mid': 128,
}
# ~4.7M parameters
```

Only use if validation loss is high AND train loss is similar to val loss.

## 4.4 Training Protocol

### Phase 1: Development (Fast Iteration)

```
Data: 64×64 downsampled
Batch size: 16
Epochs: 100
LR: 1e-3 → 1e-5 (cosine)
Goal: Verify pipeline works, tune hyperparameters
```

### Phase 2: Full Training

```
Data: 128×128 or full resolution
Batch size: 8
Epochs: 300
LR: 1e-3 → 1e-6 (cosine)
Early stopping: patience=50
```

### Phase 3: Evaluation

```
Load best checkpoint
Evaluate on test set at full resolution
Compute: RMSE, MAE, R², Spatial Anomaly R²
Visualize: Predictions vs. ground truth
```

## 4.5 Expected Results

### Baseline Comparison

**Your Random Forest:**
- Pooled R²: 0.98-0.99
- Spatial Anomaly R²: 0.48-0.75

**FNO (expected):**
- Pooled R²: 0.90-0.98
- Spatial Anomaly R²: 0.60-0.85

**Why FNO might have lower pooled R²:**
- RF with 31M observations captures pixel-level patterns well
- FNO with 230 scenes has less data

**Why FNO should have higher spatial anomaly R²:**
- FNO captures spatial dependencies (RF doesn't)
- Spatial anomaly is what FNO is designed for

### What Success Looks Like

1. **Training curve:** Loss decreases smoothly, train/val gap small
2. **Visual inspection:** Predictions capture spatial patterns
3. **Spatial anomaly R² > 0.70:** Better than RF on key metric
4. **Resolution invariance:** Same quality at 64×64 and 256×256

## 4.6 Debugging Checklist

If training fails:

**Symptom: Loss is NaN**
- Check: Data normalization (features with zero variance?)
- Fix: Add epsilon to std, clip extreme values

**Symptom: Loss doesn't decrease**
- Check: Learning rate (too low?)
- Check: Data loading (same batch every time?)
- Fix: Increase LR, verify data shuffling

**Symptom: Train loss good, val loss bad (overfitting)**
- Check: Model size vs. data size
- Fix: Reduce d_v, k_max, add dropout, add weight decay

**Symptom: Both train and val loss high (underfitting)**
- Check: Model capacity, data quality
- Fix: Increase d_v, k_max, more epochs, check data bugs

---

# Section 5: Interpretation — What Does FNO Learn?

## 5.1 Examining Spectral Weights

The spectral weights R(k) tell us which frequencies matter:

**Visualization:**
```
For each Fourier layer:
    Plot |R(k_x, k_y)| averaged over channels
    High values = important frequencies
```

**Expected patterns for temperature:**
- High weights at low k (large-scale patterns)
- Lower weights at high k (fine details)
- Possibly peaks at specific frequencies (periodic urban structure?)

## 5.2 Frequency Importance Analysis

**Method:**
1. Zero out modes at different k values
2. Measure prediction degradation

```
For k in [1, 2, 4, 8, 12, 16]:
    model_modified = copy(model)
    model_modified.R[:, k:, k:, :, :] = 0  # Zero high frequencies
    error_k = evaluate(model_modified)
```

**Interpretation:**
- Large error increase when zeroing k=1-4: City-wide patterns matter
- Large error increase when zeroing k=8-12: Neighborhood patterns matter

## 5.3 Channel Importance

**Method:** Permutation importance
1. Shuffle one input channel
2. Measure prediction degradation
3. High degradation = important channel

**Expected for your features:**
- High importance: ERA5 temperature, NDVI, building height
- Medium importance: NDBI, SVF, distance features
- Variable: Land cover (depends on area)

## 5.4 Spatial Attention Maps

**Method:** Gradient-based saliency
1. Compute gradient of output w.r.t. input at a point
2. Visualize which input locations affect the prediction

```
For target point (x0, y0):
    output = model(input)
    loss = output[x0, y0]
    loss.backward()
    saliency = |input.grad|
```

**Expected for FNO:**
- High saliency across entire domain (global receptive field)
- Peaks at nearby urban features (parks, highways)
- This demonstrates FNO captures non-local dependencies

## 5.5 Comparing with Physics Expectations

**Test: Does FNO learn vegetation cooling?**
```
For a test scene:
    Artificially increase NDVI in a region
    Predict temperature
    Verify temperature decreases
```

**Test: Does FNO learn urban heating?**
```
Artificially increase building density
Verify temperature increases
```

**Test: Does FNO learn park cooling extent?**
```
Analyze temperature prediction around parks
Does cooling extend beyond park boundaries?
```

---

# Section 6: Comparison with Other Methods

## 6.1 FNO vs. DeepONet

**DeepONet** (Deep Operator Network):
- Uses branch network (encodes input function) and trunk network (encodes output location)
- Output: $u(x) = \sum_k b_k(a) \cdot t_k(x)$

**Architecture:**
- FNO: FFT-based
- DeepONet: Basis function expansion

**Resolution invariance:**
- FNO: Yes
- DeepONet: Yes

**Global receptive field:**
- FNO: Yes (FFT)
- DeepONet: Depends on branch design

**Parameter efficiency:**
- FNO: Good
- DeepONet: Can be higher

**Implementation:**
- FNO: Moderate
- DeepONet: Simpler

**When to prefer DeepONet:**
- Irregular domains (FNO needs regular grids)
- Point-wise evaluation needed

## 6.2 FNO vs. Graph Neural Operators

**Graph Neural Operators:**
- Work on unstructured meshes
- Message passing between nodes

**Domain:**
- FNO: Regular grids
- GNO: Any mesh

**Complexity:**
- FNO: O(N log N)
- GNO: O(N × neighbors)

**Resolution:**
- FNO: Must be grid
- GNO: Adaptive mesh

**Implementation:**
- FNO: FFT-based
- GNO: Graph libraries

**When to prefer GNO:**
- Complex domain geometry
- Adaptive refinement needed
- Unstructured data

**For your problem:** FNO is appropriate (regular grid from satellite).

## 6.3 FNO vs. Transformers (Attention-Based)

**Vision Transformers for PDEs:**
- Self-attention over spatial patches
- Can model long-range dependencies

**Global context:**
- FNO: Yes (FFT)
- Transformer: Yes (attention)

**Complexity:**
- FNO: O(N log N)
- Transformer: O(N²) or O(N log N)

**Inductive bias:**
- FNO: Smoothness (spectral)
- Transformer: None (data-driven)

**Data efficiency:**
- FNO: Better
- Transformer: Needs more data

**When to prefer Transformers:**
- Very large datasets
- Complex, multi-modal inputs
- Pre-trained models available

**For your problem:** FNO is better (limited data, physics-appropriate bias).

## 6.4 FNO vs. CNNs

**Receptive field:**
- FNO: Global (immediately)
- CNN: Local (grows with depth)

**Resolution:**
- FNO: Invariant
- CNN: Fixed

**Parameters for global:**
- FNO: O(k_max² d²)
- CNN: O(N² d²) for global kernel

**Inductive bias:**
- FNO: Spectral/smooth
- CNN: Translation equivariance

**For your problem:** 
- CNN would work but needs many layers for global context
- FNO is more natural for PDE-like problems

## 6.5 FNO vs. Your Random Forest

**Spatial context:**
- Random Forest: None
- FNO: Global

**Training data:**
- Random Forest: Point observations
- FNO: Field observations

**Output:**
- Random Forest: Point predictions
- FNO: Full field

**Interpretability:**
- Random Forest: High (feature importance)
- FNO: Moderate

**Generalization:**
- Random Forest: To similar points
- FNO: To similar fields

**Key insight:** They solve different problems!
- RF: "Given features at point x, what's temperature?"
- FNO: "Given feature field, what's temperature field?"

---

# Section 7: Common Pitfalls and Solutions

## 7.1 Pitfall: Forgetting Data Normalization

**Symptom:** Training doesn't converge, NaN losses

**Solution:** 
```python
# WRONG
model(raw_input)

# RIGHT
input_normalized = (raw_input - mean) / std
model(input_normalized)
```

## 7.2 Pitfall: Wrong Tensor Shapes

**Symptom:** Runtime errors, weird results

**Expected shapes:**
- Input: (batch, Nx, Ny, channels) or (Nx, Ny, channels)
- FFT output: (batch, Nx, Ny//2+1, channels) for rfft2
- FNO output: same as input spatial dims

**Solution:** Add shape assertions throughout pipeline.

## 7.3 Pitfall: Overfitting on Small Data

**Symptom:** Train loss near zero, val loss high

**Solutions:**
1. Reduce model size (d_v, k_max)
2. Add dropout (0.1-0.2)
3. Increase weight decay (1e-3)
4. Add data augmentation
5. Add physics constraints (PINO)

## 7.4 Pitfall: Ignoring Boundary Effects

**FFT assumes periodic boundaries!**

If your domain has strong boundary effects:
- Predictions may have artifacts at edges
- Solution: Use larger domain and crop predictions
- Or: Add zero-padding before FFT

## 7.5 Pitfall: Mode Count Mismatch

**If k_max > N//2:**
- Trying to use more modes than exist
- Will cause errors or silent bugs

**Solution:** Always ensure k_max ≤ min(Nx, Ny) // 2

## 7.6 Pitfall: Evaluating Wrong Metric

**Symptom:** Good training loss, poor real-world performance

**Problem:** Optimizing for wrong objective
- MSE might not capture spatial patterns
- Raw R² inflated by temporal signal

**Solution:** Monitor spatial anomaly R² during training!

## 7.7 Pitfall: Data Leakage

**Symptom:** Test performance much better than expected

**Causes:**
- Overlapping patches between train/test
- Temporal ordering not respected
- Same scene in train and test (at different resolutions)

**Solution:** Strict train/test separation by scene ID.

---

# Section 8: Research Frontiers and Future Directions

## 8.1 Foundation Models for PDEs

Like GPT for language, researchers are building foundation models for physics:
- **FourCastNet:** Global weather prediction with AFNO
- **Pangu-Weather:** Transformer-based weather model
- **ClimaX:** Multi-task climate model

**Relevance to your work:** These models could be fine-tuned for urban temperature.

## 8.2 Neural Operator + Physics Hybrid

Combining learned operators with numerical solvers:
- Use FNO for fast approximation
- Refine with traditional PDE solver
- Best of both worlds

## 8.3 Uncertainty Quantification

FNO gives point predictions. Research on:
- Ensemble FNO (multiple models)
- Bayesian FNO (uncertainty in weights)
- Dropout as uncertainty (MC Dropout)

**For your thesis:** Could add uncertainty estimates to predictions.

## 8.4 Multi-Fidelity Learning

Combine data at different resolutions/qualities:
- High-fidelity: Few ECOSTRESS observations
- Low-fidelity: Many Landsat observations

FNO's resolution invariance enables this naturally.

---

# Section 9: Strategic Value for PhD Applications

## 9.1 For Professor Hassanzadeh (UChicago)

**Connection:** FourCastNet and climate extremes

**What to emphasize:**
- Understanding of spectral methods in neural operators
- Ability to implement FNO from scratch
- Connection to climate modeling at global scale
- Experience with satellite data processing

**Potential research direction:** 
"Applying FNO for urban heat extreme prediction under climate change scenarios"

## 9.2 For Professor Shreevastava (NYU)

**Connection:** Urban climate modeling

**What to emphasize:**
- Direct application to urban heat island
- Physics-informed approach with known UHI dynamics
- Multi-scale nature of FNO matching urban complexity
- Resolution invariance for different city scales

**Potential research direction:**
"Scale-aware neural operators for urban thermal environment prediction"

## 9.3 For Professor Villarini (Princeton)

**Connection:** Computational hydrology

**What to emphasize:**
- Operator learning generalizes to fluid dynamics
- Understanding of PDEs underlying physical processes
- Mathematical rigor in spectral methods
- Potential application to flood prediction

**Potential research direction:**
"Neural operators for rapid flood inundation mapping"

---

# Summary: Chunk 4 Key Takeaways

## Training Essentials
- [ ] Normalize inputs per-channel and outputs
- [ ] Use appropriate train/val/test split
- [ ] Adam optimizer, LR=1e-3, cosine annealing
- [ ] Early stopping with patience=50
- [ ] Monitor spatial anomaly R² (not just loss)

## Physics-Informed FNO
- [ ] Can add PDE residuals, conservation, smoothness
- [ ] λ_physics typically 0.01-0.1
- [ ] Particularly valuable with limited data
- [ ] Encode domain knowledge as soft constraints

## Advanced Architectures
- [ ] U-FNO for multi-scale + skip connections
- [ ] Factorized FNO for parameter efficiency
- [ ] AFNO for adaptive frequency attention
- [ ] Multi-scale FNO for hierarchical patterns

## Your Urban Temperature Application
- [ ] Start conservative: d_v=32, k_max=12, 4 layers
- [ ] Encode temporal info (solar zenith, day-of-year)
- [ ] Train at 64×64, evaluate at full resolution
- [ ] Target: Spatial anomaly R² > 0.70

## Interpretation
- [ ] Examine spectral weights for frequency importance
- [ ] Permutation importance for channel relevance
- [ ] Gradient saliency for spatial attention
- [ ] Verify learned patterns match physics

## Common Pitfalls
- [ ] Always normalize data
- [ ] Watch for overfitting with limited data
- [ ] Ensure k_max ≤ N//2
- [ ] Use correct evaluation metrics
- [ ] Prevent data leakage

---

# Next: Code Implementation

Chunk 4 code will implement:
1. Complete training pipeline with logging
2. Physics-informed loss functions
3. Advanced regularization techniques
4. Model interpretation tools
5. Evaluation suite with all metrics
6. Template for your ECOSTRESS data

Let me know when you're ready!
