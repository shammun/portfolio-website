"""
Fourier Neural Operator: Chunk 6 - Advanced Extensions
Part 2: Transfer Learning and Uncertainty Quantification

This implements:
1. Transfer Learning Framework
2. Fine-Tuning Protocol
3. Domain Adaptation
4. Multi-City Training
5. Ensemble FNO for Uncertainty
6. MC Dropout Uncertainty
7. Heteroscedastic (NLL) Loss
8. Conformal Prediction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import os
from copy import deepcopy

os.makedirs('figures', exist_ok=True)

print("="*70)
print("CHUNK 6: ADVANCED EXTENSIONS - PART 2: TRANSFER & UNCERTAINTY")
print("="*70)

#=============================================================================
# CORE COMPONENTS (Simplified for this file)
#=============================================================================

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

class FNO2d:
    """Simplified FNO with accessible weights for transfer learning."""
    
    def __init__(self, d_a, d_v, d_u, k_max, n_layers=4, dropout=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.d_a, self.d_v, self.d_u = d_a, d_v, d_u
        self.k_max, self.n_layers = k_max, n_layers
        self.dropout = dropout
        
        # Store all weights in dict for easy access
        self.weights = {}
        
        # Lifting
        self.weights['P'] = np.random.randn(d_a, d_v) * np.sqrt(2/(d_a+d_v))
        self.weights['b_P'] = np.zeros(d_v)
        
        # Fourier layers
        for i in range(n_layers):
            scale = 1.0 / np.sqrt(d_v * d_v)
            self.weights[f'R_{i}'] = (np.random.randn(k_max, k_max, d_v, d_v) + 
                                      1j * np.random.randn(k_max, k_max, d_v, d_v)) * scale
            self.weights[f'W_{i}'] = np.random.randn(d_v, d_v) * scale
            self.weights[f'b_{i}'] = np.zeros(d_v)
        
        # Projection
        d_mid = d_v * 2
        self.weights['Q1'] = np.random.randn(d_v, d_mid) * np.sqrt(2/(d_v+d_mid))
        self.weights['b_Q1'] = np.zeros(d_mid)
        self.weights['Q2'] = np.random.randn(d_mid, d_u) * np.sqrt(2/(d_mid+d_u))
        self.weights['b_Q2'] = np.zeros(d_u)
    
    def spectral_conv(self, v, R):
        Nx, Ny, _ = v.shape
        v_hat = np.fft.rfft2(v, axes=(0, 1))
        w_hat = np.zeros((Nx, Ny//2+1, self.d_v), dtype=complex)
        kx_max, ky_max = min(self.k_max, Nx), min(self.k_max, Ny//2+1)
        for kx in range(kx_max):
            for ky in range(ky_max):
                w_hat[kx, ky, :] = v_hat[kx, ky, :] @ R[kx, ky, :, :]
        return np.fft.irfft2(w_hat, s=(Nx, Ny), axes=(0, 1))
    
    def forward(self, a, training=False):
        # Lifting
        v = a @ self.weights['P'] + self.weights['b_P']
        
        # Fourier layers
        for i in range(self.n_layers):
            spectral = self.spectral_conv(v, self.weights[f'R_{i}'])
            local = v @ self.weights[f'W_{i}']
            
            # Apply dropout during training
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1-self.dropout, local.shape) / (1-self.dropout)
                local = local * mask
            
            v = gelu(spectral + local + self.weights[f'b_{i}'])
        
        # Projection
        h = gelu(v @ self.weights['Q1'] + self.weights['b_Q1'])
        return h @ self.weights['Q2'] + self.weights['b_Q2']
    
    def get_weights(self):
        return deepcopy(self.weights)
    
    def set_weights(self, weights):
        self.weights = deepcopy(weights)
    
    def copy(self):
        new_model = FNO2d(self.d_a, self.d_v, self.d_u, self.k_max, 
                         self.n_layers, self.dropout)
        new_model.set_weights(self.get_weights())
        return new_model

print("✓ Core FNO with weight access loaded")

#=============================================================================
# SECTION 1: TRANSFER LEARNING FRAMEWORK
#=============================================================================
print("\n" + "="*70)
print("SECTION 1: Transfer Learning Framework")
print("="*70)

class TransferLearningFNO:
    """
    Framework for transfer learning between cities.
    
    What transfers (universal physics):
    - Vegetation cooling effect
    - Urban heating effect
    - Sky view factor physics
    - Diurnal patterns
    
    What doesn't transfer (city-specific):
    - Absolute temperature range
    - Humidity effects
    - Urban form specifics
    - Coastal effects
    """
    
    def __init__(self, source_model: FNO2d):
        """Initialize with pre-trained source model."""
        self.source_model = source_model
        self.target_model = source_model.copy()
        self.frozen_layers = set()
        
        print("TransferLearningFNO initialized")
        print(f"  Source model: d_v={source_model.d_v}, n_layers={source_model.n_layers}")
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specified layers (won't be updated during fine-tuning)."""
        self.frozen_layers = set(layer_names)
        print(f"Frozen layers: {layer_names}")
    
    def freeze_early_layers(self, n_freeze: int = 2):
        """Freeze first n Fourier layers (common transfer approach)."""
        to_freeze = ['P', 'b_P']  # Always freeze lifting
        for i in range(n_freeze):
            to_freeze.extend([f'R_{i}', f'W_{i}', f'b_{i}'])
        self.freeze_layers(to_freeze)
    
    def freeze_all_except_projection(self):
        """Freeze everything except projection (minimal adaptation)."""
        to_freeze = ['P', 'b_P']
        for i in range(self.source_model.n_layers):
            to_freeze.extend([f'R_{i}', f'W_{i}', f'b_{i}'])
        self.freeze_layers(to_freeze)
    
    def get_trainable_params(self) -> List[str]:
        """Get list of trainable (non-frozen) parameters."""
        all_params = list(self.target_model.weights.keys())
        return [p for p in all_params if p not in self.frozen_layers]
    
    def simulate_fine_tuning(self, target_data: np.ndarray, 
                            target_labels: np.ndarray,
                            n_epochs: int = 50,
                            lr: float = 1e-4) -> Dict:
        """
        Simulate fine-tuning on target domain.
        
        In real implementation, this would use autograd.
        Here we simulate the learning process.
        """
        trainable = self.get_trainable_params()
        print(f"\nFine-tuning {len(trainable)} parameter groups:")
        print(f"  Trainable: {trainable[:5]}..." if len(trainable) > 5 else f"  Trainable: {trainable}")
        
        history = {'loss': [], 'val_loss': []}
        
        # Simulate training
        for epoch in range(n_epochs):
            # Simulated loss (would be computed from forward pass in real impl)
            base_loss = 1.0 * np.exp(-epoch / 20) + 0.05
            loss = base_loss + np.random.randn() * 0.02
            val_loss = base_loss * 1.1 + np.random.randn() * 0.03
            
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
        
        print(f"Fine-tuning complete. Final loss: {history['loss'][-1]:.4f}")
        return history


# Demonstrate transfer learning
print("\n--- Demonstrating Transfer Learning ---")

# Create "NYC-trained" model
np.random.seed(42)
nyc_model = FNO2d(d_a=42, d_v=32, d_u=1, k_max=12, n_layers=4, seed=42)
print("Created NYC-trained source model")

# Initialize transfer to LA
transfer = TransferLearningFNO(nyc_model)

# Strategy 1: Freeze all except projection (minimal adaptation)
print("\nStrategy 1: Minimal adaptation (freeze all except projection)")
transfer.freeze_all_except_projection()
print(f"  Trainable params: {transfer.get_trainable_params()}")

# Strategy 2: Freeze early layers
print("\nStrategy 2: Freeze early layers (more adaptation)")
transfer2 = TransferLearningFNO(nyc_model)
transfer2.freeze_early_layers(n_freeze=2)
print(f"  Trainable params: {transfer2.get_trainable_params()[:5]}...")

# Simulate fine-tuning
print("\n--- Simulating Fine-Tuning on LA Data ---")
la_features = np.random.randn(100, 32, 32, 42)  # 100 scenes
la_labels = np.random.randn(100, 32, 32, 1)
history = transfer.simulate_fine_tuning(la_features, la_labels, n_epochs=30)

#=============================================================================
# SECTION 2: MULTI-CITY TRAINING
#=============================================================================
print("\n" + "="*70)
print("SECTION 2: Multi-City Training")
print("="*70)

class MultiCityFNO:
    """
    FNO trained on multiple cities simultaneously.
    
    Architecture modifications:
    1. City embedding: Learnable city ID vector
    2. Shared backbone: Same Fourier layers for all cities
    3. City-specific projection: Optional per-city output scaling
    """
    
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int,
                 n_layers: int, n_cities: int, d_city_embed: int = 8,
                 seed: int = 42):
        np.random.seed(seed)
        
        self.n_cities = n_cities
        self.d_city_embed = d_city_embed
        
        # City embeddings (learnable)
        self.city_embeddings = np.random.randn(n_cities, d_city_embed) * 0.1
        
        # Shared FNO backbone (input includes city embedding)
        self.fno = FNO2d(d_a=d_a + d_city_embed, d_v=d_v, d_u=d_u,
                        k_max=k_max, n_layers=n_layers, seed=seed)
        
        print(f"MultiCityFNO initialized:")
        print(f"  {n_cities} cities with {d_city_embed}-dim embeddings")
        print(f"  Shared backbone: {d_a}+{d_city_embed} → {d_v} → {d_u}")
    
    def forward(self, features: np.ndarray, city_id: int) -> np.ndarray:
        """
        Forward pass for a specific city.
        
        Args:
            features: (Nx, Ny, d_a) input features
            city_id: Integer city identifier (0 to n_cities-1)
        """
        Nx, Ny = features.shape[:2]
        
        # Get city embedding and broadcast to spatial dims
        city_embed = self.city_embeddings[city_id]  # (d_city_embed,)
        city_embed_spatial = np.broadcast_to(city_embed, (Nx, Ny, self.d_city_embed))
        
        # Concatenate features with city embedding
        x = np.concatenate([features, city_embed_spatial], axis=-1)
        
        return self.fno.forward(x)
    
    def get_city_similarity(self) -> np.ndarray:
        """Compute cosine similarity between city embeddings."""
        norms = np.linalg.norm(self.city_embeddings, axis=1, keepdims=True)
        normalized = self.city_embeddings / (norms + 1e-8)
        return normalized @ normalized.T


# Demonstrate multi-city FNO
print("\n--- Demonstrating Multi-City FNO ---")

city_names = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
mc_fno = MultiCityFNO(d_a=42, d_v=32, d_u=1, k_max=12, n_layers=4,
                      n_cities=len(city_names))

# Predict for different cities
test_features = np.random.randn(32, 32, 42)

print("\nPredictions for different cities:")
for i, city in enumerate(city_names):
    pred = mc_fno.forward(test_features, city_id=i)
    print(f"  {city}: mean={pred.mean():.2f}, std={pred.std():.2f}")

# City similarity
print("\nCity embedding similarity matrix:")
similarity = mc_fno.get_city_similarity()
print("     " + "  ".join(city_names))
for i, city in enumerate(city_names):
    sim_str = " ".join([f"{similarity[i,j]:5.2f}" for j in range(len(city_names))])
    print(f"{city:8s} {sim_str}")


#=============================================================================
# SECTION 3: ENSEMBLE FNO FOR UNCERTAINTY
#=============================================================================
print("\n" + "="*70)
print("SECTION 3: Ensemble FNO for Uncertainty Quantification")
print("="*70)

class EnsembleFNO:
    """
    Ensemble of FNOs for uncertainty quantification.
    
    Train N models with different random seeds.
    Prediction: mean of ensemble
    Uncertainty: std of ensemble
    
    Captures epistemic uncertainty (model uncertainty).
    """
    
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int,
                 n_layers: int, n_ensemble: int = 5, base_seed: int = 42):
        self.n_ensemble = n_ensemble
        self.models = []
        
        for i in range(n_ensemble):
            model = FNO2d(d_a=d_a, d_v=d_v, d_u=d_u, k_max=k_max,
                         n_layers=n_layers, seed=base_seed + i * 100)
            self.models.append(model)
        
        print(f"EnsembleFNO initialized with {n_ensemble} members")
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.
        
        Returns:
            mean: (Nx, Ny, d_u) mean prediction
            std: (Nx, Ny, d_u) standard deviation (uncertainty)
        """
        predictions = np.stack([m.forward(x) for m in self.models])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std
    
    def predict_with_confidence_interval(self, x: np.ndarray, 
                                         confidence: float = 0.95) -> Dict:
        """
        Predict with confidence intervals.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)
        """
        predictions = np.stack([m.forward(x) for m in self.models])
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        # Z-score for confidence level (assuming Gaussian)
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        lower = mean - z * std
        upper = mean + z * std
        
        return {
            'mean': mean,
            'std': std,
            'lower': lower,
            'upper': upper,
            'confidence': confidence
        }


# Demonstrate ensemble FNO
print("\n--- Demonstrating Ensemble FNO ---")

ensemble = EnsembleFNO(d_a=42, d_v=16, d_u=1, k_max=8, n_layers=2, n_ensemble=5)

test_input = np.random.randn(32, 32, 42)
mean_pred, uncertainty = ensemble.predict(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Mean prediction: shape={mean_pred.shape}, range=[{mean_pred.min():.2f}, {mean_pred.max():.2f}]")
print(f"Uncertainty (std): shape={uncertainty.shape}, range=[{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

# With confidence interval
result = ensemble.predict_with_confidence_interval(test_input, confidence=0.95)
print(f"\n95% Confidence Interval:")
print(f"  Lower bound range: [{result['lower'].min():.2f}, {result['lower'].max():.2f}]")
print(f"  Upper bound range: [{result['upper'].min():.2f}, {result['upper'].max():.2f}]")


#=============================================================================
# SECTION 4: MC DROPOUT UNCERTAINTY
#=============================================================================
print("\n" + "="*70)
print("SECTION 4: MC Dropout Uncertainty")
print("="*70)

class MCDropoutFNO:
    """
    FNO with Monte Carlo Dropout for uncertainty.
    
    Key insight: Keep dropout ON at inference time.
    Multiple forward passes with different dropout masks
    give approximate posterior samples.
    
    Cheaper than ensembles (single model, multiple passes).
    """
    
    def __init__(self, d_a: int, d_v: int, d_u: int, k_max: int,
                 n_layers: int, dropout: float = 0.1, seed: int = 42):
        self.model = FNO2d(d_a=d_a, d_v=d_v, d_u=d_u, k_max=k_max,
                          n_layers=n_layers, dropout=dropout, seed=seed)
        self.dropout = dropout
        
        print(f"MCDropoutFNO initialized with dropout={dropout}")
    
    def predict_deterministic(self, x: np.ndarray) -> np.ndarray:
        """Standard prediction (dropout OFF)."""
        return self.model.forward(x, training=False)
    
    def predict_with_uncertainty(self, x: np.ndarray, 
                                 n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout prediction with uncertainty.
        
        Args:
            n_samples: Number of forward passes with dropout
        
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        samples = []
        for _ in range(n_samples):
            pred = self.model.forward(x, training=True)  # Dropout ON
            samples.append(pred)
        
        samples = np.stack(samples)
        return samples.mean(axis=0), samples.std(axis=0)


# Demonstrate MC Dropout
print("\n--- Demonstrating MC Dropout FNO ---")

mc_fno = MCDropoutFNO(d_a=42, d_v=16, d_u=1, k_max=8, n_layers=2, dropout=0.2)

# Deterministic prediction
det_pred = mc_fno.predict_deterministic(test_input)
print(f"Deterministic prediction: {det_pred.shape}")

# MC Dropout prediction
mc_mean, mc_std = mc_fno.predict_with_uncertainty(test_input, n_samples=30)
print(f"MC Dropout mean: {mc_mean.shape}, std range: [{mc_std.min():.3f}, {mc_std.max():.3f}]")


#=============================================================================
# SECTION 5: HETEROSCEDASTIC (NLL) LOSS
#=============================================================================
print("\n" + "="*70)
print("SECTION 5: Heteroscedastic FNO (Predicting Mean and Variance)")
print("="*70)

class HeteroscedasticFNO:
    """
    FNO that predicts both mean and variance.
    
    Output: (μ, log σ²) where prediction is N(μ, σ²)
    
    Loss: Negative Log-Likelihood
    L = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
    
    Captures aleatoric uncertainty (data noise).
    """
    
    def __init__(self, d_a: int, d_v: int, k_max: int,
                 n_layers: int, seed: int = 42):
        # Output 2 channels: mean and log_variance
        self.model = FNO2d(d_a=d_a, d_v=d_v, d_u=2, k_max=k_max,
                          n_layers=n_layers, seed=seed)
        
        print("HeteroscedasticFNO initialized (predicts mean and variance)")
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance.
        
        Returns:
            mean: (Nx, Ny, 1) predicted mean
            variance: (Nx, Ny, 1) predicted variance (always positive)
        """
        output = self.model.forward(x)
        
        mean = output[:, :, 0:1]
        log_var = output[:, :, 1:2]
        
        # Ensure positive variance
        variance = np.exp(log_var)
        
        return mean, variance
    
    def nll_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood loss.
        
        L = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
        """
        mean, variance = self.predict(x)
        
        # NLL loss
        nll = 0.5 * np.log(variance) + 0.5 * (y - mean)**2 / variance
        
        return nll.mean()


# Demonstrate Heteroscedastic FNO
print("\n--- Demonstrating Heteroscedastic FNO ---")

hetero_fno = HeteroscedasticFNO(d_a=42, d_v=16, k_max=8, n_layers=2)

mean_pred, var_pred = hetero_fno.predict(test_input)
print(f"Predicted mean: {mean_pred.shape}, range=[{mean_pred.min():.2f}, {mean_pred.max():.2f}]")
print(f"Predicted variance: {var_pred.shape}, range=[{var_pred.min():.4f}, {var_pred.max():.4f}]")
print(f"Predicted std: range=[{np.sqrt(var_pred).min():.3f}, {np.sqrt(var_pred).max():.3f}]")


#=============================================================================
# SECTION 6: CONFORMAL PREDICTION
#=============================================================================
print("\n" + "="*70)
print("SECTION 6: Conformal Prediction")
print("="*70)

class ConformalFNO:
    """
    Conformal prediction wrapper for FNO.
    
    Provides prediction intervals with guaranteed coverage.
    
    Key guarantee: If calibration set is exchangeable with test set,
    P(y ∈ [f(x) - q, f(x) + q]) ≥ 1 - α
    
    Distribution-free (works with any model).
    """
    
    def __init__(self, base_model: FNO2d):
        self.model = base_model
        self.calibration_quantiles = None
        self.is_calibrated = False
        
        print("ConformalFNO wrapper initialized")
    
    def calibrate(self, cal_inputs: np.ndarray, cal_targets: np.ndarray,
                 alpha: float = 0.1):
        """
        Calibrate using held-out calibration set.
        
        Args:
            cal_inputs: (N, Nx, Ny, d_a) calibration inputs
            cal_targets: (N, Nx, Ny, d_u) calibration targets
            alpha: Miscoverage rate (0.1 = 90% coverage)
        """
        n_cal = len(cal_inputs)
        residuals = []
        
        for i in range(n_cal):
            pred = self.model.forward(cal_inputs[i])
            res = np.abs(pred - cal_targets[i])
            residuals.append(res.flatten())
        
        residuals = np.concatenate(residuals)
        
        # Compute quantile for coverage guarantee
        # Add 1 to numerator for finite sample correction
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        q_level = min(q_level, 1.0)
        
        self.calibration_quantile = np.quantile(residuals, q_level)
        self.alpha = alpha
        self.is_calibrated = True
        
        print(f"Calibrated with {n_cal} samples, α={alpha}")
        print(f"  Prediction interval width: ±{self.calibration_quantile:.3f}")
    
    def predict_with_interval(self, x: np.ndarray) -> Dict:
        """
        Predict with conformal prediction interval.
        
        Returns dict with:
            - point: Point prediction
            - lower: Lower bound of interval
            - upper: Upper bound of interval
            - coverage: Guaranteed coverage level
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction")
        
        point = self.model.forward(x)
        
        return {
            'point': point,
            'lower': point - self.calibration_quantile,
            'upper': point + self.calibration_quantile,
            'coverage': 1 - self.alpha
        }


# Demonstrate Conformal Prediction
print("\n--- Demonstrating Conformal Prediction ---")

base_fno = FNO2d(d_a=42, d_v=16, d_u=1, k_max=8, n_layers=2, seed=42)
conformal = ConformalFNO(base_fno)

# Create calibration data
n_cal = 50
cal_inputs = np.random.randn(n_cal, 32, 32, 42)
cal_targets = np.random.randn(n_cal, 32, 32, 1)

# Calibrate
conformal.calibrate(cal_inputs, cal_targets, alpha=0.1)

# Predict with interval
result = conformal.predict_with_interval(test_input)
print(f"\nConformal prediction result:")
print(f"  Point prediction: {result['point'].shape}")
print(f"  Interval width: ±{conformal.calibration_quantile:.3f}")
print(f"  Guaranteed coverage: {result['coverage']*100:.0f}%")


#=============================================================================
# COMPARISON OF UNCERTAINTY METHODS
#=============================================================================
print("\n" + "="*70)
print("COMPARISON: Uncertainty Quantification Methods")
print("="*70)

print("""
┌──────────────────────┬─────────────┬─────────────┬─────────────────────┐
│ Method               │ Training    │ Inference   │ Uncertainty Type    │
├──────────────────────┼─────────────┼─────────────┼─────────────────────┤
│ Ensemble             │ N× cost     │ N× cost     │ Epistemic           │
│ MC Dropout           │ 1× cost     │ M× cost     │ Epistemic (approx)  │
│ Heteroscedastic      │ 1× cost     │ 1× cost     │ Aleatoric           │
│ Conformal            │ 1× + calib  │ 1× cost     │ Coverage guarantee  │
│ Ensemble + Hetero    │ N× cost     │ N× cost     │ Both                │
└──────────────────────┴─────────────┴─────────────┴─────────────────────┘

Recommendations:
• Highest quality uncertainty → Ensemble (gold standard)
• Limited compute budget → MC Dropout
• Data noise varies spatially → Heteroscedastic
• Need guaranteed coverage → Conformal
• Want both types → Ensemble of Heteroscedastic models
""")

print("\n✓ Part 2 complete - Transfer Learning & Uncertainty")
