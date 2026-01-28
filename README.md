# Geometric Safety Features for AI Embedding Spaces

**Topology-based detection of behavioral instability in AI models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18290279.svg)](https://doi.org/10.5281/zenodo.18290279)
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()

---

## Key Discovery

**Constrained local dimensionality predicts behavioral instability.**

Contrary to intuition, unsafe regions in AI embedding spaces are NOT characterized by high variance. They're characterized by **low participation ratio** and **low spectral entropy** — "narrow passages" where variance is squeezed into few dimensions.

| Feature | Global r | Borderline r | Interpretation |
|---------|----------|--------------|----------------|
| `participation_ratio` | **-0.394** | **-0.529** | LOW = HIGH RISK |
| `spectral_entropy` | **-0.384** | **-0.496** | LOW = HIGH RISK |
| `knn_std_distance` | +0.286 | +0.399 | Tier-0 baseline |

**Simple detection rule:**
```python
HIGH_RISK = (participation_ratio < 30th_percentile) OR (spectral_entropy < 30th_percentile)
# Flags 32.7% of samples with 48.6% precision on borderline cases
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from mirrorfield.geometry import compute_safety_diagnostics

# Compute all features + state mapping
diag = compute_safety_diagnostics(query_embeddings, reference_embeddings)

# Flag high-risk samples
high_risk = diag.get_high_risk_mask(threshold=0.25)
print(f"High risk: {high_risk.sum()}/{len(query_embeddings)}")

# Get all 14 features as matrix
features = diag.get_all_features()  # (N, 14)

# Access specific features
pr = diag.phase2_5_features[:, 5]  # participation_ratio
se = diag.phase2_5_features[:, 4]  # spectral_entropy
```

---

## Features (14 total)

### Tier-0: k-NN Geometry
| Feature | Measures |
|---------|----------|
| `knn_mean_distance` | Average neighbor distance |
| `knn_std_distance` | Neighborhood dispersion |
| `knn_max_distance` | Furthest neighbor (sparsity) |
| `local_curvature` | Manifold anisotropy |
| `ridge_proximity` | Coefficient of variation |

### Phase 1: Flow
| Feature | Measures |
|---------|----------|
| `local_gradient_magnitude` | Density gradient strength |

### Phase 2: Weather Metaphors
| Feature | Measures |
|---------|----------|
| `turbulence_index` | Local mixing/disorder |
| `thermal_gradient` | Boundary-focused gradient |
| `vorticity` | Rotational tendency |

### Phase 5: Topology (Strongest Signals)
| Feature | Measures |
|---------|----------|
| `participation_ratio` | Effective dimensionality (PR = (Σλ)²/Σλ²) |
| `spectral_entropy` | Eigenvalue distribution spread |
| `d_eff` | Dimensions for 90% variance |

### Phase 4: Cognitive States
| State | Meaning | Risk |
|-------|---------|------|
| `uncertain` | Dispersed, boundary-adjacent | HIGH |
| `novel_territory` | Sparse, unfamiliar | HIGH |
| `constraint_pressure` | Multi-basin tension | MEDIUM |
| `searching` | Turbulent exploration | LOW |
| `confident` | Tight clustering | LOW |
| `coherent` | Stable attractor flow | LOW |

---

## The "Narrow Passage" Interpretation

Why does **low** dimensionality predict **high** risk?

Near decision boundaries, the embedding manifold is geometrically constrained. The boundary "squeezes" nearby representations, concentrating variance along boundary-parallel directions. Points in these narrow passages can absorb perturbations in some directions but are sensitive in others — small changes in the constrained directions cause behavioral flips.

```
HIGH participation_ratio → isotropic neighborhood → robust → SAFE
LOW participation_ratio  → anisotropic neighborhood → fragile → RISKY
```

---

## Repository Structure

```
geometric_safety_features/
├── mirrorfield/geometry/
│   ├── bundle.py                     # GeometryBundle API (Tier-0)
│   ├── features.py                   # 7 k-NN features
│   ├── phase1_flow_features.py       # Gradient magnitude
│   ├── phase2_weather_features.py    # Weather + topology features
│   ├── phase3_trajectory_features.py # For generative models
│   ├── phase4_state_mapping.py       # Cognitive state classification
│   ├── unified_pipeline.py           # compute_safety_diagnostics()
│   └── __init__.py                   # v1.5.0
├── experiments/
│   ├── boundary_sliced_evaluation.py # Main evaluation
│   ├── key_findings_analysis.py      # Discovery analysis
│   └── full_pipeline_demo.py         # End-to-end demo
├── examples/
│   └── detection_rule_demo.py        # Simple detection rule
├── paper/
│   └── PAPER_DRAFT.md                # Paper draft
└── docs/
    └── PHASE1_2_5_STATUS.md          # Implementation status
```

---

## Evaluation Results

### Borderline Amplification
Features become MORE predictive in boundary regions:

| Feature | Global r | Borderline r | Amplification |
|---------|----------|--------------|---------------|
| `participation_ratio` | -0.394 | -0.529 | **1.34×** |
| `ridge_proximity` | +0.215 | +0.361 | **1.68×** |
| `thermal_gradient` | +0.226 | +0.372 | **1.65×** |

### Incremental R² Improvement
```
Embeddings only:      R² = 0.741 (global), 0.233 (borderline)
+ Tier-0 features:    R² = 0.770 (+3.9%)
+ Phase 5 topology:   R² = 0.794 (+3.1% additional)
```

---

## Citation

```bibtex
@software{coghlan2026geometric,
  author = {Coghlan, Dillan John},
  title = {Geometric Safety Features for AI Embedding Spaces},
  year = {2026},
  url = {https://github.com/DillanJC/geometric_safety_features},
  doi = {10.5281/zenodo.18290279}
}
```

---

## Requirements

- Python 3.9+
- NumPy, SciPy, scikit-learn
- Optional: PyTorch (GPU acceleration), FAISS (large-scale k-NN)

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

Conceptual development with DeepSeek. Literature grounding by Kosmos/EdisonScientific. Synthesis and implementation with Claude.
