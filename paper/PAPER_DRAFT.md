# Constrained Local Dimensionality Predicts Behavioral Instability in AI Embedding Spaces

**Authors:** Dillan John Coghlan

**Affiliations:** Independent Researcher

---

## Abstract

We introduce a geometric method for diagnosing behavioral instability in AI models by analyzing the local topology of their embedding spaces. Through boundary-stratified evaluation on sentiment classification embeddings, we find that behaviorally unstable regions are characterized not by high variance, but by **constrained local dimensionality**—measured via participation ratio and spectral entropy of local neighborhood covariance. These topology-derived features exhibit strong negative correlations with distance from decision boundaries (participation ratio: r = -0.53 on borderline samples), and their predictive power amplifies by 1.3–1.7× in boundary regions where safety-critical decisions concentrate. We present a simple detection rule achieving 48.6% precision at 32.7% coverage for flagging high-risk inputs. Our results suggest that AI safety evaluation can be augmented by structural analysis of internal representations, complementing traditional output monitoring approaches.

---

## 1. Introduction

AI safety evaluation has traditionally focused on model outputs—measuring accuracy, calibration, and behavioral consistency under perturbation. While effective, this approach treats the model as a black box, offering limited insight into *why* certain inputs provoke unreliable responses.

We propose a complementary approach: analyzing the **geometric structure** of a model's internal representations to identify regions where behavioral instability is likely. The intuition is that embedding spaces encode semantic relationships, and the local geometry around a query point may signal whether the model is operating in well-characterized territory or near decision boundaries where small perturbations cause large behavioral changes.

Our key finding challenges a natural assumption. One might expect that unstable regions would exhibit high local variance—scattered, uncertain representations. Instead, we find the opposite: **constrained local dimensionality** predicts instability. Points near decision boundaries tend to have neighborhoods where variance is concentrated in fewer dimensions, as measured by participation ratio and spectral entropy. This "narrow passage" geometry appears to mark regions where the embedding manifold is squeezed, creating sensitivity to perturbations along the constrained directions.

**Contributions:**
1. We introduce topology-derived features (participation ratio, spectral entropy) for AI safety diagnostics and demonstrate they outperform previously proposed flow-based and distance-based features.
2. We present boundary-stratified evaluation methodology that reveals feature performance specifically in high-risk regions.
3. We provide a simple, interpretable detection rule and open-source implementation.

---

## 2. Related Work

**k-NN Methods for Uncertainty Quantification.** Sun et al. (2022) demonstrated that k-nearest neighbor distances in deep feature spaces effectively detect out-of-distribution samples, outperforming Mahalanobis distance baselines. Bahri et al. (2021) showed that k-NN density estimates on intermediate embeddings detect OOD examples without relying on softmax confidence. Our work extends this line by examining not just distances but the *shape* of local neighborhoods.

**Decision Boundary Analysis.** Prior work has characterized model uncertainty through proximity to decision boundaries (Lee et al., 2018) and local curvature of decision surfaces. We contribute by showing that local dimensionality—orthogonal to boundary distance—provides complementary signal.

**Geometric Features for Safety.** Recent work has explored geometric properties of embedding spaces for safety applications, including local density estimation and manifold curvature. Our contribution is demonstrating that topology-derived features (participation ratio, spectral entropy) substantially outperform simpler geometric measures.

**Participation Ratio in Physics and Neuroscience.** Participation ratio, defined as PR = (Σλ)²/Σλ², quantifies the effective dimensionality of a distribution. It has been used extensively in physics (localization phenomena) and neuroscience (neural population coding). To our knowledge, this is the first application to AI safety diagnostics.

---

## 3. Method

### 3.1 Problem Setting

Given a pre-trained model with embedding layer producing representations x ∈ ℝᴰ, we seek features that predict behavioral instability—operationalized as proximity to decision boundaries where model outputs are sensitive to small input perturbations.

We use boundary distance as our target variable: signed distance from a linear decision boundary fit on training embeddings. Negative values indicate the model's prediction disagrees with the boundary-implied label (misclassified region); values near zero indicate boundary proximity (uncertain region); positive values indicate confident correct predictions.

### 3.2 Boundary-Stratified Evaluation

Standard aggregate metrics can obscure where features provide value. We stratify evaluation into three zones:

- **Safe zone:** boundary distance > 0.5 (confident, correct)
- **Borderline zone:** |boundary distance| ≤ 0.5 (uncertain)
- **Unsafe zone:** boundary distance < -0.5 (confident, incorrect)

This reveals that features useful in aggregate may fail precisely where they're needed most, and vice versa.

### 3.3 Feature Extraction

We compute features from k-nearest neighbors (k=50) in a reference embedding set.

**Tier-0 (Baseline k-NN Features):**
- `knn_mean_distance`, `knn_std_distance`, `knn_max_distance`
- `ridge_proximity` (coefficient of variation)
- `local_curvature` (SVD-based anisotropy)

**Phase 1 (Flow Features):**
- `local_gradient_magnitude`: norm of mean-shift vector, measuring pull toward density modes

**Phase 5 (Topology Features):**
Given the k nearest neighbors, compute the local covariance matrix and its eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₖ.

- **Spectral entropy:** H = -Σ pᵢ log(pᵢ) where pᵢ = λᵢ/Σλ
- **Participation ratio:** PR = (Σλ)²/Σλ²
- **Effective dimension:** d_eff = min{d : Σᵢ≤ᵈ λᵢ ≥ 0.9·Σλ}

Low spectral entropy indicates eigenvalue concentration (one or few dominant directions). Low participation ratio indicates the same. Both signal constrained local geometry.

### 3.4 Simple Detection Rule

Based on our findings, we propose:

```
HIGH_RISK := (participation_ratio < P₃₀) OR (spectral_entropy < P₃₀)
```

where P₃₀ denotes the 30th percentile computed on a reference set.

---

## 4. Experiments

### 4.1 Dataset

We evaluate on sentiment classification embeddings (N=1,099) from OpenAI's text-embedding-3-large model (D=256, L2 normalized). We use 80% as reference and 20% as query set. Boundary distances are computed from a linear SVM fit on the reference set.

### 4.2 Evaluation Protocol

For each feature, we compute:
1. Pearson correlation with boundary distance (global and per-zone)
2. Incremental R² improvement when added to baseline models
3. Statistical significance via permutation testing

### 4.3 Results

**Table 1: Feature Correlations with Boundary Distance**

| Rank | Feature | Global r | Borderline r | Amplification |
|------|---------|----------|--------------|---------------|
| 1 | participation_ratio | -0.394*** | -0.529*** | 1.34× |
| 2 | spectral_entropy | -0.384*** | -0.496*** | 1.29× |
| 3 | knn_max_distance | +0.345*** | +0.168 | 0.49× |
| 4 | local_gradient_magnitude | +0.330*** | — | — |
| 5 | knn_std_distance | +0.286*** | +0.399*** | 1.40× |
| 6 | thermal_gradient | +0.226*** | +0.372*** | 1.65× |
| 7 | ridge_proximity | +0.215** | +0.361*** | 1.68× |

***: p < 0.001, **: p < 0.01

**Key observations:**

1. **Topology features dominate.** Participation ratio and spectral entropy show the strongest correlations, surpassing both Tier-0 baselines and flow-based features.

2. **Negative correlation.** The topology features exhibit *negative* correlation: lower participation ratio / spectral entropy corresponds to *closer* boundary proximity (higher risk).

3. **Borderline amplification.** Most features become more predictive in the borderline zone, with amplification factors of 1.3–1.7×. This is the opposite of what would occur if features merely correlated with easy-to-classify samples.

**Table 2: Incremental R² Improvement**

| Model | Global R² | Borderline R² |
|-------|-----------|---------------|
| Embeddings only | 0.741 | 0.233 |
| + Tier-0 features | 0.770 (+3.9%) | 0.254 (+9.0%) |
| + Phase 1 flow | 0.773 (+0.4%) | 0.276 (+8.7%) |
| + Phase 5 topology | 0.794 (+2.7%) | — |

**Table 3: Simple Detection Rule Performance**

| Metric | Value |
|--------|-------|
| Samples flagged | 72/220 (32.7%) |
| Precision (flagged ∩ borderline) | 48.6% |
| Recall (borderline flagged) | 44.3% |

---

## 5. Analysis

### 5.1 The "Narrow Passage" Interpretation

Why does *constrained* local dimensionality predict instability? We propose the following interpretation:

Near decision boundaries, the embedding manifold is geometrically constrained. The boundary itself imposes a hyperplane that "squeezes" nearby representations. Points in these regions have neighborhoods where variance is concentrated along the boundary-parallel directions, with little variance in the boundary-orthogonal direction (which determines class membership).

This creates a "narrow passage" geometry: the representation has freedom to vary along certain dimensions but is tightly constrained along others. Small perturbations in the constrained directions can push the point across the boundary, causing behavioral flips.

In contrast, points deep within class clusters have isotropic neighborhoods—variance is spread across many dimensions, yielding high participation ratio. These points can absorb perturbations without crossing boundaries.

### 5.2 Why Negative Correlation?

The negative correlation between topology features and boundary distance (where positive = safe) admits a geometric explanation:

- **High PR / High H_spec:** Variance spread across dimensions → isotropic neighborhood → robust to perturbations → far from boundary → positive boundary distance
- **Low PR / Low H_spec:** Variance concentrated → anisotropic neighborhood → sensitive to perturbations → near boundary → low/negative boundary distance

This is consistent with the manifold hypothesis: decision boundaries partition the embedding manifold, and the geometry near these partitions differs systematically from the interior.

### 5.3 Limitations

1. **Single dataset.** Our evaluation uses one sentiment classification dataset. Cross-domain validation is needed.
2. **Linear boundary assumption.** We use linear SVM boundaries; real decision surfaces may be nonlinear.
3. **Embedding model dependence.** Results may vary across embedding models and architectures.
4. **Correlation vs. causation.** We demonstrate predictive correlation, not that constrained geometry *causes* instability.

---

## 6. Discussion

### Implications for AI Safety

Our findings suggest a shift in AI safety evaluation: from monitoring outputs to analyzing the **structural geometry of internal representations**. This offers several advantages:

1. **Proactive detection.** Flag risky inputs *before* generating outputs.
2. **Model-agnostic.** Applies to any model producing embeddings.
3. **Interpretable.** "Low participation ratio" is more explanatory than "model said something wrong."

### Practical Deployment

The simple detection rule (PR or H_spec below 30th percentile) provides an actionable baseline. In deployment:
- Compute reference statistics on known-safe inputs
- Flag new inputs falling below thresholds for human review
- Adjust thresholds based on precision/recall requirements

### Future Work

1. **Cross-model validation.** Test whether topology features transfer across embedding models.
2. **Nonlinear boundaries.** Extend to neural network decision surfaces.
3. **Data poisoning detection.** Investigate whether poisoned samples exhibit distinctive topology.
4. **Trajectory analysis.** For generative models, track topology features through generation.

---

## 7. Conclusion

We have shown that local topology features—specifically participation ratio and spectral entropy—are strong predictors of behavioral instability in AI embedding spaces. The discovery that *constrained* local dimensionality signals risk challenges intuitions about uncertainty geometry and opens new directions for AI safety diagnostics.

Our boundary-stratified evaluation methodology reveals that these features are most informative precisely in the high-risk regions where safety matters most. The provided detection rule and open-source implementation enable practical deployment.

This work contributes to a broader research program: understanding the geometric structure of AI cognition to build safer, more interpretable systems.

---

## References

Bahri, D., Jiang, H., & Tay, Y. (2021). Label Smoothed Embedding Hypothesis for Out-of-Distribution Detection. arXiv:2102.13100.

Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS.

Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. ICML.

---

## Appendix A: Implementation Details

All code is available at: https://github.com/DillanJC/geometric_safety_features

**Key functions:**
- `compute_safety_diagnostics()`: Main entry point for full pipeline
- `compute_topology_lite_features()`: Participation ratio, spectral entropy, d_eff
- `compute_state_scores()`: Maps geometry to interpretable cognitive states

**Computational requirements:**
- Single k-NN query (reused across features)
- SVD on k×D matrices (k=50, D=256)
- O(N·k·D) time complexity
- Runs in <1 second for N=1000 on consumer hardware

---

## Appendix B: Feature Definitions

**Participation Ratio:**
$$PR = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

Ranges from 1 (single dominant eigenvalue) to k (uniform eigenvalues).

**Spectral Entropy:**
$$H_{spec} = -\sum_i p_i \log(p_i), \quad p_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

Higher entropy indicates more uniform eigenvalue distribution.

**Effective Dimension:**
$$d_{eff} = \min\{d : \sum_{i=1}^{d} \lambda_i \geq 0.9 \cdot \sum_j \lambda_j\}$$

Number of dimensions capturing 90% of variance.
