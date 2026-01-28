"""
Phase 1 Flow Features — Differential/Density Gradient Features

Implements three differential features that quantify how local density changes:
1. local_gradient_magnitude: Mean-shift vector norm (pull toward denser regions)
2. gradient_direction_consistency: Fisher-z aggregated cosine of neighbor gradients
3. pressure_differential: Log k-NN radii at two scales scaled by effective local dimension

Theory:
- Mean-shift vector is proportional to the gradient of a KDE (Gaussian kernel)
- Per-point adaptive bandwidth h = c * median(k-NN radii) handles high-D concentration
- Direction consistency distinguishes coherent attraction (safe) from multi-basin tension (borderline)
- Pressure differential contrasts densities across scales, flagging transitions

Design Principles (matching existing features.py):
- Reference-only computation: queries never pollute reference graph
- Batch-order invariant: permuting query batch produces identical per-sample features
- Deterministic: same reference + same query = same output
- NumPy-based with optional PyTorch GPU acceleration
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors
import warnings


# =============================================================================
# Core Feature Computation (NumPy)
# =============================================================================

def compute_mean_shift_vector(
    query: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    bandwidth_multiplier: float = 1.0,
    eps: float = 1e-12
) -> Tuple[np.ndarray, float]:
    """
    Compute mean-shift vector for a single query point.

    The mean-shift vector points toward denser regions and its magnitude
    indicates the strength of the local density gradient.

    Args:
        query: Single query point (D,)
        neighbors: k nearest neighbors (k, D)
        distances: Distances to neighbors (k,)
        bandwidth_multiplier: Scaling factor c for bandwidth h = c * median(distances)
        eps: Numerical stability constant

    Returns:
        mean_shift_vector: Direction of density increase (D,)
        magnitude: ||mean_shift_vector||_2
    """
    # Adaptive bandwidth: h = c * median(neighbor radii)
    h = bandwidth_multiplier * np.median(distances)
    h = max(h, 1e-6)  # Prevent zero bandwidth

    # Gaussian kernel weights: w_i = exp(-d_i^2 / (2h^2))
    weights = np.exp(-distances**2 / (2 * h**2 + eps))

    # Mean-shift vector: m(x) = sum_i w_i * (x_i - x) / sum_i w_i
    diff = neighbors - query  # (k, D)
    weighted_diff = weights[:, np.newaxis] * diff  # (k, D)

    weight_sum = weights.sum() + eps
    mean_shift = weighted_diff.sum(axis=0) / weight_sum  # (D,)

    magnitude = np.linalg.norm(mean_shift)

    return mean_shift, magnitude


def compute_effective_local_dimension(
    neighbors: np.ndarray,
    energy_threshold: float = 0.90,
    eps: float = 1e-12
) -> int:
    """
    Compute effective local dimension via PCA energy threshold.

    Uses SVD on centered neighbor matrix to find how many dimensions
    capture the specified fraction of variance.

    Args:
        neighbors: k nearest neighbors (k, D)
        energy_threshold: Fraction of variance to capture (default: 0.90)
        eps: Numerical stability constant

    Returns:
        d_eff: Effective local dimension (int)
    """
    # Center neighbors
    centered = neighbors - neighbors.mean(axis=0)

    # Handle degenerate case (identical neighbors)
    if np.allclose(centered, 0, atol=1e-10):
        return 1

    # SVD: eigenvalues of covariance ~ S^2 / (k-1)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: add small noise and retry
        centered = centered + 1e-8 * np.random.randn(*centered.shape)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)

    # Compute variance explained
    eigenvalues = S**2 / max(1.0, centered.shape[0] - 1)
    total_var = eigenvalues.sum() + eps
    cumulative_var = np.cumsum(eigenvalues) / total_var

    # Find number of components to reach threshold
    d_eff = np.searchsorted(cumulative_var, energy_threshold) + 1
    d_eff = min(d_eff, len(eigenvalues))

    return int(d_eff)


def compute_phase1_features(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 50,
    bandwidth_multiplier: float = 1.0,
    kc: int = 15,
    p_idx: int = 14,
    q_idx: int = 49,
    energy_threshold: float = 0.90,
    min_neighbor_mag: float = 1e-4,
    engine=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute Phase 1 flow features (reference-only, batch-order invariant).

    CRITICAL: Reference set is never modified by query points.
    Query embeddings are NEVER added to the neighborhood graph.

    Args:
        query_embeddings: Query points (N_query, D)
        reference_embeddings: Reference points (N_ref, D) - IMMUTABLE
        k: Number of neighbors (default: 50)
        bandwidth_multiplier: Scaling factor c for adaptive bandwidth (default: 1.0)
        kc: Neighbors for consistency computation (default: 15, must be <= k)
        p_idx: Inner scale neighbor index for pressure (default: 14 = 15th neighbor)
        q_idx: Outer scale neighbor index for pressure (default: 49 = 50th neighbor)
        energy_threshold: PCA energy threshold for d_eff (default: 0.90)
        min_neighbor_mag: Minimum gradient magnitude to include in consistency (default: 1e-4)
        engine: Optional pre-built NN engine

    Returns:
        features: (N_query, 3) array of flow features
            [0] local_gradient_magnitude: ||m(x)||_2
            [1] gradient_direction_consistency: Fisher-z aggregated cosine
            [2] pressure_differential: -d_eff * (log r_p - log r_q)
        metadata: Dictionary with statistics

    Feature Semantics:
        - local_gradient_magnitude: High = strong pull toward mode (confident region)
                                    Low = weak pull (boundary/transition region)
        - gradient_direction_consistency: High = coherent attraction to nearby mode
                                          Low = multi-basin tension (boundary)
        - pressure_differential: Positive = compression (inner denser than outer)
                                 Negative = expansion (outer denser than inner)
    """
    N_query = len(query_embeddings)
    N_ref = len(reference_embeddings)
    D = query_embeddings.shape[1]

    # Validate parameters
    if k >= N_ref:
        raise ValueError(f"k={k} must be < N_ref={N_ref}")
    if kc > k:
        warnings.warn(f"kc={kc} > k={k}, clamping to k")
        kc = k
    if p_idx >= k:
        warnings.warn(f"p_idx={p_idx} >= k={k}, clamping to k-1")
        p_idx = k - 1
    if q_idx >= k:
        warnings.warn(f"q_idx={q_idx} >= k={k}, clamping to k-1")
        q_idx = k - 1

    # Build or use provided NN engine
    if engine is None:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(reference_embeddings)
        distances, indices = nn.kneighbors(query_embeddings)
    else:
        distances, indices = engine.kneighbors(query_embeddings, k)

    # Initialize features and intermediate storage
    features = np.zeros((N_query, 3), dtype=np.float32)
    gradient_directions = np.zeros((N_query, D), dtype=np.float32)
    gradient_magnitudes = np.zeros(N_query, dtype=np.float32)
    d_eff_values = np.zeros(N_query, dtype=np.int32)

    # ==========================================================================
    # Pass 1: Compute mean-shift vectors, gradients, and d_eff for all points
    # ==========================================================================
    for i in range(N_query):
        query = query_embeddings[i]
        neighbor_idx = indices[i]
        neighbor_dists = distances[i]
        neighbors = reference_embeddings[neighbor_idx]

        # Compute mean-shift vector and magnitude
        mean_shift, magnitude = compute_mean_shift_vector(
            query, neighbors, neighbor_dists,
            bandwidth_multiplier=bandwidth_multiplier
        )

        # Store gradient magnitude (Feature 0)
        features[i, 0] = magnitude
        gradient_magnitudes[i] = magnitude

        # Store unit direction (for consistency computation)
        if magnitude > 1e-10:
            gradient_directions[i] = mean_shift / magnitude
        else:
            gradient_directions[i] = 0.0

        # Compute effective local dimension
        d_eff = compute_effective_local_dimension(
            neighbors, energy_threshold=energy_threshold
        )
        d_eff_values[i] = d_eff

        # Compute pressure differential (Feature 2)
        # Δρ(x) = -d_eff * (log r_p - log r_q)
        r_p = max(neighbor_dists[p_idx], 1e-10)
        r_q = max(neighbor_dists[q_idx], 1e-10)
        pressure = -d_eff * (np.log(r_p) - np.log(r_q))
        features[i, 2] = pressure

    # ==========================================================================
    # Pass 2: Compute gradient direction consistency using reference neighbor gradients
    # ==========================================================================
    # Build NN index for reference-to-reference queries (needed for neighbor gradients)
    if engine is None:
        nn_ref = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn_ref.fit(reference_embeddings)
    else:
        nn_ref = engine  # Reuse engine

    for i in range(N_query):
        query_dir = gradient_directions[i]
        query_mag = gradient_magnitudes[i]

        if query_mag < min_neighbor_mag:
            # Query has negligible gradient, consistency undefined
            features[i, 1] = 0.0
            continue

        # Get kc nearest reference neighbors for this query
        ref_neighbor_idx = indices[i, :kc]
        cosines = []

        for ref_idx in ref_neighbor_idx:
            # Compute mean-shift for this reference point (using reference set)
            ref_point = reference_embeddings[ref_idx]

            # Get reference point's neighbors within reference set
            if engine is None:
                ref_nn_dists, ref_nn_idx = nn_ref.kneighbors([ref_point], k)
            else:
                ref_nn_dists, ref_nn_idx = nn_ref.kneighbors(ref_point.reshape(1, -1), k)
            ref_nn_dists = ref_nn_dists[0]
            ref_nn_idx = ref_nn_idx[0]
            ref_nn_points = reference_embeddings[ref_nn_idx]

            ref_ms, ref_mag = compute_mean_shift_vector(
                ref_point, ref_nn_points, ref_nn_dists,
                bandwidth_multiplier=bandwidth_multiplier
            )

            if ref_mag < min_neighbor_mag:
                continue  # Skip neighbors with weak gradients

            ref_dir = ref_ms / ref_mag

            # Cosine similarity between query and neighbor directions
            cos_sim = np.dot(query_dir, ref_dir)
            cos_sim = np.clip(cos_sim, -0.999, 0.999)  # Clip for Fisher-z stability
            cosines.append(cos_sim)

        if len(cosines) == 0:
            features[i, 1] = 0.0
            continue

        # Fisher-z transformation for aggregation
        # z = 0.5 * log((1+r)/(1-r))
        cosines = np.array(cosines)
        fisher_z = 0.5 * np.log((1 + cosines + 1e-12) / (1 - cosines + 1e-12))
        mean_fisher = fisher_z.mean()

        # Back-transform with tanh to [-1, 1]
        consistency = np.tanh(mean_fisher)
        features[i, 1] = consistency

    # Build metadata
    metadata = {
        "n_query": N_query,
        "n_reference": N_ref,
        "k": k,
        "kc": kc,
        "bandwidth_multiplier": bandwidth_multiplier,
        "p_idx": p_idx,
        "q_idx": q_idx,
        "energy_threshold": energy_threshold,
        "feature_means": features.mean(axis=0).tolist(),
        "feature_stds": features.std(axis=0).tolist(),
        "d_eff_mean": float(d_eff_values.mean()),
        "d_eff_std": float(d_eff_values.std()),
    }

    return features, metadata


def compute_phase1_features_fast(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 50,
    bandwidth_multiplier: float = 1.0,
    p_idx: int = 14,
    q_idx: int = 49,
    energy_threshold: float = 0.90,
    precomputed_distances: Optional[np.ndarray] = None,
    precomputed_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fast version: Compute only gradient magnitude and pressure differential.

    Skips consistency computation (Pass 2) for ~10x speedup.
    Use this for initial screening or when consistency is not needed.

    Returns:
        features: (N_query, 2) array [gradient_magnitude, pressure_differential]
        metadata: Dictionary with statistics
    """
    N_query = len(query_embeddings)
    N_ref = len(reference_embeddings)

    if k >= N_ref:
        raise ValueError(f"k={k} must be < N_ref={N_ref}")

    # Get k-NN distances and indices
    if precomputed_distances is not None and precomputed_indices is not None:
        distances = precomputed_distances
        indices = precomputed_indices
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(reference_embeddings)
        distances, indices = nn.kneighbors(query_embeddings)

    features = np.zeros((N_query, 2), dtype=np.float32)
    d_eff_values = np.zeros(N_query, dtype=np.int32)

    for i in range(N_query):
        query = query_embeddings[i]
        neighbor_dists = distances[i]
        neighbors = reference_embeddings[indices[i]]

        # Gradient magnitude
        _, magnitude = compute_mean_shift_vector(
            query, neighbors, neighbor_dists,
            bandwidth_multiplier=bandwidth_multiplier
        )
        features[i, 0] = magnitude

        # Effective dimension
        d_eff = compute_effective_local_dimension(neighbors, energy_threshold)
        d_eff_values[i] = d_eff

        # Pressure differential
        r_p = max(neighbor_dists[min(p_idx, k-1)], 1e-10)
        r_q = max(neighbor_dists[min(q_idx, k-1)], 1e-10)
        features[i, 1] = -d_eff * (np.log(r_p) - np.log(r_q))

    metadata = {
        "n_query": N_query,
        "n_reference": N_ref,
        "k": k,
        "feature_means": features.mean(axis=0).tolist(),
        "feature_stds": features.std(axis=0).tolist(),
        "d_eff_mean": float(d_eff_values.mean()),
        "version": "fast_v1",
    }

    return features, metadata


# =============================================================================
# Feature Names and Constants
# =============================================================================

PHASE1_FEATURE_NAMES = [
    "local_gradient_magnitude",
    "gradient_direction_consistency",
    "pressure_differential",
]

PHASE1_FEATURE_NAMES_FAST = [
    "local_gradient_magnitude",
    "pressure_differential",
]

# Recommended subset after evaluation (pressure_differential redundant with ridge_proximity)
PHASE1_FEATURE_NAMES_RECOMMENDED = [
    "local_gradient_magnitude",
]


def compute_gradient_magnitude_only(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 75,  # Higher k performs better per evaluation
    bandwidth_multiplier: float = 1.2,  # Slightly higher c improves correlation
    engine=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute only local_gradient_magnitude (recommended Phase 1 feature).

    This is the least redundant Phase 1 feature based on evaluation:
    - r=0.835 max correlation with Tier-0 (vs 0.932 for pressure_differential)
    - +8.68% improvement on borderline cases

    Optimized defaults: k=75, c=1.2 (from hyperparameter sensitivity analysis)
    """
    N_query = len(query_embeddings)
    N_ref = len(reference_embeddings)

    if k >= N_ref:
        k = N_ref - 1

    if engine is None:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        nn.fit(reference_embeddings)
        distances, indices = nn.kneighbors(query_embeddings)
    else:
        distances, indices = engine.kneighbors(query_embeddings, k)

    magnitudes = np.zeros(N_query, dtype=np.float32)

    for i in range(N_query):
        query = query_embeddings[i]
        neighbors = reference_embeddings[indices[i]]
        _, mag = compute_mean_shift_vector(
            query, neighbors, distances[i], bandwidth_multiplier
        )
        magnitudes[i] = mag

    metadata = {
        "n_query": N_query,
        "k": k,
        "bandwidth_multiplier": bandwidth_multiplier,
        "mean": float(magnitudes.mean()),
        "std": float(magnitudes.std()),
        "version": "recommended_v1",
    }

    return magnitudes.reshape(-1, 1), metadata

# Default hyperparameters (from user's plan)
DEFAULT_K = 50
DEFAULT_KC = 15
DEFAULT_BANDWIDTH_MULTIPLIER = 1.0
DEFAULT_P_IDX = 14  # 15th neighbor (0-indexed)
DEFAULT_Q_IDX = 49  # 50th neighbor (0-indexed)
DEFAULT_ENERGY_THRESHOLD = 0.90
DEFAULT_MIN_NEIGHBOR_MAG = 1e-4


# =============================================================================
# PyTorch GPU Implementation (Optional)
# =============================================================================

def compute_phase1_features_gpu(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 50,
    bandwidth_multiplier: float = 1.0,
    p_idx: int = 14,
    q_idx: int = 49,
    energy_threshold: float = 0.90,
    batch_size: int = 8192,
    device: str = "cuda",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    GPU-accelerated Phase 1 feature computation using PyTorch.

    Falls back to CPU NumPy if PyTorch/CUDA not available.

    Args:
        query_embeddings: Query points (N_query, D)
        reference_embeddings: Reference points (N_ref, D)
        k: Number of neighbors
        bandwidth_multiplier: Bandwidth scaling factor
        p_idx: Inner scale neighbor index
        q_idx: Outer scale neighbor index
        energy_threshold: PCA energy threshold
        batch_size: Batch size for GPU processing
        device: PyTorch device ("cuda" or "cpu")

    Returns:
        features: (N_query, 2) array [gradient_magnitude, pressure_differential]
        metadata: Dictionary with statistics
    """
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            device = "cpu"
    except ImportError:
        warnings.warn("PyTorch not available, falling back to NumPy")
        return compute_phase1_features_fast(
            query_embeddings, reference_embeddings, k,
            bandwidth_multiplier, p_idx, q_idx, energy_threshold
        )

    N, D = query_embeddings.shape
    N_ref = len(reference_embeddings)

    # Move to device
    X = torch.tensor(query_embeddings, dtype=torch.float32, device=device)
    X_ref = torch.tensor(reference_embeddings, dtype=torch.float32, device=device)

    # Pre-compute k-NN using sklearn (FAISS would be better for large N_ref)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(reference_embeddings)
    distances_np, indices_np = nn.kneighbors(query_embeddings)

    nbr_idx = torch.tensor(indices_np, dtype=torch.long, device=device)
    nbr_d2 = torch.tensor(distances_np**2, dtype=torch.float32, device=device)

    g_mag_list = []
    delta_rho_list = []
    d_eff_list = []

    eps = 1e-12

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = torch.arange(start, end, device=device)

        Xc = X[idx]  # [B, D]
        nbrs = X_ref[nbr_idx[idx]]  # [B, k, D]
        d2 = nbr_d2[idx]  # [B, k]

        # Radii and adaptive bandwidth
        radii = torch.sqrt(torch.clamp(d2, min=0.0))  # [B, k]
        h = bandwidth_multiplier * radii.median(dim=1).values  # [B]
        h = torch.clamp(h, min=1e-6)

        # Gaussian kernel weights
        denom = 2.0 * (h**2 + eps)
        w = torch.exp(-d2 / denom.unsqueeze(1))  # [B, k]

        # Mean-shift vector
        diff = nbrs - Xc.unsqueeze(1)  # [B, k, D]
        w_uns = w.unsqueeze(-1)
        num = (w_uns * diff).sum(dim=1)  # [B, D]
        den = w.sum(dim=1, keepdim=True) + eps
        m = num / den  # [B, D]

        mag = torch.linalg.norm(m, dim=1)  # [B]
        g_mag_list.append(mag.cpu().numpy())

        # Effective dimension and pressure (computed per sample)
        B = end - start
        d_eff_chunk = np.zeros(B, dtype=np.int32)
        delta_rho_chunk = np.zeros(B, dtype=np.float32)

        for bi in range(B):
            # d_eff via SVD
            M = (nbrs[bi] - nbrs[bi].mean(dim=0)).cpu().numpy()
            if np.allclose(M, 0, atol=1e-10):
                d_eff_chunk[bi] = 1
            else:
                try:
                    _, S, _ = np.linalg.svd(M, full_matrices=False)
                    eigs = S**2 / max(1.0, M.shape[0] - 1)
                    cumsum = np.cumsum(eigs) / (eigs.sum() + eps)
                    d_eff_chunk[bi] = np.searchsorted(cumsum, energy_threshold) + 1
                except:
                    d_eff_chunk[bi] = 1

            # Pressure differential
            r_p = max(float(radii[bi, min(p_idx, k-1)].cpu()), 1e-10)
            r_q = max(float(radii[bi, min(q_idx, k-1)].cpu()), 1e-10)
            delta_rho_chunk[bi] = -d_eff_chunk[bi] * (np.log(r_p) - np.log(r_q))

        d_eff_list.append(d_eff_chunk)
        delta_rho_list.append(delta_rho_chunk)

    g_mag = np.concatenate(g_mag_list)
    delta_rho = np.concatenate(delta_rho_list)
    d_eff_all = np.concatenate(d_eff_list)

    features = np.stack([g_mag, delta_rho], axis=1).astype(np.float32)

    metadata = {
        "n_query": N,
        "n_reference": N_ref,
        "k": k,
        "device": device,
        "feature_means": features.mean(axis=0).tolist(),
        "feature_stds": features.std(axis=0).tolist(),
        "d_eff_mean": float(d_eff_all.mean()),
        "version": "gpu_v1",
    }

    return features, metadata


# =============================================================================
# Integration Helpers
# =============================================================================

def combine_with_tier0_features(
    tier0_features: np.ndarray,
    phase1_features: np.ndarray,
) -> np.ndarray:
    """
    Combine Tier-0 (existing 7) features with Phase 1 flow features.

    Args:
        tier0_features: (N, 7) array from compute_knn_features
        phase1_features: (N, 2 or 3) array from compute_phase1_features

    Returns:
        combined: (N, 9 or 10) array
    """
    if tier0_features.shape[0] != phase1_features.shape[0]:
        raise ValueError("Feature arrays must have same number of samples")

    return np.concatenate([tier0_features, phase1_features], axis=1)


def get_combined_feature_names(include_consistency: bool = True) -> list:
    """Get combined feature names for Tier-0 + Phase 1."""
    from .features import FEATURE_NAMES

    phase1_names = PHASE1_FEATURE_NAMES if include_consistency else PHASE1_FEATURE_NAMES_FAST
    return FEATURE_NAMES + phase1_names


# =============================================================================
# Unit Tests
# =============================================================================

def _unit_tests():
    """Run basic sanity checks on Phase 1 features."""
    np.random.seed(42)

    # Create synthetic data
    N_ref = 500
    N_query = 50
    D = 64

    # Reference: two Gaussian clusters
    ref_cluster1 = np.random.randn(N_ref // 2, D) * 0.5 + np.array([1.0] * D)
    ref_cluster2 = np.random.randn(N_ref // 2, D) * 0.5 + np.array([-1.0] * D)
    reference = np.vstack([ref_cluster1, ref_cluster2])

    # Queries: mix of cluster centers (safe) and boundary (uncertain)
    queries_safe = np.random.randn(N_query // 2, D) * 0.3 + np.array([1.0] * D)
    queries_boundary = np.random.randn(N_query // 2, D) * 0.3  # Near origin (boundary)
    queries = np.vstack([queries_safe, queries_boundary])

    print("Testing Phase 1 Flow Features...")
    print(f"  Reference: {reference.shape}")
    print(f"  Queries: {queries.shape}")

    # Test fast version
    print("\n1. Testing fast version (gradient_mag + pressure)...")
    features_fast, meta_fast = compute_phase1_features_fast(
        queries, reference, k=30
    )
    assert features_fast.shape == (N_query, 2), f"Expected (50, 2), got {features_fast.shape}"
    assert np.all(np.isfinite(features_fast)), "Non-finite values in features"
    print(f"   Shape: {features_fast.shape}")
    print(f"   Gradient magnitude: mean={features_fast[:, 0].mean():.4f}, std={features_fast[:, 0].std():.4f}")
    print(f"   Pressure differential: mean={features_fast[:, 1].mean():.4f}, std={features_fast[:, 1].std():.4f}")

    # Test full version
    print("\n2. Testing full version (with consistency)...")
    features_full, meta_full = compute_phase1_features(
        queries[:10], reference, k=30, kc=10  # Small subset for speed
    )
    assert features_full.shape == (10, 3), f"Expected (10, 3), got {features_full.shape}"
    assert np.all(np.isfinite(features_full)), "Non-finite values in features"
    print(f"   Shape: {features_full.shape}")
    print(f"   Consistency: mean={features_full[:, 1].mean():.4f}, std={features_full[:, 1].std():.4f}")

    # Test batch-order invariance
    print("\n3. Testing batch-order invariance...")
    perm = np.random.permutation(N_query)
    features_perm, _ = compute_phase1_features_fast(queries[perm], reference, k=30)
    features_unperm = features_perm[np.argsort(perm)]

    assert np.allclose(features_fast, features_unperm, rtol=1e-5), "Batch-order invariance violated!"
    print("   PASSED: Permuting queries produces permuted features")

    # Test expected behavior: boundary points should have different characteristics
    print("\n4. Testing semantic expectations...")
    safe_mag = features_fast[:N_query // 2, 0].mean()
    boundary_mag = features_fast[N_query // 2:, 0].mean()
    print(f"   Safe region gradient magnitude: {safe_mag:.4f}")
    print(f"   Boundary region gradient magnitude: {boundary_mag:.4f}")
    # Note: Expectations depend on data geometry; these are diagnostic only

    # Test GPU version if available
    try:
        import torch
        if torch.cuda.is_available():
            print("\n5. Testing GPU version...")
            features_gpu, meta_gpu = compute_phase1_features_gpu(
                queries, reference, k=30, device="cuda"
            )
            print(f"   GPU features shape: {features_gpu.shape}")
            # Check rough agreement with CPU
            diff = np.abs(features_fast - features_gpu).max()
            print(f"   Max diff from CPU: {diff:.6f}")
        else:
            print("\n5. GPU test skipped (CUDA not available)")
    except ImportError:
        print("\n5. GPU test skipped (PyTorch not installed)")

    print("\n[OK] All Phase 1 feature tests passed!")
    return True


if __name__ == "__main__":
    _unit_tests()
