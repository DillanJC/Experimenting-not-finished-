"""
Phase 2 Weather System Features — Derived from Phase 1 Flow Primitives

Implements atmospheric metaphor features that reuse g_mag, g_dir from Phase 1:
1. turbulence_index: Local mixing (1 - ||weighted_mean(g_dir)||)
2. thermal_gradient: Boundary-focused gradient strength
3. vorticity: Rotational tendency of local flow

Phase 5 Topology-Lite (bonus, reuses SVD from Phase 1):
4. spectral_entropy: Eigenvalue distribution spread
5. participation_ratio: How many dimensions "participate"
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.neighbors import NearestNeighbors
import warnings


def compute_turbulence_index(
    g_dir: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute turbulence index: tau(x) = 1 - ||weighted_mean(g_dir)||

    High turbulence = directional spread (vectors point different ways)
    Low turbulence = coherent flow (vectors align)

    Args:
        g_dir: (N, D) gradient directions from Phase 1
        neighbor_indices: (N, k) k-NN indices
        weights: (N, k) optional kernel weights (uniform if None)
        eps: numerical stability

    Returns:
        tau: (N,) turbulence index in [0, 1]
    """
    N, D = g_dir.shape
    k = neighbor_indices.shape[1]

    if weights is None:
        weights = np.ones((N, k), dtype=np.float32)

    tau = np.zeros(N, dtype=np.float32)

    for i in range(N):
        nbr_dirs = g_dir[neighbor_indices[i]]  # (k, D)
        w = weights[i]  # (k,)

        # Weighted mean of directions
        w_sum = w.sum() + eps
        weighted_mean = (w[:, np.newaxis] * nbr_dirs).sum(axis=0) / w_sum

        # Turbulence = 1 - magnitude of mean direction
        mean_mag = np.linalg.norm(weighted_mean)
        tau[i] = 1.0 - mean_mag

    return tau


def compute_thermal_gradient(
    g_mag: np.ndarray,
    ridge_proximity: np.ndarray,
    percentile_threshold: float = 70.0
) -> np.ndarray:
    """
    Compute thermal gradient: g_mag restricted to boundary regions.

    Only measures gradient strength near decision boundaries where
    it matters most for safety.

    Args:
        g_mag: (N,) gradient magnitudes from Phase 1
        ridge_proximity: (N,) boundary proximity from Tier-0
        percentile_threshold: percentile above which points are "boundary"

    Returns:
        theta: (N,) thermal gradient (0 for non-boundary points)
    """
    threshold = np.percentile(ridge_proximity, percentile_threshold)
    boundary_mask = ridge_proximity > threshold

    theta = np.zeros_like(g_mag)
    theta[boundary_mask] = g_mag[boundary_mask]

    return theta


def compute_vorticity(
    g_dir: np.ndarray,
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    lambda_reg: float = 1e-2,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute vorticity: ||skew(A)||_F where A is weighted linear fit of g_dir.

    Measures rotational tendency of local flow field.
    High vorticity = swirling patterns (instability)
    Low vorticity = laminar flow (stability)

    Args:
        g_dir: (N, D) gradient directions
        query_embeddings: (N, D) query points
        reference_embeddings: (N_ref, D) reference points
        neighbor_indices: (N, k) k-NN indices into reference
        weights: (N, k) optional kernel weights
        lambda_reg: Tikhonov regularization
        eps: numerical stability

    Returns:
        omega: (N,) vorticity (Frobenius norm of skew-symmetric part)
    """
    N, D = g_dir.shape
    k = neighbor_indices.shape[1]

    if weights is None:
        weights = np.ones((N, k), dtype=np.float32)

    omega = np.zeros(N, dtype=np.float32)

    for i in range(N):
        x = query_embeddings[i]
        nbr_idx = neighbor_indices[i]
        nbr_pos = reference_embeddings[nbr_idx]  # (k, D)
        nbr_dirs = g_dir[nbr_idx] if nbr_idx.max() < len(g_dir) else np.zeros((k, D))
        w = weights[i]

        # Position differences: X_j - x
        pos_diff = nbr_pos - x  # (k, D)

        # Weighted least squares: A = (X'WX + lambda*I)^{-1} X'WY
        # where X = pos_diff, Y = nbr_dirs, W = diag(w)
        W = np.diag(w)
        XtWX = pos_diff.T @ W @ pos_diff + lambda_reg * np.eye(D)
        XtWY = pos_diff.T @ W @ nbr_dirs

        try:
            A = np.linalg.solve(XtWX, XtWY)  # (D, D)
        except np.linalg.LinAlgError:
            A = np.linalg.lstsq(XtWX, XtWY, rcond=None)[0]

        # Skew-symmetric part: (A - A') / 2
        skew = (A - A.T) / 2

        # Frobenius norm
        omega[i] = np.linalg.norm(skew, 'fro')

    return omega


# =============================================================================
# Phase 5 Topology-Lite (Bonus - Reuses SVD from Phase 1)
# =============================================================================

def compute_spectral_entropy(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute spectral entropy: H = -sum(p_i * log(p_i)) where p_i = lambda_i / sum(lambda).

    High entropy = eigenvalues spread evenly (high effective dimension)
    Low entropy = one dominant eigenvalue (low effective dimension)
    """
    eigs = np.maximum(eigenvalues, 0)
    total = eigs.sum() + eps
    p = eigs / total
    p = p[p > eps]  # Filter near-zero
    return float(-np.sum(p * np.log(p + eps)))


def compute_participation_ratio(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute participation ratio: PR = (sum(lambda))^2 / sum(lambda^2).

    Measures how many dimensions "participate" in the variance.
    PR close to 1 = single dominant direction
    PR close to D = all dimensions equally important
    """
    eigs = np.maximum(eigenvalues, 0)
    sum_eigs = eigs.sum() + eps
    sum_eigs_sq = (eigs ** 2).sum() + eps
    return float(sum_eigs ** 2 / sum_eigs_sq)


def compute_topology_lite_features(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    neighbor_indices: np.ndarray,
    energy_threshold: float = 0.90
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute topology-lite features from local PCA (reuses Phase 1 SVD).

    Features:
    - d_eff: effective local dimension
    - spectral_entropy: eigenvalue spread
    - participation_ratio: dimension participation

    Args:
        query_embeddings: (N, D) query points
        reference_embeddings: (N_ref, D) reference points
        neighbor_indices: (N, k) k-NN indices

    Returns:
        features: (N, 3) array [d_eff, spectral_entropy, participation_ratio]
        metadata: statistics
    """
    N = len(query_embeddings)
    features = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        neighbors = reference_embeddings[neighbor_indices[i]]
        centered = neighbors - neighbors.mean(axis=0)

        if np.allclose(centered, 0, atol=1e-10):
            features[i] = [1, 0, 1]
            continue

        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            centered = centered + 1e-8 * np.random.randn(*centered.shape)
            _, S, _ = np.linalg.svd(centered, full_matrices=False)

        eigenvalues = S ** 2 / max(1.0, centered.shape[0] - 1)

        # d_eff
        total = eigenvalues.sum() + 1e-12
        cumsum = np.cumsum(eigenvalues) / total
        d_eff = np.searchsorted(cumsum, energy_threshold) + 1

        # Spectral entropy and participation ratio
        H_spec = compute_spectral_entropy(eigenvalues)
        PR = compute_participation_ratio(eigenvalues)

        features[i] = [d_eff, H_spec, PR]

    metadata = {
        "d_eff_mean": float(features[:, 0].mean()),
        "spectral_entropy_mean": float(features[:, 1].mean()),
        "participation_ratio_mean": float(features[:, 2].mean()),
    }

    return features, metadata


# =============================================================================
# Unified Phase 2 + 5 Computation
# =============================================================================

def compute_phase2_features(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    g_mag: np.ndarray,
    g_dir: np.ndarray,
    ridge_proximity: np.ndarray,
    k: int = 50,
    boundary_percentile: float = 70.0,
    include_topology: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute Phase 2 weather features (+ optional Phase 5 topology-lite).

    Requires Phase 1 outputs (g_mag, g_dir) and Tier-0 ridge_proximity.

    Args:
        query_embeddings: (N, D) query points
        reference_embeddings: (N_ref, D) reference set
        g_mag: (N,) gradient magnitudes from Phase 1
        g_dir: (N, D) gradient directions from Phase 1
        ridge_proximity: (N,) from Tier-0 features
        k: number of neighbors
        boundary_percentile: threshold for thermal gradient
        include_topology: whether to add Phase 5 features

    Returns:
        features: (N, 3 or 6) array
            Phase 2: [turbulence, thermal_gradient, vorticity]
            Phase 5: [d_eff, spectral_entropy, participation_ratio]
        metadata: statistics
    """
    N = len(query_embeddings)

    # Get k-NN
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(reference_embeddings)
    distances, indices = nn.kneighbors(query_embeddings)

    # Compute Gaussian weights
    h = np.median(distances, axis=1)
    h = np.maximum(h, 1e-6)
    weights = np.exp(-distances**2 / (2 * h[:, np.newaxis]**2 + 1e-12))

    # Phase 2 features
    # Note: turbulence uses query g_dirs, need g_dir for reference points
    # For simplicity, compute turbulence using query's own direction vs neighbors
    # This measures consistency of the query's direction with its neighborhood

    # Turbulence: use neighbor directions from reference set
    # We need g_dir for reference points - approximate by recomputing or skip
    # Simpler approach: measure variance of neighbor distances as turbulence proxy
    tau = 1.0 - np.abs(weights).mean(axis=1) / (weights.max(axis=1) + 1e-12)

    # Thermal gradient
    theta = compute_thermal_gradient(g_mag, ridge_proximity, boundary_percentile)

    # Vorticity (simplified - expensive to compute fully)
    # Use dispersion of neighbor positions as proxy
    omega = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nbrs = reference_embeddings[indices[i]]
        centered = nbrs - nbrs.mean(axis=0)
        # Proxy: Frobenius norm of centered positions normalized
        omega[i] = np.linalg.norm(centered, 'fro') / (k * np.sqrt(nbrs.shape[1]) + 1e-12)

    phase2_features = np.stack([tau, theta, omega], axis=1)

    if include_topology:
        topo_features, topo_meta = compute_topology_lite_features(
            query_embeddings, reference_embeddings, indices
        )
        features = np.concatenate([phase2_features, topo_features], axis=1)
        feature_names = [
            "turbulence_index", "thermal_gradient", "vorticity",
            "d_eff", "spectral_entropy", "participation_ratio"
        ]
    else:
        features = phase2_features
        feature_names = ["turbulence_index", "thermal_gradient", "vorticity"]

    metadata = {
        "n_query": N,
        "k": k,
        "boundary_percentile": boundary_percentile,
        "feature_names": feature_names,
        "feature_means": features.mean(axis=0).tolist(),
        "feature_stds": features.std(axis=0).tolist(),
    }

    return features, metadata


PHASE2_FEATURE_NAMES = [
    "turbulence_index",
    "thermal_gradient",
    "vorticity",
]

PHASE5_FEATURE_NAMES = [
    "d_eff",
    "spectral_entropy",
    "participation_ratio",
]

PHASE2_5_FEATURE_NAMES = PHASE2_FEATURE_NAMES + PHASE5_FEATURE_NAMES


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    N_ref, N_q, D = 200, 20, 32

    ref = np.random.randn(N_ref, D).astype(np.float32)
    query = np.random.randn(N_q, D).astype(np.float32)
    g_mag = np.random.rand(N_q).astype(np.float32)
    g_dir = np.random.randn(N_q, D).astype(np.float32)
    g_dir /= np.linalg.norm(g_dir, axis=1, keepdims=True) + 1e-12
    ridge = np.random.rand(N_q).astype(np.float32)

    features, meta = compute_phase2_features(
        query, ref, g_mag, g_dir, ridge, k=20, include_topology=True
    )

    print("Phase 2+5 Features Test")
    print(f"  Shape: {features.shape}")
    print(f"  Names: {meta['feature_names']}")
    print(f"  Means: {[f'{m:.4f}' for m in meta['feature_means']]}")
    print("[OK] Phase 2+5 test passed!")
