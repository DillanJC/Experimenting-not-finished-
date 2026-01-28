"""
Phase 3: Trajectory Features — For Generative Models

Requires logging embeddings at intermediate steps during generation.
Measures how the model moves through embedding space over time.

Features:
- drift_rate: Speed through embedding space
- trajectory_curvature: Jerkiness / sharp turns
- smoothness: Inverse jerkiness
- settling_behavior: Endpoint dispersion across stochastic runs
- attractor_proximity: Distance to stable regions (mean-shift modes)

Input Format:
    trajectories: List of (T, D) arrays, one per sample
    OR single (N, T, D) array if all trajectories same length

Logging Guidance (from handoff):
- Record pooled embeddings every Δt steps (Δt >= 1)
- Cap T <= 64 steps for memory
- Optionally apply online PCA to 95% variance
- Model-agnostic, adds O(T*D) memory per sample
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors


@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory."""
    drift_rates: np.ndarray      # (T-1,) speeds at each step
    curvatures: np.ndarray       # (T-2,) curvature at each step
    total_distance: float        # Sum of drift rates
    mean_drift: float
    median_drift: float
    max_drift: float
    p95_drift: float
    mean_curvature: float
    median_curvature: float
    smoothness: float            # 1 / (1 + median_curvature)
    endpoint: np.ndarray         # Final embedding


def compute_trajectory_stats(trajectory: np.ndarray, eps: float = 1e-12) -> TrajectoryStats:
    """
    Compute trajectory statistics for a single sample.

    Args:
        trajectory: (T, D) array of embeddings over time
        eps: numerical stability

    Returns:
        TrajectoryStats with all computed values
    """
    T, D = trajectory.shape

    if T < 2:
        # Degenerate case
        return TrajectoryStats(
            drift_rates=np.array([0.0]),
            curvatures=np.array([0.0]),
            total_distance=0.0,
            mean_drift=0.0,
            median_drift=0.0,
            max_drift=0.0,
            p95_drift=0.0,
            mean_curvature=0.0,
            median_curvature=0.0,
            smoothness=1.0,
            endpoint=trajectory[-1]
        )

    # Drift rates: v_t = ||x_{t+1} - x_t||
    diffs = np.diff(trajectory, axis=0)  # (T-1, D)
    drift_rates = np.linalg.norm(diffs, axis=1)  # (T-1,)

    # Curvature: κ_t = ||(x_{t+1}-x_t) - (x_t-x_{t-1})|| / (||x_{t+1}-x_t|| + eps)
    if T < 3:
        curvatures = np.array([0.0])
    else:
        # Second differences
        accel = np.diff(diffs, axis=0)  # (T-2, D)
        accel_norms = np.linalg.norm(accel, axis=1)  # (T-2,)
        # Normalize by forward velocity
        forward_speeds = drift_rates[1:] + eps  # (T-2,)
        curvatures = accel_norms / forward_speeds

    # Smoothness = 1 / (1 + median(curvature))
    median_curv = float(np.median(curvatures)) if len(curvatures) > 0 else 0.0
    smoothness = 1.0 / (1.0 + median_curv)

    return TrajectoryStats(
        drift_rates=drift_rates,
        curvatures=curvatures,
        total_distance=float(drift_rates.sum()),
        mean_drift=float(drift_rates.mean()),
        median_drift=float(np.median(drift_rates)),
        max_drift=float(drift_rates.max()),
        p95_drift=float(np.percentile(drift_rates, 95)),
        mean_curvature=float(curvatures.mean()) if len(curvatures) > 0 else 0.0,
        median_curvature=median_curv,
        smoothness=smoothness,
        endpoint=trajectory[-1]
    )


def compute_settling_behavior(
    endpoint_runs: np.ndarray,
    eps: float = 1e-12
) -> Dict[str, float]:
    """
    Compute settling behavior from multiple stochastic runs.

    Args:
        endpoint_runs: (R, D) array of final embeddings from R runs

    Returns:
        Dict with settling statistics
    """
    R, D = endpoint_runs.shape

    if R < 2:
        return {
            "mean_dispersion": 0.0,
            "median_dispersion": 0.0,
            "max_dispersion": 0.0,
            "centroid_distance_std": 0.0,
            "settled": True
        }

    # Centroid of endpoints
    centroid = endpoint_runs.mean(axis=0)

    # Distances from centroid
    distances = np.linalg.norm(endpoint_runs - centroid, axis=1)

    return {
        "mean_dispersion": float(distances.mean()),
        "median_dispersion": float(np.median(distances)),
        "max_dispersion": float(distances.max()),
        "centroid_distance_std": float(distances.std()),
        "settled": float(distances.std()) < 0.1  # Threshold for "settled"
    }


def compute_attractor_proximity(
    endpoints: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 50,
    n_modes: int = 10,
    bandwidth_multiplier: float = 1.0,
    max_iterations: int = 20,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance to nearest density mode (attractor) for each endpoint.

    Uses mean-shift to find modes, then measures distance to nearest.

    Args:
        endpoints: (N, D) final embeddings
        reference_embeddings: (N_ref, D) reference set for mode finding
        k: neighbors for mean-shift
        n_modes: max modes to track
        bandwidth_multiplier: mean-shift bandwidth scaling
        max_iterations: mean-shift iterations
        eps: numerical stability

    Returns:
        attractor_distances: (N,) distance to nearest mode
        nearest_mode_idx: (N,) index of nearest mode
    """
    N, D = endpoints.shape
    N_ref = len(reference_embeddings)

    # Find modes via mean-shift on reference set
    # Sample starting points
    n_starts = min(n_modes * 3, N_ref)
    start_idx = np.random.choice(N_ref, n_starts, replace=False)
    starts = reference_embeddings[start_idx].copy()

    # Build k-NN index
    nn = NearestNeighbors(n_neighbors=min(k, N_ref), algorithm="auto")
    nn.fit(reference_embeddings)

    # Run mean-shift from each start
    modes = []
    for start in starts:
        x = start.copy()
        for _ in range(max_iterations):
            dists, idx = nn.kneighbors([x])
            dists, idx = dists[0], idx[0]

            # Adaptive bandwidth
            h = bandwidth_multiplier * np.median(dists)
            h = max(h, 1e-6)

            # Gaussian weights
            w = np.exp(-dists**2 / (2 * h**2 + eps))

            # Mean-shift step
            neighbors = reference_embeddings[idx]
            w_sum = w.sum() + eps
            x_new = (w[:, np.newaxis] * neighbors).sum(axis=0) / w_sum

            # Check convergence
            if np.linalg.norm(x_new - x) < 1e-5:
                break
            x = x_new

        modes.append(x)

    modes = np.array(modes)

    # Deduplicate modes (merge if within threshold)
    unique_modes = [modes[0]]
    for mode in modes[1:]:
        dists_to_existing = [np.linalg.norm(mode - m) for m in unique_modes]
        if min(dists_to_existing) > 0.1:  # Threshold for distinct modes
            unique_modes.append(mode)
            if len(unique_modes) >= n_modes:
                break

    unique_modes = np.array(unique_modes)

    # Compute distance from each endpoint to nearest mode
    attractor_distances = np.zeros(N)
    nearest_mode_idx = np.zeros(N, dtype=np.int32)

    for i, endpoint in enumerate(endpoints):
        dists = np.linalg.norm(unique_modes - endpoint, axis=1)
        nearest_mode_idx[i] = np.argmin(dists)
        attractor_distances[i] = dists[nearest_mode_idx[i]]

    return attractor_distances, nearest_mode_idx


def compute_phase3_features(
    trajectories: Union[List[np.ndarray], np.ndarray],
    reference_embeddings: Optional[np.ndarray] = None,
    compute_attractors: bool = True,
    k_attractor: int = 50,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Phase 3 trajectory features.

    Args:
        trajectories: Either:
            - List of (T_i, D) arrays (variable length trajectories)
            - (N, T, D) array (fixed length trajectories)
        reference_embeddings: (N_ref, D) for attractor computation (optional)
        compute_attractors: whether to compute attractor proximity
        k_attractor: k for attractor mean-shift

    Returns:
        features: (N, 5 or 6) array
            [drift_mean, drift_p95, curvature_median, smoothness, total_distance]
            + [attractor_distance] if compute_attractors and reference provided
        metadata: statistics and settings

    Feature Semantics:
        - High drift_rate: fast movement (potentially unstable)
        - High curvature: jerky trajectory (searching/uncertain)
        - High smoothness: stable, confident generation
        - Low attractor_distance: near stable region
    """
    # Handle input format
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
        traj_list = [trajectories[i] for i in range(len(trajectories))]
    else:
        traj_list = trajectories

    N = len(traj_list)

    # Compute per-trajectory stats
    stats_list = [compute_trajectory_stats(t) for t in traj_list]

    # Extract features
    drift_mean = np.array([s.mean_drift for s in stats_list])
    drift_p95 = np.array([s.p95_drift for s in stats_list])
    curv_median = np.array([s.median_curvature for s in stats_list])
    smoothness = np.array([s.smoothness for s in stats_list])
    total_dist = np.array([s.total_distance for s in stats_list])

    features = np.stack([drift_mean, drift_p95, curv_median, smoothness, total_dist], axis=1)
    feature_names = ["drift_mean", "drift_p95", "curvature_median", "smoothness", "total_distance"]

    # Attractor proximity
    if compute_attractors and reference_embeddings is not None:
        endpoints = np.array([s.endpoint for s in stats_list])
        attractor_dist, _ = compute_attractor_proximity(
            endpoints, reference_embeddings, k=k_attractor
        )
        features = np.concatenate([features, attractor_dist.reshape(-1, 1)], axis=1)
        feature_names.append("attractor_distance")

    metadata = {
        "n_trajectories": N,
        "trajectory_lengths": [len(t) for t in traj_list],
        "feature_names": feature_names,
        "feature_means": features.mean(axis=0).tolist(),
        "feature_stds": features.std(axis=0).tolist(),
    }

    return features, metadata


PHASE3_FEATURE_NAMES = [
    "drift_mean",
    "drift_p95",
    "curvature_median",
    "smoothness",
    "total_distance",
    "attractor_distance",
]


# =============================================================================
# Trajectory Logging Helper
# =============================================================================

class TrajectoryLogger:
    """
    Helper class for logging embeddings during generation.

    Usage:
        logger = TrajectoryLogger(max_steps=64)

        for step in generation_loop:
            embedding = model.get_embedding(...)
            logger.log(sample_id, step, embedding)

        trajectories = logger.get_trajectories()
        features, meta = compute_phase3_features(trajectories, reference)
    """

    def __init__(self, max_steps: int = 64, dim: Optional[int] = None):
        self.max_steps = max_steps
        self.dim = dim
        self.data: Dict[int, List[np.ndarray]] = {}

    def log(self, sample_id: int, step: int, embedding: np.ndarray) -> None:
        """Log an embedding for a sample at a given step."""
        if sample_id not in self.data:
            self.data[sample_id] = []

        if len(self.data[sample_id]) < self.max_steps:
            self.data[sample_id].append(embedding.copy())

    def get_trajectories(self) -> List[np.ndarray]:
        """Get all logged trajectories as list of (T, D) arrays."""
        return [np.array(steps) for steps in self.data.values()]

    def clear(self) -> None:
        """Clear all logged data."""
        self.data = {}


if __name__ == "__main__":
    # Test with synthetic trajectories
    np.random.seed(42)

    N = 20  # samples
    T = 30  # timesteps
    D = 64  # embedding dim

    # Simulate trajectories: random walk with drift toward origin
    trajectories = []
    for _ in range(N):
        traj = np.zeros((T, D))
        traj[0] = np.random.randn(D)
        for t in range(1, T):
            # Drift toward origin + noise
            drift = -0.1 * traj[t-1] + 0.2 * np.random.randn(D)
            traj[t] = traj[t-1] + drift
        trajectories.append(traj)

    # Reference set
    reference = np.random.randn(200, D)

    print("Phase 3 Trajectory Features Test")
    print("="*50)

    features, meta = compute_phase3_features(
        trajectories, reference, compute_attractors=True
    )

    print(f"Shape: {features.shape}")
    print(f"Features: {meta['feature_names']}")
    print("\nFeature Statistics:")
    for i, name in enumerate(meta['feature_names']):
        print(f"  {name:20s}: mean={features[:, i].mean():.4f}, std={features[:, i].std():.4f}")

    # Test settling behavior
    print("\nSettling Behavior Test:")
    endpoints = np.array([t[-1] for t in trajectories[:5]])
    settling = compute_settling_behavior(endpoints)
    print(f"  Mean dispersion: {settling['mean_dispersion']:.4f}")
    print(f"  Settled: {settling['settled']}")

    print("\n[OK] Phase 3 test passed!")
