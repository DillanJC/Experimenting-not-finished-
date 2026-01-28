"""
Unified Geometric Safety Pipeline — All Phases Combined

Single entry point for computing all geometric safety features
and state mappings from embeddings.
"""

import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field

from .bundle import GeometryBundle
from .phase1_flow_features import compute_gradient_magnitude_only, compute_phase1_features_fast
from .phase2_weather_features import compute_phase2_features, PHASE2_5_FEATURE_NAMES
from .phase4_state_mapping import compute_state_scores, get_state_summary, STATE_NAMES


@dataclass
class SafetyDiagnostics:
    """Complete geometric safety diagnostics for a set of queries."""

    # Raw features
    tier0_features: np.ndarray          # (N, 7) Tier-0 k-NN features
    phase1_features: np.ndarray         # (N, 1-3) Flow features
    phase2_5_features: np.ndarray       # (N, 6) Weather + topology features

    # State mapping
    state_labels: np.ndarray            # (N,) Argmax state index
    state_scores: Dict[str, np.ndarray] # State name -> (N,) probabilities
    state_names: list                   # Ordered state names

    # Metadata
    n_samples: int
    feature_names: Dict[str, list]
    summary: Dict

    def get_high_risk_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Get mask for samples with high uncertainty/novel_territory scores."""
        uncertain = self.state_scores.get('uncertain', np.zeros(self.n_samples))
        novel = self.state_scores.get('novel_territory', np.zeros(self.n_samples))
        return (uncertain > threshold) | (novel > threshold)

    def get_all_features(self) -> np.ndarray:
        """Concatenate all features into single matrix."""
        return np.concatenate([
            self.tier0_features,
            self.phase1_features,
            self.phase2_5_features
        ], axis=1)

    def get_feature_names(self) -> list:
        """Get ordered list of all feature names."""
        return (
            self.feature_names['tier0'] +
            self.feature_names['phase1'] +
            self.feature_names['phase2_5']
        )


def compute_safety_diagnostics(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 50,
    k_phase1: int = 75,
    bandwidth_multiplier: float = 1.2,
    include_full_phase1: bool = False,
    return_intermediate: bool = False,
) -> SafetyDiagnostics:
    """
    Compute complete geometric safety diagnostics.

    This is the main entry point for the geometric safety pipeline.
    Computes all features from Phases 0, 1, 2, 4, and 5.

    Args:
        query_embeddings: (N, D) embeddings to analyze
        reference_embeddings: (N_ref, D) reference/training embeddings
        k: k for Tier-0 and Phase 2 features
        k_phase1: k for Phase 1 gradient features (higher is better)
        bandwidth_multiplier: bandwidth scaling for Phase 1
        include_full_phase1: if True, compute consistency (slower)
        return_intermediate: if True, include intermediate computations

    Returns:
        SafetyDiagnostics object with all features and state mappings

    Example:
        >>> from mirrorfield.geometry import compute_safety_diagnostics
        >>> diag = compute_safety_diagnostics(queries, reference)
        >>> high_risk = diag.get_high_risk_mask(threshold=0.3)
        >>> print(f"High risk samples: {high_risk.sum()}/{len(queries)}")
    """
    N = len(query_embeddings)

    # =========================================================================
    # Phase 0: Tier-0 k-NN Features
    # =========================================================================
    bundle = GeometryBundle(reference_embeddings, k=k)
    tier0_results = bundle.compute(query_embeddings)
    tier0_features = bundle.get_feature_matrix(tier0_results)

    tier0_names = [
        'knn_mean_distance', 'knn_std_distance', 'knn_min_distance',
        'knn_max_distance', 'local_curvature', 'ridge_proximity',
        'dist_to_ref_nearest'
    ]

    # =========================================================================
    # Phase 1: Flow Features
    # =========================================================================
    if include_full_phase1:
        phase1_features, p1_meta = compute_phase1_features_fast(
            query_embeddings, reference_embeddings,
            k=k_phase1, bandwidth_multiplier=bandwidth_multiplier
        )
        phase1_names = ['local_gradient_magnitude', 'pressure_differential']
    else:
        phase1_features, p1_meta = compute_gradient_magnitude_only(
            query_embeddings, reference_embeddings,
            k=k_phase1, bandwidth_multiplier=bandwidth_multiplier
        )
        phase1_names = ['local_gradient_magnitude']

    # =========================================================================
    # Phase 2+5: Weather + Topology Features
    # =========================================================================
    ridge_proximity = tier0_features[:, 5]
    g_mag = phase1_features[:, 0] if phase1_features.ndim > 1 else phase1_features.flatten()
    g_dir_placeholder = np.zeros((N, query_embeddings.shape[1]))

    phase2_5_features, p2_meta = compute_phase2_features(
        query_embeddings, reference_embeddings,
        g_mag, g_dir_placeholder, ridge_proximity,
        k=k, include_topology=True
    )

    # =========================================================================
    # Phase 4: State Mapping
    # =========================================================================
    feature_dict = {
        'g_mag': g_mag,
        'consistency': np.ones(N) * 0.5,  # Placeholder if not computed
        'turbulence': phase2_5_features[:, 0],
        'delta_rho': phase1_features[:, 1] if phase1_features.shape[1] > 1 else np.zeros(N),
        'knn_std_distance': tier0_features[:, 1],
        'knn_max_distance': tier0_features[:, 3],
        'ridge_proximity': ridge_proximity,
    }

    state_labels, state_scores, state_names = compute_state_scores(feature_dict)
    summary = get_state_summary(state_labels, state_scores, state_names)

    # =========================================================================
    # Build diagnostics object
    # =========================================================================
    return SafetyDiagnostics(
        tier0_features=tier0_features,
        phase1_features=phase1_features,
        phase2_5_features=phase2_5_features,
        state_labels=state_labels,
        state_scores=state_scores,
        state_names=state_names,
        n_samples=N,
        feature_names={
            'tier0': tier0_names,
            'phase1': phase1_names,
            'phase2_5': PHASE2_5_FEATURE_NAMES,
        },
        summary=summary,
    )


def print_diagnostics_summary(diag: SafetyDiagnostics) -> None:
    """Print a human-readable summary of safety diagnostics."""
    print("=" * 60)
    print("GEOMETRIC SAFETY DIAGNOSTICS SUMMARY")
    print("=" * 60)

    print(f"\nSamples analyzed: {diag.n_samples}")
    print(f"Total features: {len(diag.get_feature_names())}")

    print("\nState Distribution:")
    for name in diag.state_names:
        pct = diag.summary["state_percentages"][name]
        score = diag.summary["mean_scores"][name]
        print(f"  {name:20s}: {pct:5.1f}% (mean score: {score:.3f})")

    high_risk = diag.get_high_risk_mask(threshold=0.25)
    print(f"\nHigh-risk samples (uncertain/novel > 0.25): {high_risk.sum()} ({100*high_risk.mean():.1f}%)")

    print("\nFeature Summary:")
    all_features = diag.get_all_features()
    all_names = diag.get_feature_names()
    for i, name in enumerate(all_names):
        print(f"  {name:30s}: mean={all_features[:, i].mean():+.4f}, std={all_features[:, i].std():.4f}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Fix imports when running as script
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from mirrorfield.geometry import compute_safety_diagnostics
    from mirrorfield.geometry.unified_pipeline import print_diagnostics_summary

    # Load test data
    base = Path(__file__).parent.parent.parent
    embeddings = np.load(base / "embeddings.npy")
    boundary_distances = np.load(base / "boundary_distances.npy")

    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]

    print("Computing safety diagnostics...")
    diag = compute_safety_diagnostics(queries, reference)

    print_diagnostics_summary(diag)

    # Validate against boundary distance
    print("\n" + "=" * 60)
    print("VALIDATION: State vs Boundary Distance")
    print("=" * 60)
    for i, name in enumerate(diag.state_names):
        mask = diag.state_labels == i
        if mask.sum() > 0:
            bd = boundary_distances[split:][mask].mean()
            print(f"  {name:20s}: boundary_dist={bd:+.3f}")

    print("\n[OK] Unified pipeline test complete!")
