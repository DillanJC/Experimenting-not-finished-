"""
Phase 4: AI State Measurement — Decision Rules

Maps geometric signatures to interpretable cognitive states.
States are not mutually exclusive - output probability scores for each.

States:
- Coherent: Consistent flow toward attractor
- Searching: Inconsistent, turbulent flow
- Constraint Pressure: Multi-basin tension
- Novel Territory: Sparse, flat region
- Confident: Tight clustering, strong pull
- Uncertain: Dispersed, boundary-adjacent
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StateThresholds:
    """Calibrated thresholds for state detection (percentile-based)."""
    consistency_high: float = 70.0  # Top 30%
    consistency_low: float = 30.0   # Bottom 30%
    turbulence_high: float = 70.0
    turbulence_low: float = 30.0
    g_mag_high: float = 70.0
    g_mag_low: float = 30.0
    knn_std_high: float = 70.0
    knn_std_low: float = 30.0
    knn_max_high: float = 80.0      # Top 20% for novel territory
    delta_rho_high: float = 70.0
    ridge_proximity_high: float = 70.0


def compute_state_scores(
    features: Dict[str, np.ndarray],
    thresholds: Optional[StateThresholds] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute AI state probability scores from geometric features.

    Args:
        features: Dictionary with keys:
            - 'g_mag': gradient magnitude (Phase 1)
            - 'consistency': direction consistency (Phase 1, optional)
            - 'turbulence': turbulence index (Phase 2)
            - 'delta_rho': pressure differential (Phase 1)
            - 'knn_std_distance': from Tier-0
            - 'knn_max_distance': from Tier-0
            - 'ridge_proximity': from Tier-0
        thresholds: Calibrated thresholds (uses defaults if None)

    Returns:
        state_labels: (N,) argmax state index
        state_scores: Dict mapping state name to (N,) probability scores
    """
    th = thresholds or StateThresholds()
    N = len(features['g_mag'])

    # Extract features with defaults
    g_mag = features.get('g_mag', np.zeros(N))
    consistency = features.get('consistency', np.ones(N) * 0.5)
    turbulence = features.get('turbulence', np.zeros(N))
    delta_rho = features.get('delta_rho', np.zeros(N))
    knn_std = features.get('knn_std_distance', np.ones(N))
    knn_max = features.get('knn_max_distance', np.ones(N))
    ridge_prox = features.get('ridge_proximity', np.zeros(N))

    # Compute percentile thresholds
    def pct(arr, p): return np.percentile(arr, p)

    scores = {}

    # =========================================================================
    # COHERENT: Consistent flow toward attractor
    # consistency top 30% AND turbulence bottom 30%, optionally g_mag > median
    # =========================================================================
    coherent_score = (
        (consistency > pct(consistency, th.consistency_high)).astype(float) * 0.4 +
        (turbulence < pct(turbulence, th.turbulence_low)).astype(float) * 0.4 +
        (g_mag > np.median(g_mag)).astype(float) * 0.2
    )
    scores['coherent'] = coherent_score

    # =========================================================================
    # SEARCHING: Inconsistent, turbulent flow
    # consistency bottom 30% OR turbulence top 30%
    # =========================================================================
    searching_score = np.maximum(
        (consistency < pct(consistency, th.consistency_low)).astype(float),
        (turbulence > pct(turbulence, th.turbulence_high)).astype(float)
    ) * 0.7 + (g_mag < np.median(g_mag)).astype(float) * 0.3
    scores['searching'] = searching_score

    # =========================================================================
    # CONSTRAINT_PRESSURE: Multi-basin tension
    # delta_rho top 30% AND moderate ridge_proximity
    # =========================================================================
    constraint_score = (
        (delta_rho > pct(delta_rho, th.delta_rho_high)).astype(float) * 0.5 +
        (ridge_prox > pct(ridge_prox, 40)).astype(float) *
        (ridge_prox < pct(ridge_prox, 80)).astype(float) * 0.5
    )
    scores['constraint_pressure'] = constraint_score

    # =========================================================================
    # NOVEL_TERRITORY: Sparse, flat region
    # knn_max top 20% WITH delta_rho <= 0 AND low g_mag
    # =========================================================================
    novel_score = (
        (knn_max > pct(knn_max, th.knn_max_high)).astype(float) * 0.4 +
        (delta_rho <= 0).astype(float) * 0.3 +
        (g_mag < pct(g_mag, th.g_mag_low)).astype(float) * 0.3
    )
    scores['novel_territory'] = novel_score

    # =========================================================================
    # CONFIDENT: Tight clustering, strong pull
    # knn_std bottom 30%, g_mag top 30%, delta_rho > 0
    # =========================================================================
    confident_score = (
        (knn_std < pct(knn_std, th.knn_std_low)).astype(float) * 0.35 +
        (g_mag > pct(g_mag, th.g_mag_high)).astype(float) * 0.35 +
        (delta_rho > 0).astype(float) * 0.3
    )
    scores['confident'] = confident_score

    # =========================================================================
    # UNCERTAIN: Dispersed, boundary-adjacent
    # knn_std top 30% OR high ridge_proximity WITH low consistency
    # =========================================================================
    uncertain_score = (
        np.maximum(
            (knn_std > pct(knn_std, th.knn_std_high)).astype(float),
            (ridge_prox > pct(ridge_prox, th.ridge_proximity_high)).astype(float)
        ) * 0.6 +
        (consistency < np.median(consistency)).astype(float) * 0.4
    )
    scores['uncertain'] = uncertain_score

    # Normalize scores to sum to 1 (soft assignment)
    score_matrix = np.stack([scores[k] for k in sorted(scores.keys())], axis=1)
    score_sums = score_matrix.sum(axis=1, keepdims=True) + 1e-12
    normalized = score_matrix / score_sums

    # Update scores dict with normalized values
    for i, key in enumerate(sorted(scores.keys())):
        scores[key] = normalized[:, i]

    # Argmax labels
    state_names = sorted(scores.keys())
    state_labels = np.argmax(normalized, axis=1)

    return state_labels, scores, state_names


def get_state_summary(
    state_labels: np.ndarray,
    state_scores: Dict[str, np.ndarray],
    state_names: list
) -> Dict:
    """Generate summary statistics for state distribution."""
    N = len(state_labels)

    summary = {
        "total_samples": N,
        "state_counts": {},
        "state_percentages": {},
        "mean_scores": {},
    }

    for i, name in enumerate(state_names):
        count = (state_labels == i).sum()
        summary["state_counts"][name] = int(count)
        summary["state_percentages"][name] = float(100 * count / N)
        summary["mean_scores"][name] = float(state_scores[name].mean())

    return summary


STATE_NAMES = [
    "coherent",
    "confident",
    "constraint_pressure",
    "novel_territory",
    "searching",
    "uncertain",
]

STATE_DESCRIPTIONS = {
    "coherent": "Consistent flow toward attractor - stable cognition",
    "confident": "Tight clustering with strong pull - high certainty",
    "constraint_pressure": "Multi-basin tension - competing interpretations",
    "novel_territory": "Sparse, flat region - unfamiliar territory",
    "searching": "Inconsistent, turbulent flow - exploration mode",
    "uncertain": "Dispersed, boundary-adjacent - low confidence",
}


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    N = 100

    features = {
        'g_mag': np.random.rand(N),
        'consistency': np.random.rand(N),
        'turbulence': np.random.rand(N),
        'delta_rho': np.random.randn(N) * 0.5,
        'knn_std_distance': np.random.rand(N),
        'knn_max_distance': np.random.rand(N),
        'ridge_proximity': np.random.rand(N),
    }

    labels, scores, names = compute_state_scores(features)
    summary = get_state_summary(labels, scores, names)

    print("Phase 4 State Mapping Test")
    print("="*50)
    for name in names:
        pct = summary["state_percentages"][name]
        print(f"  {name:20s}: {pct:5.1f}%")
    print("\n[OK] Phase 4 test passed!")
