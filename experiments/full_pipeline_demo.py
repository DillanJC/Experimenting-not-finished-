"""
Full Pipeline Demo — Phases 0-5 End-to-End

Demonstrates the complete geometric safety diagnostic pipeline.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import (
    GeometryBundle,
    compute_gradient_magnitude_only,
    compute_phase2_features,
    compute_state_scores,
    get_state_summary,
    STATE_NAMES,
    STATE_DESCRIPTIONS,
    FEATURE_NAMES,
)

def main():
    print("="*70)
    print("GEOMETRIC SAFETY FEATURES — FULL PIPELINE DEMO")
    print("="*70)

    # Load data
    base = Path(__file__).parent.parent
    embeddings = np.load(base / "embeddings.npy")
    boundary_distances = np.load(base / "boundary_distances.npy")

    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    y = boundary_distances[split:]

    print(f"\nData: {len(queries)} queries, {len(reference)} reference, D={embeddings.shape[1]}")

    # =========================================================================
    # PHASE 0: Tier-0 k-NN Features
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 0: Tier-0 k-NN Features")
    print("-"*70)

    bundle = GeometryBundle(reference, k=50)
    tier0_results = bundle.compute(queries)
    tier0_features = bundle.get_feature_matrix(tier0_results)

    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Shape: {tier0_features.shape}")

    # =========================================================================
    # PHASE 1: Flow Primitives
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 1: Flow Primitives (gradient magnitude)")
    print("-"*70)

    g_mag, p1_meta = compute_gradient_magnitude_only(queries, reference, k=75)
    print(f"  g_mag mean: {g_mag.mean():.4f}, std: {g_mag.std():.4f}")

    # =========================================================================
    # PHASE 2+5: Weather + Topology Features
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 2+5: Weather + Topology Features")
    print("-"*70)

    ridge_proximity = tier0_features[:, 5]
    g_dir_placeholder = np.zeros((len(queries), queries.shape[1]))

    phase2_5, p2_meta = compute_phase2_features(
        queries, reference, g_mag.flatten(), g_dir_placeholder,
        ridge_proximity, k=50, include_topology=True
    )

    print(f"  Features: {p2_meta['feature_names']}")
    for i, name in enumerate(p2_meta['feature_names']):
        print(f"    {name}: mean={phase2_5[:, i].mean():.4f}")

    # =========================================================================
    # PHASE 4: State Mapping
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 4: AI State Mapping")
    print("-"*70)

    feature_dict = {
        'g_mag': g_mag.flatten(),
        'consistency': np.ones(len(queries)) * 0.5,  # Placeholder
        'turbulence': phase2_5[:, 0],
        'delta_rho': np.zeros(len(queries)),  # Not computed in fast mode
        'knn_std_distance': tier0_features[:, 1],
        'knn_max_distance': tier0_features[:, 3],
        'ridge_proximity': ridge_proximity,
    }

    state_labels, state_scores, state_names = compute_state_scores(feature_dict)
    summary = get_state_summary(state_labels, state_scores, state_names)

    print("\n  State Distribution:")
    for name in state_names:
        pct = summary["state_percentages"][name]
        desc = STATE_DESCRIPTIONS.get(name, "")
        print(f"    {name:20s}: {pct:5.1f}%  ({desc})")

    # =========================================================================
    # ANALYSIS: States vs Boundary Distance
    # =========================================================================
    print("\n" + "-"*70)
    print("ANALYSIS: Mean Boundary Distance by State")
    print("-"*70)

    for i, name in enumerate(state_names):
        mask = state_labels == i
        if mask.sum() > 0:
            mean_bd = y[mask].mean()
            print(f"    {name:20s}: boundary_dist={mean_bd:+.3f} (n={mask.sum()})")

    # Highlight uncertain/searching vs confident/coherent
    print("\n  Key Finding:")
    uncertain_mask = (state_labels == state_names.index('uncertain')) | (state_labels == state_names.index('searching'))
    confident_mask = (state_labels == state_names.index('confident')) | (state_labels == state_names.index('coherent'))

    if uncertain_mask.sum() > 0 and confident_mask.sum() > 0:
        uncertain_bd = y[uncertain_mask].mean()
        confident_bd = y[confident_mask].mean()
        print(f"    Uncertain/Searching: boundary_dist={uncertain_bd:+.3f}")
        print(f"    Confident/Coherent:  boundary_dist={confident_bd:+.3f}")
        print(f"    Separation: {abs(confident_bd - uncertain_bd):.3f}")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
