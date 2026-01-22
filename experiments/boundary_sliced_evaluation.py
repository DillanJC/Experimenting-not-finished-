"""
Boundary-Sliced Evaluation — Where Does Geometry Help Most?

Tests baseline vs geometry across three zones:
1. SAFE zone (boundary_distance > 0.5): Model confident & correct
2. BORDERLINE zone (|boundary_distance| < 0.5): High uncertainty
3. UNSAFE zone (boundary_distance < -0.5): Model confident & wrong

Hypothesis: Geometry features help most in borderline zone where
baseline methods struggle.

Methodology:
- 20 independent trials per zone
- Ridge regression (α=1.0)
- Compare R² improvement: geometry vs baseline
- Report where gains concentrate
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp


def load_data():
    """Load sentiment classification data."""
    # Try multiple locations for data
    possible_paths = [
        Path(__file__).parent.parent,  # Current project dir
        Path(__file__).parent.parent / "runs" / "openai_3_large_test_20251231_024532",
        Path("runs/openai_3_large_test_20251231_024532"),
        Path(
            "C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532"
        ),  # Fallback
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Data not found. Tried:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
            + f"\n\nPlease ensure data files are in the runs/ directory."
        )

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    return embeddings, boundary_distances


def compute_geometry_features(embeddings, boundary_distances):
    """Compute geometry features for query set."""
    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    # Compute geometry
    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries)

    # Extract feature matrix
    features = bundle.get_feature_matrix(results)

    return features, query_boundaries, queries


def define_zones(boundary_distances, safe_threshold=0.5, unsafe_threshold=-0.5):
    """
    Split data into three zones based on boundary distance.

    Args:
        boundary_distances: Array of boundary distances
        safe_threshold: Minimum distance for safe zone (default: 0.5)
        unsafe_threshold: Maximum distance for unsafe zone (default: -0.5)

    Returns:
        Dictionary with masks for each zone
    """
    safe_mask = boundary_distances > safe_threshold
    borderline_mask = (boundary_distances >= unsafe_threshold) & (
        boundary_distances <= safe_threshold
    )
    unsafe_mask = boundary_distances < unsafe_threshold

    return {"safe": safe_mask, "borderline": borderline_mask, "unsafe": unsafe_mask}


def evaluate_zone(X_baseline, X_geometry, y, zone_name, n_trials=20):
    """
    Evaluate baseline vs geometry on a specific zone.

    Args:
        X_baseline: Baseline features (embeddings only)
        X_geometry: Geometry features (embeddings + 7 k-NN features)
        y: Target (boundary distances)
        zone_name: Name of zone for logging
        n_trials: Number of independent trials

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {zone_name.upper()} ZONE")
    print(f"{'=' * 80}\n")

    print(f"Zone size: {len(y)} samples")
    print(f"Boundary distance range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Boundary distance mean: {y.mean():.3f} ± {y.std():.3f}")
    print()

    if len(y) < 10:
        print(f"WARNING:  WARNING: Zone has only {len(y)} samples (< 10)")
        print(f"   Results may be unreliable due to small sample size")
        print()

    results = []

    print(f"Running {n_trials} trials...")
    for i in range(n_trials):
        seed = 42 + i

        # Split data: 80% train, 20% test (per zone)
        X_base_train, X_base_test, X_geom_train, X_geom_test, y_train, y_test = (
            train_test_split(
                X_baseline, X_geometry, y, test_size=0.2, random_state=seed
            )
        )

        # Baseline: embeddings only
        ridge_baseline = Ridge(alpha=1.0, random_state=seed)
        ridge_baseline.fit(X_base_train, y_train)
        pred_baseline = ridge_baseline.predict(X_base_test)
        r2_baseline = r2_score(y_test, pred_baseline)
        mae_baseline = mean_absolute_error(y_test, pred_baseline)

        # Geometry: embeddings + 7 k-NN features
        ridge_geometry = Ridge(alpha=1.0, random_state=seed)
        ridge_geometry.fit(X_geom_train, y_train)
        pred_geometry = ridge_geometry.predict(X_geom_test)
        r2_geometry = r2_score(y_test, pred_geometry)
        mae_geometry = mean_absolute_error(y_test, pred_geometry)

        # Improvement
        improvement = r2_geometry - r2_baseline
        improvement_pct = 100 * improvement / (abs(r2_baseline) + 1e-10)
        mae_improvement = mae_baseline - mae_geometry

        results.append(
            {
                "seed": seed,
                "r2_baseline": float(r2_baseline),
                "r2_geometry": float(r2_geometry),
                "mae_baseline": float(mae_baseline),
                "mae_geometry": float(mae_geometry),
                "improvement": float(improvement),
                "improvement_pct": float(improvement_pct),
                "mae_improvement": float(mae_improvement),
            }
        )

        if i < 3:  # Print first 3 trials
            print(
                f"  Trial {i + 1:2d}: Baseline R²={r2_baseline:.3f}, "
                f"Geometry R²={r2_geometry:.3f}, "
                f"Delta={improvement:+.4f} ({improvement_pct:+.1f}%)"
            )

    print(f"  ... ({n_trials - 3} more trials)")
    print()

    # Aggregate statistics
    r2_baseline_mean = np.mean([r["r2_baseline"] for r in results])
    r2_baseline_std = np.std([r["r2_baseline"] for r in results])

    r2_geometry_mean = np.mean([r["r2_geometry"] for r in results])
    r2_geometry_std = np.std([r["r2_geometry"] for r in results])

    improvement_mean = np.mean([r["improvement"] for r in results])
    improvement_std = np.std([r["improvement"] for r in results])
    improvement_pct_mean = np.mean([r["improvement_pct"] for r in results])

    mae_baseline_mean = np.mean([r["mae_baseline"] for r in results])
    mae_geometry_mean = np.mean([r["mae_geometry"] for r in results])
    mae_improvement_mean = np.mean([r["mae_improvement"] for r in results])

    # Statistical significance
    improvements = [r["improvement"] for r in results]
    if len(set(improvements)) > 1:  # Check if there's variation
        t_stat, p_value = ttest_1samp(improvements, 0)
    else:
        t_stat, p_value = float("inf"), 0.0

    n_wins = sum(1 for r in results if r["improvement"] > 0)
    win_rate = 100 * n_wins / n_trials

    print("RESULTS:")
    print(f"  Baseline R²: {r2_baseline_mean:.3f} ± {r2_baseline_std:.3f}")
    print(f"  Geometry R²: {r2_geometry_mean:.3f} ± {r2_geometry_std:.3f}")
    print(
        f"  Improvement: {improvement_mean:+.4f} ± {improvement_std:.3f} ({improvement_pct_mean:+.1f}%)"
    )
    print()
    print(f"  Baseline MAE: {mae_baseline_mean:.3f}")
    print(f"  Geometry MAE: {mae_geometry_mean:.3f}")
    print(f"  MAE reduction: {mae_improvement_mean:+.3f}")
    print()
    print(f"  Win rate: {n_wins}/{n_trials} ({win_rate:.0f}%)")
    print(f"  Statistical significance: t={t_stat:.2f}, p={p_value:.2e}")

    if p_value < 0.001:
        sig_label = "*** HIGHLY SIGNIFICANT"
    elif p_value < 0.01:
        sig_label = "** SIGNIFICANT"
    elif p_value < 0.05:
        sig_label = "* SIGNIFICANT"
    else:
        sig_label = "not significant"
    print(f"  {sig_label}")

    return {
        "zone_name": zone_name,
        "n_samples": len(y),
        "boundary_distance_mean": float(y.mean()),
        "boundary_distance_std": float(y.std()),
        "boundary_distance_range": [float(y.min()), float(y.max())],
        "r2_baseline_mean": float(r2_baseline_mean),
        "r2_baseline_std": float(r2_baseline_std),
        "r2_geometry_mean": float(r2_geometry_mean),
        "r2_geometry_std": float(r2_geometry_std),
        "improvement_mean": float(improvement_mean),
        "improvement_std": float(improvement_std),
        "improvement_pct_mean": float(improvement_pct_mean),
        "mae_baseline_mean": float(mae_baseline_mean),
        "mae_geometry_mean": float(mae_geometry_mean),
        "mae_improvement_mean": float(mae_improvement_mean),
        "win_rate": float(win_rate),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significance": sig_label,
        "trials": results,
    }


def main():
    print("\n" + "=" * 80)
    print("BOUNDARY-SLICED EVALUATION")
    print("=" * 80 + "\n")

    print("Testing: Where do geometry features help most?")
    print()
    print("Zones:")
    print("  SAFE: boundary_distance > 0.5 (model confident & correct)")
    print("  BORDERLINE: |boundary_distance| < 0.5 (high uncertainty)")
    print("  UNSAFE: boundary_distance < -0.5 (model confident & wrong)")
    print()

    # Load data
    embeddings, boundary_distances = load_data()
    print(f"Loaded: N={len(embeddings)}, D={embeddings.shape[1]}")

    # Compute geometry features
    print("Computing geometry features...")
    geometry_features, query_boundaries, query_embeddings = compute_geometry_features(
        embeddings, boundary_distances
    )
    print(f"Query set: N={len(query_boundaries)}")
    print()

    # Define zones
    zones = define_zones(query_boundaries, safe_threshold=0.5, unsafe_threshold=-0.5)

    print("Zone breakdown:")
    for zone_name, mask in zones.items():
        count = mask.sum()
        pct = 100 * count / len(mask)
        print(f"  {zone_name.upper():12s}: {count:3d} samples ({pct:5.1f}%)")
    print()

    # Prepare features
    X_baseline = query_embeddings
    X_geometry = np.concatenate([query_embeddings, geometry_features], axis=1)

    # Evaluate each zone
    zone_results = {}

    for zone_name, mask in zones.items():
        if mask.sum() == 0:
            print(f"\nWARNING:  Skipping {zone_name} zone (no samples)")
            continue

        zone_results[zone_name] = evaluate_zone(
            X_baseline[mask],
            X_geometry[mask],
            query_boundaries[mask],
            zone_name,
            n_trials=20,
        )

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: IMPROVEMENT BY ZONE")
    print("=" * 80 + "\n")

    print(
        f"{'Zone':<12s} {'N':>5s} {'Baseline R²':>12s} {'Geometry R²':>12s} {'Improvement':>12s} {'Sig':>6s}"
    )
    print("-" * 80)

    for zone_name in ["safe", "borderline", "unsafe"]:
        if zone_name not in zone_results:
            continue

        result = zone_results[zone_name]
        n = result["n_samples"]
        r2_base = result["r2_baseline_mean"]
        r2_geom = result["r2_geometry_mean"]
        imp_pct = result["improvement_pct_mean"]

        sig_marker = (
            "***"
            if result["p_value"] < 0.001
            else "**"
            if result["p_value"] < 0.01
            else "*"
            if result["p_value"] < 0.05
            else ""
        )

        print(
            f"{zone_name.upper():<12s} {n:5d} "
            f"{r2_base:12.3f} {r2_geom:12.3f} "
            f"{imp_pct:+11.1f}% {sig_marker:>6s}"
        )

    print()

    # Identify where improvement is greatest
    if len(zone_results) > 0:
        max_improvement_zone = max(
            zone_results.items(), key=lambda x: x[1]["improvement_pct_mean"]
        )

        print("=" * 80)
        print("CONCLUSION")
        print("=" * 80 + "\n")

        print(f"LARGEST IMPROVEMENT: {max_improvement_zone[0].upper()} zone")
        print(
            f"   Geometry provides {max_improvement_zone[1]['improvement_pct_mean']:+.1f}% improvement"
        )
        print(f"   (p = {max_improvement_zone[1]['p_value']:.2e})")
        print()

        # Check if borderline is the winner
        if "borderline" in zone_results:
            borderline_imp = zone_results["borderline"]["improvement_pct_mean"]
            other_zones_imp = [
                zone_results[z]["improvement_pct_mean"]
                for z in zone_results
                if z != "borderline"
            ]

            if other_zones_imp and borderline_imp > max(other_zones_imp):
                print("HYPOTHESIS CONFIRMED:")
                print("  Geometry features help MOST on borderline cases")
                print("  where baseline methods struggle.")
                print()
                print("  This validates the core value proposition:")
                print("  Geometric features provide safety signals where")
                print("  embedding-only methods are uncertain.")
            elif other_zones_imp and borderline_imp > np.mean(other_zones_imp):
                print("✓ HYPOTHESIS PARTIALLY CONFIRMED:")
                print("  Geometry helps on borderline, but also effective elsewhere.")
            else:
                print("WARNING:  UNEXPECTED:")
                print("  Geometry helps less on borderline than other zones.")
                print("  This suggests value may be in detecting extreme cases.")

        print()

    # Save results
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"boundary_sliced_evaluation_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "methodology": {
            "safe_threshold": 0.5,
            "unsafe_threshold": -0.5,
            "n_trials_per_zone": 20,
            "model": "Ridge(alpha=1.0)",
            "metric": "R²",
        },
        "zones": zone_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
