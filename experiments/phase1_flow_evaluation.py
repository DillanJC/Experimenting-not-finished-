"""
Phase 1 Flow Features Evaluation — Do Differential Features Add Non-Redundant Signal?

Tests the incremental utility of Phase 1 flow features:
1. local_gradient_magnitude: Mean-shift vector norm
2. gradient_direction_consistency: Fisher-z aggregated cosine
3. pressure_differential: Log k-NN radii at two scales

Evaluation Protocol (from plan):
1. Signal: Correlations and partial correlations controlling for Tier-0 features
2. Performance: ΔAUC/ΔR² via stratified CV, focusing on borderline bins
3. Redundancy: Check |r| > 0.9 with Tier-0 features
4. Stability: ICC/CV% over multiple seeds, sensitivity to k and bandwidth

Hypothesis: Flow features provide non-redundant signal, especially on borderline cases
where baseline methods struggle (4.8× improvement target).
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle
from mirrorfield.geometry.phase1_flow_features import (
    compute_phase1_features,
    compute_phase1_features_fast,
    combine_with_tier0_features,
    PHASE1_FEATURE_NAMES,
    PHASE1_FEATURE_NAMES_FAST,
)
from mirrorfield.geometry.features import FEATURE_NAMES

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import pearsonr, spearmanr, ttest_1samp
import warnings


def load_data():
    """Load sentiment classification data."""
    possible_paths = [
        Path(__file__).parent.parent,  # Project root
        Path(__file__).parent.parent / "openai_3_large_test_20251231_024532",
        Path(__file__).parent.parent / "runs" / "openai_3_large_test_20251231_024532",
        Path("runs/openai_3_large_test_20251231_024532"),
        Path("C:/Users/User/geometric_safety_features-Experiment"),
        Path("C:/Users/User/geometric_safety_features-Experiment/openai_3_large_test_20251231_024532"),
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
        )

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    return embeddings, boundary_distances


def compute_all_features(embeddings, boundary_distances, k=50, use_fast=True):
    """Compute Tier-0 and Phase 1 features for query set."""
    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    print(f"  Reference set: {reference.shape[0]} samples")
    print(f"  Query set: {queries.shape[0]} samples")

    # Compute Tier-0 geometry features
    print("  Computing Tier-0 features...")
    bundle = GeometryBundle(reference, k=k)
    tier0_results = bundle.compute(queries)
    tier0_features = bundle.get_feature_matrix(tier0_results)

    # Compute Phase 1 flow features
    print("  Computing Phase 1 flow features...")
    if use_fast:
        phase1_features, phase1_meta = compute_phase1_features_fast(
            queries, reference, k=k
        )
        phase1_names = PHASE1_FEATURE_NAMES_FAST
    else:
        phase1_features, phase1_meta = compute_phase1_features(
            queries, reference, k=k
        )
        phase1_names = PHASE1_FEATURE_NAMES

    print(f"  Phase 1 d_eff: {phase1_meta['d_eff_mean']:.1f} ± {phase1_meta.get('d_eff_std', 0):.1f}")

    return {
        "queries": queries,
        "query_boundaries": query_boundaries,
        "tier0_features": tier0_features,
        "phase1_features": phase1_features,
        "phase1_meta": phase1_meta,
        "tier0_names": FEATURE_NAMES,
        "phase1_names": phase1_names,
        "reference": reference,
    }


def analyze_correlations(data):
    """Analyze correlations between features and boundary distance."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    tier0 = data["tier0_features"]
    phase1 = data["phase1_features"]
    y = data["query_boundaries"]

    tier0_names = data["tier0_names"]
    phase1_names = data["phase1_names"]

    results = {
        "tier0_correlations": {},
        "phase1_correlations": {},
        "inter_feature_correlations": {},
    }

    # Tier-0 correlations with boundary distance
    print("\nTier-0 Features vs Boundary Distance:")
    print("-" * 50)
    for i, name in enumerate(tier0_names):
        r, p = pearsonr(tier0[:, i], y)
        rho, p_s = spearmanr(tier0[:, i], y)
        results["tier0_correlations"][name] = {"pearson_r": r, "spearman_rho": rho, "p_value": p}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:30s}: r={r:+.3f}, rho={rho:+.3f} {sig}")

    # Phase 1 correlations with boundary distance
    print("\nPhase 1 Features vs Boundary Distance:")
    print("-" * 50)
    for i, name in enumerate(phase1_names):
        r, p = pearsonr(phase1[:, i], y)
        rho, p_s = spearmanr(phase1[:, i], y)
        results["phase1_correlations"][name] = {"pearson_r": r, "spearman_rho": rho, "p_value": p}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:30s}: r={r:+.3f}, rho={rho:+.3f} {sig}")

    # Redundancy check: Phase 1 vs Tier-0 correlations
    print("\nRedundancy Check (Phase 1 vs Tier-0):")
    print("-" * 50)
    print("  Checking for |r| > 0.9 (high redundancy)...")

    redundant_pairs = []
    for i, p1_name in enumerate(phase1_names):
        for j, t0_name in enumerate(tier0_names):
            r, _ = pearsonr(phase1[:, i], tier0[:, j])
            results["inter_feature_correlations"][f"{p1_name}_vs_{t0_name}"] = r
            if abs(r) > 0.9:
                redundant_pairs.append((p1_name, t0_name, r))
                print(f"  WARNING: {p1_name} highly correlated with {t0_name}: r={r:.3f}")

    if not redundant_pairs:
        print("  [OK] No high redundancy detected (all |r| < 0.9)")

    # Top inter-feature correlations
    print("\nTop Inter-Feature Correlations:")
    sorted_corrs = sorted(
        results["inter_feature_correlations"].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    for pair, r in sorted_corrs:
        print(f"  {pair}: r={r:+.3f}")

    return results


def define_zones(boundary_distances, safe_threshold=0.5, unsafe_threshold=-0.5):
    """Split data into zones based on boundary distance."""
    safe_mask = boundary_distances > safe_threshold
    borderline_mask = (boundary_distances >= unsafe_threshold) & (boundary_distances <= safe_threshold)
    unsafe_mask = boundary_distances < unsafe_threshold
    return {"safe": safe_mask, "borderline": borderline_mask, "unsafe": unsafe_mask}


def evaluate_incremental_utility(data, n_trials=20):
    """
    Evaluate incremental utility of Phase 1 features over Tier-0.

    Compares three models:
    1. Baseline: Embeddings only
    2. Tier-0: Embeddings + 7 k-NN features
    3. Combined: Embeddings + Tier-0 + Phase 1 features
    """
    print("\n" + "=" * 80)
    print("INCREMENTAL UTILITY EVALUATION")
    print("=" * 80)

    queries = data["queries"]
    tier0 = data["tier0_features"]
    phase1 = data["phase1_features"]
    y = data["query_boundaries"]

    # Feature sets
    X_baseline = queries
    X_tier0 = np.concatenate([queries, tier0], axis=1)
    X_combined = np.concatenate([queries, tier0, phase1], axis=1)

    print(f"\nFeature dimensions:")
    print(f"  Baseline (embeddings): {X_baseline.shape[1]}")
    print(f"  Tier-0 (+ 7 k-NN):     {X_tier0.shape[1]}")
    print(f"  Combined (+ Phase 1):  {X_combined.shape[1]}")

    # Define zones
    zones = define_zones(y)

    results = {"global": {}, "by_zone": {}}

    # Global evaluation
    print(f"\n{'=' * 60}")
    print("GLOBAL EVALUATION (all samples)")
    print(f"{'=' * 60}")

    global_results = evaluate_feature_sets(
        X_baseline, X_tier0, X_combined, y,
        n_trials=n_trials, zone_name="global"
    )
    results["global"] = global_results

    # Zone-specific evaluation
    for zone_name, mask in zones.items():
        if mask.sum() < 20:
            print(f"\nSkipping {zone_name} zone (only {mask.sum()} samples)")
            continue

        print(f"\n{'=' * 60}")
        print(f"{zone_name.upper()} ZONE ({mask.sum()} samples)")
        print(f"{'=' * 60}")

        zone_results = evaluate_feature_sets(
            X_baseline[mask], X_tier0[mask], X_combined[mask], y[mask],
            n_trials=n_trials, zone_name=zone_name
        )
        results["by_zone"][zone_name] = zone_results

    return results


def evaluate_feature_sets(X_baseline, X_tier0, X_combined, y, n_trials=20, zone_name=""):
    """Compare three feature sets on the same data."""
    results_baseline = []
    results_tier0 = []
    results_combined = []

    for i in range(n_trials):
        seed = 42 + i

        # Split data
        splits = train_test_split(
            X_baseline, X_tier0, X_combined, y,
            test_size=0.2, random_state=seed
        )
        X_base_train, X_base_test = splits[0], splits[1]
        X_t0_train, X_t0_test = splits[2], splits[3]
        X_comb_train, X_comb_test = splits[4], splits[5]
        y_train, y_test = splits[6], splits[7]

        # Baseline
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(X_base_train, y_train)
        r2_base = r2_score(y_test, ridge.predict(X_base_test))

        # Tier-0
        ridge.fit(X_t0_train, y_train)
        r2_tier0 = r2_score(y_test, ridge.predict(X_t0_test))

        # Combined
        ridge.fit(X_comb_train, y_train)
        r2_comb = r2_score(y_test, ridge.predict(X_comb_test))

        results_baseline.append(r2_base)
        results_tier0.append(r2_tier0)
        results_combined.append(r2_comb)

    # Statistics
    r2_base_mean = np.mean(results_baseline)
    r2_tier0_mean = np.mean(results_tier0)
    r2_comb_mean = np.mean(results_combined)

    tier0_improvement = r2_tier0_mean - r2_base_mean
    phase1_improvement = r2_comb_mean - r2_tier0_mean
    total_improvement = r2_comb_mean - r2_base_mean

    tier0_pct = 100 * tier0_improvement / (abs(r2_base_mean) + 1e-10)
    phase1_pct = 100 * phase1_improvement / (abs(r2_tier0_mean) + 1e-10)
    total_pct = 100 * total_improvement / (abs(r2_base_mean) + 1e-10)

    # Statistical significance of Phase 1 increment
    increments = [c - t for c, t in zip(results_combined, results_tier0)]
    if len(set(increments)) > 1:
        t_stat, p_value = ttest_1samp(increments, 0)
    else:
        t_stat, p_value = 0, 1.0

    # Print results
    print(f"\nResults ({n_trials} trials):")
    print(f"  Baseline R²:  {r2_base_mean:.4f} ± {np.std(results_baseline):.4f}")
    print(f"  Tier-0 R²:    {r2_tier0_mean:.4f} ± {np.std(results_tier0):.4f}  (+{tier0_pct:.1f}%)")
    print(f"  Combined R²:  {r2_comb_mean:.4f} ± {np.std(results_combined):.4f}  (+{total_pct:.1f}%)")
    print()
    print(f"  Tier-0 increment:   {tier0_improvement:+.4f} ({tier0_pct:+.1f}%)")
    print(f"  Phase 1 increment:  {phase1_improvement:+.4f} ({phase1_pct:+.1f}%)")
    print(f"  Phase 1 significance: t={t_stat:.2f}, p={p_value:.3e}")

    sig_label = (
        "*** HIGHLY SIGNIFICANT" if p_value < 0.001 else
        "** SIGNIFICANT" if p_value < 0.01 else
        "* SIGNIFICANT" if p_value < 0.05 else
        "not significant"
    )
    print(f"  {sig_label}")

    # Win rate
    n_wins = sum(1 for c, t in zip(results_combined, results_tier0) if c > t)
    print(f"  Phase 1 wins: {n_wins}/{n_trials} ({100*n_wins/n_trials:.0f}%)")

    return {
        "r2_baseline_mean": float(r2_base_mean),
        "r2_baseline_std": float(np.std(results_baseline)),
        "r2_tier0_mean": float(r2_tier0_mean),
        "r2_tier0_std": float(np.std(results_tier0)),
        "r2_combined_mean": float(r2_comb_mean),
        "r2_combined_std": float(np.std(results_combined)),
        "tier0_improvement": float(tier0_improvement),
        "tier0_improvement_pct": float(tier0_pct),
        "phase1_improvement": float(phase1_improvement),
        "phase1_improvement_pct": float(phase1_pct),
        "total_improvement": float(total_improvement),
        "total_improvement_pct": float(total_pct),
        "phase1_t_statistic": float(t_stat),
        "phase1_p_value": float(p_value),
        "phase1_significance": sig_label,
        "phase1_win_rate": float(n_wins / n_trials),
    }


def analyze_hyperparameter_sensitivity(data, k_values=[15, 30, 50, 75], c_values=[0.8, 1.0, 1.2]):
    """
    Test sensitivity to k and bandwidth multiplier c.

    From plan: "Probe sensitivity over k∈{15,30,50,75} and bandwidth multiplier c∈{0.8,1.0,1.2}"
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    reference = data["reference"]
    queries = data["queries"]
    y = data["query_boundaries"]

    results = {}

    print("\nTesting k sensitivity (bandwidth c=1.0):")
    print("-" * 50)

    for k in k_values:
        if k >= len(reference):
            print(f"  k={k}: Skipped (k >= reference size)")
            continue

        phase1_features, meta = compute_phase1_features_fast(
            queries, reference, k=k, bandwidth_multiplier=1.0
        )

        # Correlations with boundary distance
        r_mag, _ = pearsonr(phase1_features[:, 0], y)
        r_press, _ = pearsonr(phase1_features[:, 1], y)

        results[f"k={k}"] = {
            "gradient_mag_corr": float(r_mag),
            "pressure_diff_corr": float(r_press),
            "d_eff_mean": meta["d_eff_mean"],
        }

        print(f"  k={k:2d}: grad_mag r={r_mag:+.3f}, pressure r={r_press:+.3f}, d_eff={meta['d_eff_mean']:.1f}")

    print("\nTesting bandwidth sensitivity (k=50):")
    print("-" * 50)

    for c in c_values:
        phase1_features, meta = compute_phase1_features_fast(
            queries, reference, k=50, bandwidth_multiplier=c
        )

        r_mag, _ = pearsonr(phase1_features[:, 0], y)
        r_press, _ = pearsonr(phase1_features[:, 1], y)

        results[f"c={c}"] = {
            "gradient_mag_corr": float(r_mag),
            "pressure_diff_corr": float(r_press),
        }

        print(f"  c={c:.1f}: grad_mag r={r_mag:+.3f}, pressure r={r_press:+.3f}")

    # Stability assessment
    print("\nStability Assessment:")
    k_corrs = [results[f"k={k}"]["gradient_mag_corr"] for k in k_values if f"k={k}" in results]
    c_corrs = [results[f"c={c}"]["gradient_mag_corr"] for c in c_values if f"c={c}" in results]

    if len(k_corrs) > 1:
        k_range = max(k_corrs) - min(k_corrs)
        print(f"  Gradient magnitude correlation range over k: {k_range:.3f}")
        if k_range < 0.05:
            print("  [OK] Stable across k values")
        else:
            print("  ! Some sensitivity to k detected")

    if len(c_corrs) > 1:
        c_range = max(c_corrs) - min(c_corrs)
        print(f"  Gradient magnitude correlation range over c: {c_range:.3f}")
        if c_range < 0.05:
            print("  [OK] Stable across bandwidth values")
        else:
            print("  ! Some sensitivity to bandwidth detected")

    return results


def generate_summary(corr_results, utility_results, sensitivity_results):
    """Generate final summary and recommendations."""
    print("\n" + "=" * 80)
    print("PHASE 1 EVALUATION SUMMARY")
    print("=" * 80)

    # Check Phase 1 utility
    global_phase1_pct = utility_results["global"]["phase1_improvement_pct"]
    global_p_value = utility_results["global"]["phase1_p_value"]

    print("\n1. INCREMENTAL UTILITY:")
    print(f"   Global Phase 1 improvement: {global_phase1_pct:+.2f}%")
    print(f"   Statistical significance: p={global_p_value:.3e}")

    # Zone-specific
    if "borderline" in utility_results["by_zone"]:
        borderline_pct = utility_results["by_zone"]["borderline"]["phase1_improvement_pct"]
        print(f"   Borderline zone improvement: {borderline_pct:+.2f}%")

        if borderline_pct > global_phase1_pct:
            print("   [OK] HYPOTHESIS SUPPORTED: Phase 1 helps MORE on borderline cases")
        else:
            print("   ! Borderline improvement not larger than global")

    # Redundancy check
    print("\n2. REDUNDANCY CHECK:")
    phase1_corrs = corr_results.get("phase1_correlations", {})
    inter_corrs = corr_results.get("inter_feature_correlations", {})

    max_redundancy = max(abs(v) for v in inter_corrs.values()) if inter_corrs else 0
    print(f"   Max Phase1-Tier0 correlation: |r|={max_redundancy:.3f}")
    if max_redundancy < 0.9:
        print("   [OK] PASSED: No high redundancy with Tier-0 features")
    else:
        print("   ! WARNING: High redundancy detected")

    # Stability
    print("\n3. STABILITY:")
    if sensitivity_results:
        k_corrs = [v.get("gradient_mag_corr", 0) for k, v in sensitivity_results.items() if "k=" in k]
        if len(k_corrs) > 1:
            k_cv = np.std(k_corrs) / (np.mean(np.abs(k_corrs)) + 1e-10)
            print(f"   Coefficient of variation over k: {k_cv:.2f}")
            if k_cv < 0.2:
                print("   [OK] Stable across k values")

    # Recommendation
    print("\n4. RECOMMENDATION:")
    threshold = 0.5  # ≥0.5% ΔAUC/R² as per plan

    if global_phase1_pct >= threshold and global_p_value < 0.05:
        print("   [OK] RETAIN Phase 1 features")
        print(f"     - Improvement ({global_phase1_pct:.2f}%) meets threshold (≥{threshold}%)")
        print("     - Statistically significant")
        if max_redundancy < 0.9:
            print("     - Non-redundant with Tier-0")
    elif global_phase1_pct >= 0:
        print("   ? MARGINAL: Consider tuning k and bandwidth")
        print("     - Small positive improvement detected")
        print("     - May benefit from hyperparameter optimization")
    else:
        print("   [X] Phase 1 features do not meet retention criteria")
        print("     - Consider local whitening or alternative formulations")

    return {
        "global_improvement_pct": global_phase1_pct,
        "global_p_value": global_p_value,
        "max_redundancy": max_redundancy,
        "recommendation": "retain" if global_phase1_pct >= threshold and global_p_value < 0.05 else "marginal" if global_phase1_pct >= 0 else "reject"
    }


def main():
    print("\n" + "=" * 80)
    print("PHASE 1 FLOW FEATURES EVALUATION")
    print("=" * 80)
    print()
    print("Testing incremental utility of differential/density gradient features:")
    print("  - local_gradient_magnitude: Mean-shift vector norm")
    print("  - pressure_differential: Log k-NN radii at two scales")
    print()

    # Load data
    print("Loading data...")
    embeddings, boundary_distances = load_data()
    print(f"Loaded: N={len(embeddings)}, D={embeddings.shape[1]}")

    # Compute features
    print("\nComputing features...")
    data = compute_all_features(embeddings, boundary_distances, k=50, use_fast=True)

    # Analysis 1: Correlations
    corr_results = analyze_correlations(data)

    # Analysis 2: Incremental utility
    utility_results = evaluate_incremental_utility(data, n_trials=20)

    # Analysis 3: Hyperparameter sensitivity
    sensitivity_results = analyze_hyperparameter_sensitivity(data)

    # Summary
    summary = generate_summary(corr_results, utility_results, sensitivity_results)

    # Save results
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"phase1_evaluation_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "methodology": {
            "k": 50,
            "bandwidth_multiplier": 1.0,
            "n_trials": 20,
            "model": "Ridge(alpha=1.0)",
            "features": PHASE1_FEATURE_NAMES_FAST,
        },
        "correlation_analysis": corr_results,
        "utility_evaluation": utility_results,
        "sensitivity_analysis": sensitivity_results,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nFull report saved: {output_path}")


if __name__ == "__main__":
    main()
