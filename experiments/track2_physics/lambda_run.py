"""
Track 2: Architectural Insight — The Lambda Run

Computes the global G ratio for clean vs poisoned embeddings:
    G = min(participation_ratio) / mean(participation_ratio)

Hypothesis: Poisoned models (especially cluster) have anomalously low G,
indicating more uniformly constrained geometry across all samples.

A low G means the minimum PR is close to the mean — geometry is uniformly
constrained. A high G means there are outliers with much lower PR than average.
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mirrorfield.geometry import compute_safety_diagnostics


def compute_g_ratio(embeddings, reference, name):
    """Compute G ratio for a set of embeddings."""

    # Compute diagnostics
    diag = compute_safety_diagnostics(embeddings, reference)

    # Extract participation_ratio (index 5 in phase2_5_features based on feature order)
    # Order: turbulence_index, thermal_gradient, vorticity, d_eff, spectral_entropy, participation_ratio
    participation_ratio = diag.phase2_5_features[:, 5]

    # Filter out any NaN values
    valid = ~np.isnan(participation_ratio)
    pr_valid = participation_ratio[valid]

    if len(pr_valid) == 0:
        return None

    pr_min = pr_valid.min()
    pr_mean = pr_valid.mean()
    pr_max = pr_valid.max()
    pr_std = pr_valid.std()

    # G ratio: how close is the minimum to the mean?
    # G = min / mean (lower G = more uniform constraint)
    G = pr_min / pr_mean if pr_mean > 0 else 0

    # Also compute coefficient of variation
    cv = pr_std / pr_mean if pr_mean > 0 else 0

    print(f"\n{name}:")
    print(f"  Participation Ratio Statistics:")
    print(f"    min:  {pr_min:.4f}")
    print(f"    mean: {pr_mean:.4f}")
    print(f"    max:  {pr_max:.4f}")
    print(f"    std:  {pr_std:.4f}")
    print(f"    CV:   {cv:.4f}")
    print(f"  G ratio (min/mean): {G:.4f}")

    return {
        "pr_min": float(pr_min),
        "pr_mean": float(pr_mean),
        "pr_max": float(pr_max),
        "pr_std": float(pr_std),
        "cv": float(cv),
        "G": float(G),
        "n_samples": int(len(pr_valid)),
    }


def main():
    print("=" * 70)
    print("TRACK 2: THE LAMBDA RUN — Global Geometry Ratio Analysis")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent.parent / "track1_poison" / "data"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    embeddings = np.load(data_dir / "embeddings.npy")

    # Split into reference and query (same as Track 1)
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]

    print(f"\nData: {len(queries)} query samples, {len(reference)} reference samples")

    # Load poison masks
    strategies = ["random", "boundary", "cluster"]

    results = {}

    # 1. Clean perspective (all samples treated equally)
    print("\n" + "=" * 60)
    print("CLEAN PERSPECTIVE")
    print("=" * 60)
    results["clean_all"] = compute_g_ratio(queries, reference, "All Test Samples")

    # 2. Per-strategy analysis: Compare clean vs poisoned samples
    for strategy in strategies:
        print("\n" + "=" * 60)
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 60)

        poison_mask = np.load(data_dir / f"poison_mask_{strategy}.npy")
        test_poison_mask = poison_mask[split:]

        # Clean samples only
        clean_queries = queries[~test_poison_mask]
        results[f"{strategy}_clean"] = compute_g_ratio(
            clean_queries, reference, f"{strategy} - Clean Samples Only"
        )

        # Poisoned samples only (if enough)
        if test_poison_mask.sum() >= 5:
            poison_queries = queries[test_poison_mask]
            results[f"{strategy}_poison"] = compute_g_ratio(
                poison_queries, reference, f"{strategy} - Poisoned Samples Only"
            )

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: G RATIO COMPARISON")
    print("=" * 70)

    print(f"\n  {'Dataset':<30} | {'G Ratio':>10} | {'PR Mean':>10} | {'PR Min':>10} | {'CV':>10}")
    print("  " + "-" * 80)

    for name, data in results.items():
        if data:
            print(f"  {name:<30} | {data['G']:>10.4f} | {data['pr_mean']:>10.4f} | {data['pr_min']:>10.4f} | {data['cv']:>10.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Compare G ratios
    clean_G = results["clean_all"]["G"]

    print(f"""
    G Ratio Analysis:
    - G = min(PR) / mean(PR)
    - Lower G = minimum is far below mean = outliers with constrained geometry
    - Higher G = minimum is close to mean = uniform geometry across samples

    Baseline (all samples): G = {clean_G:.4f}
    """)

    for strategy in strategies:
        if f"{strategy}_poison" in results and results[f"{strategy}_poison"]:
            poison_G = results[f"{strategy}_poison"]["G"]
            clean_G_strat = results[f"{strategy}_clean"]["G"]

            delta = poison_G - clean_G_strat
            interpretation = "more uniform constraint" if delta > 0 else "more extreme outliers"

            print(f"    {strategy.upper()}:")
            print(f"      Clean G:   {clean_G_strat:.4f}")
            print(f"      Poison G:  {poison_G:.4f}")
            print(f"      Delta:     {delta:+.4f} ({interpretation})")

    # Save results
    with open(output_dir / "lambda_run_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_dir / 'lambda_run_results.json'}")


if __name__ == "__main__":
    main()
