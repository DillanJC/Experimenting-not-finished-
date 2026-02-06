"""
Track 1: Poison Detection — Step 3: Compute Geometric Features

Computes full geometric diagnostics for:
1. Clean model perspective
2. Poisoned model perspectives (3 strategies)

Then analyzes whether poisoned samples have distinctive geometry.
"""

import sys
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, ttest_ind

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mirrorfield.geometry import compute_safety_diagnostics


def compute_features_for_model(embeddings, boundary_distances, model_name, output_dir):
    """Compute all geometric features using given boundary distances."""

    print(f"\nComputing features for: {model_name}")
    print("-" * 50)

    # Split 80/20 (same as training)
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]

    # Compute safety diagnostics
    diag = compute_safety_diagnostics(queries, reference)

    # Get all features
    all_features = diag.get_all_features()
    feature_names = diag.get_feature_names()

    print(f"  Features computed: {all_features.shape}")
    print(f"  Feature names: {feature_names}")

    # Save features
    np.save(output_dir / f"features_{model_name}.npy", all_features)

    # Save feature names
    with open(output_dir / f"feature_names.json", "w") as f:
        json.dump(feature_names, f)

    return all_features, feature_names, diag


def analyze_poison_geometry(features, poison_mask, feature_names, strategy_name):
    """Analyze whether poisoned samples have distinctive geometry."""

    print(f"\n{'='*60}")
    print(f"GEOMETRIC ANALYSIS: {strategy_name}")
    print("=" * 60)

    clean_mask = ~poison_mask
    n_clean = clean_mask.sum()
    n_poison = poison_mask.sum()

    print(f"  Clean samples: {n_clean}")
    print(f"  Poisoned samples: {n_poison}")

    if n_poison < 5:
        print("  Not enough poisoned samples for analysis")
        return None

    results = {}

    print(f"\n  Feature differences (poisoned vs clean):")
    print(f"  {'Feature':<30} | {'Clean Mean':>10} | {'Poison Mean':>10} | {'Delta':>10} | {'p-value':>10}")
    print("  " + "-" * 85)

    significant_features = []

    for i, name in enumerate(feature_names):
        clean_vals = features[clean_mask, i]
        poison_vals = features[poison_mask, i]

        clean_mean = clean_vals.mean()
        poison_mean = poison_vals.mean()
        delta = poison_mean - clean_mean

        # T-test
        if len(poison_vals) > 1 and clean_vals.std() > 0:
            t_stat, p_value = ttest_ind(clean_vals, poison_vals)
        else:
            p_value = 1.0

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

        if p_value < 0.05:
            significant_features.append((name, delta, p_value))

        print(f"  {name:<30} | {clean_mean:>10.4f} | {poison_mean:>10.4f} | {delta:>+10.4f} | {p_value:>10.4f} {sig}")

        results[name] = {
            "clean_mean": float(clean_mean),
            "poison_mean": float(poison_mean),
            "delta": float(delta),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }

    print(f"\n  Significant features (p < 0.05): {len(significant_features)}")
    for name, delta, p in significant_features:
        direction = "HIGHER" if delta > 0 else "LOWER"
        print(f"    - {name}: {direction} in poisoned (delta={delta:+.4f}, p={p:.4f})")

    return results


def compute_svd_spectral_signatures(embeddings, labels, poison_mask, output_dir, strategy_name):
    """
    Compute per-class SVD to find spectral signatures of poisoning.

    Based on "Spectral Signatures" method: poisoned samples concentrate
    along top singular vectors of class-conditional representations.
    """
    print(f"\n{'='*60}")
    print(f"SVD SPECTRAL SIGNATURES: {strategy_name}")
    print("=" * 60)

    results = {}

    for class_label in [0, 1]:
        class_mask = labels == class_label
        class_embeddings = embeddings[class_mask]
        class_poison = poison_mask[class_mask]

        n_class = len(class_embeddings)
        n_poison_class = class_poison.sum()

        print(f"\n  Class {class_label}: {n_class} samples, {n_poison_class} poisoned")

        if n_poison_class < 2:
            print("    Not enough poisoned samples for SVD analysis")
            continue

        # Center the embeddings
        centered = class_embeddings - class_embeddings.mean(axis=0)

        # SVD
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)

        # Project onto top singular vectors
        n_top = min(10, len(S))
        projections = centered @ Vh[:n_top].T  # (N, n_top)

        # Check if poisoned samples are outliers in top singular directions
        print(f"\n    Top {n_top} singular direction analysis:")
        print(f"    {'SV#':<5} | {'Clean Mean':>10} | {'Poison Mean':>10} | {'Delta':>10} | {'Outlier Score':>12}")
        print("    " + "-" * 65)

        outlier_scores = []
        for sv_idx in range(n_top):
            clean_proj = projections[~class_poison, sv_idx]
            poison_proj = projections[class_poison, sv_idx]

            clean_mean = clean_proj.mean()
            clean_std = clean_proj.std() + 1e-10
            poison_mean = poison_proj.mean()

            # Outlier score: how many std away is poison mean from clean mean
            outlier_score = abs(poison_mean - clean_mean) / clean_std

            outlier_scores.append(outlier_score)

            sig = "***" if outlier_score > 3 else "**" if outlier_score > 2 else "*" if outlier_score > 1 else ""
            print(f"    {sv_idx:<5} | {clean_mean:>10.4f} | {poison_mean:>10.4f} | {poison_mean-clean_mean:>+10.4f} | {outlier_score:>12.2f} {sig}")

        results[f"class_{class_label}"] = {
            "n_samples": int(n_class),
            "n_poisoned": int(n_poison_class),
            "singular_values": S[:n_top].tolist(),
            "outlier_scores": outlier_scores,
            "max_outlier_sv": int(np.argmax(outlier_scores)),
            "max_outlier_score": float(max(outlier_scores)),
        }

        if max(outlier_scores) > 2:
            print(f"\n    DETECTION: Poisoned samples concentrate on SV#{np.argmax(outlier_scores)} (score={max(outlier_scores):.2f})")

    return results


def main():
    print("=" * 70)
    print("TRACK 1: POISON DETECTION — Computing Geometric Features")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent / "data"
    models_dir = Path(__file__).parent / "models"
    output_dir = Path(__file__).parent / "features"
    output_dir.mkdir(exist_ok=True)

    # Load data
    embeddings = np.load(data_dir / "embeddings.npy")
    original_labels = np.load(data_dir / "original_labels.npy")

    # Split for test set analysis
    split = int(len(embeddings) * 0.8)
    test_embeddings = embeddings[split:]
    test_labels = original_labels[split:]

    # Compute features (using clean model's perspective - embeddings are the same)
    features, feature_names, diag = compute_features_for_model(
        embeddings,
        np.load(models_dir / "boundary_distances_clean.npy"),
        "all_samples",
        output_dir
    )

    # Analyze each poisoning strategy
    strategies = ["random", "boundary", "cluster"]
    all_results = {}

    for strategy in strategies:
        poison_mask = np.load(data_dir / f"poison_mask_{strategy}.npy")
        poisoned_labels = np.load(data_dir / f"poisoned_labels_{strategy}.npy")

        # Only analyze test set
        test_poison_mask = poison_mask[split:]

        # Feature analysis
        geom_results = analyze_poison_geometry(
            features, test_poison_mask, feature_names, strategy
        )

        # SVD spectral signatures (on full dataset by class)
        svd_results = compute_svd_spectral_signatures(
            embeddings, poisoned_labels, poison_mask, output_dir, strategy
        )

        all_results[strategy] = {
            "geometry": geom_results,
            "svd_signatures": svd_results,
        }

    # Save all results
    with open(output_dir / "poison_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
