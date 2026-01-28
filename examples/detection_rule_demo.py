"""
Detection Rule Demo — Simple High-Risk Flagging

Demonstrates the key finding: LOW participation_ratio and LOW spectral_entropy
predict behavioral instability (proximity to decision boundaries).

Usage:
    python examples/detection_rule_demo.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import compute_safety_diagnostics, print_diagnostics_summary


def simple_detection_rule(participation_ratio, spectral_entropy, percentile=30):
    """
    Simple high-risk detection rule based on topology features.

    Rule: Flag as HIGH RISK if participation_ratio OR spectral_entropy
          falls below the specified percentile threshold.

    Args:
        participation_ratio: (N,) array
        spectral_entropy: (N,) array
        percentile: threshold percentile (default: 30)

    Returns:
        high_risk_mask: (N,) boolean array
    """
    pr_threshold = np.percentile(participation_ratio, percentile)
    se_threshold = np.percentile(spectral_entropy, percentile)

    high_risk = (participation_ratio < pr_threshold) | (spectral_entropy < se_threshold)
    return high_risk


def main():
    print("="*70)
    print("GEOMETRIC SAFETY FEATURES — DETECTION RULE DEMO")
    print("="*70)

    # Load data
    base = Path(__file__).parent.parent
    embeddings = np.load(base / "embeddings.npy")
    boundary_distances = np.load(base / "boundary_distances.npy")

    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    y = boundary_distances[split:]

    print(f"\nData: {len(queries)} query samples, {len(reference)} reference samples")

    # Compute safety diagnostics
    print("\nComputing geometric features...")
    diag = compute_safety_diagnostics(queries, reference)

    # Extract topology features (Phase 5)
    # Index 4 = spectral_entropy, Index 5 = participation_ratio
    spectral_entropy = diag.phase2_5_features[:, 4]
    participation_ratio = diag.phase2_5_features[:, 5]

    print(f"\nTopology Feature Statistics:")
    print(f"  participation_ratio: mean={participation_ratio.mean():.3f}, std={participation_ratio.std():.3f}")
    print(f"  spectral_entropy:    mean={spectral_entropy.mean():.3f}, std={spectral_entropy.std():.3f}")

    # Apply detection rule
    print("\n" + "="*70)
    print("APPLYING DETECTION RULE")
    print("="*70)

    high_risk = simple_detection_rule(participation_ratio, spectral_entropy, percentile=30)

    print(f"\nRule: Flag if participation_ratio < P30 OR spectral_entropy < P30")
    print(f"  P30 thresholds: PR < {np.percentile(participation_ratio, 30):.3f}, SE < {np.percentile(spectral_entropy, 30):.3f}")
    print(f"\nResults:")
    print(f"  Samples flagged: {high_risk.sum()}/{len(queries)} ({100*high_risk.mean():.1f}%)")

    # Evaluate against actual boundary distance
    borderline = (y >= -0.5) & (y <= 0.5)

    if high_risk.sum() > 0:
        precision = borderline[high_risk].mean()
        print(f"  Precision (flagged that are borderline): {100*precision:.1f}%")

    if borderline.sum() > 0:
        recall = high_risk[borderline].mean()
        print(f"  Recall (borderline that are flagged): {100*recall:.1f}%")

    # Show examples
    print("\n" + "="*70)
    print("EXAMPLE SAMPLES")
    print("="*70)

    # Highest risk (lowest PR)
    risk_order = np.argsort(participation_ratio)

    print("\nTop 5 Highest Risk (lowest participation_ratio):")
    print("  Idx  | PR      | SE      | Boundary Dist | Zone")
    print("  " + "-"*55)
    for idx in risk_order[:5]:
        bd = y[idx]
        zone = "BORDERLINE" if -0.5 <= bd <= 0.5 else "SAFE" if bd > 0.5 else "UNSAFE"
        print(f"  {idx:4d} | {participation_ratio[idx]:.3f}   | {spectral_entropy[idx]:.3f}   | {bd:+.3f}         | {zone}")

    print("\nTop 5 Lowest Risk (highest participation_ratio):")
    print("  Idx  | PR      | SE      | Boundary Dist | Zone")
    print("  " + "-"*55)
    for idx in risk_order[-5:][::-1]:
        bd = y[idx]
        zone = "BORDERLINE" if -0.5 <= bd <= 0.5 else "SAFE" if bd > 0.5 else "UNSAFE"
        print(f"  {idx:4d} | {participation_ratio[idx]:.3f}   | {spectral_entropy[idx]:.3f}   | {bd:+.3f}         | {zone}")

    # Summary statistics by risk group
    print("\n" + "="*70)
    print("SUMMARY BY RISK GROUP")
    print("="*70)

    print("\n                    | High Risk    | Low Risk")
    print("  " + "-"*50)
    print(f"  Count             | {high_risk.sum():12d} | {(~high_risk).sum():12d}")
    print(f"  Mean boundary dist| {y[high_risk].mean():+12.3f} | {y[~high_risk].mean():+12.3f}")
    print(f"  % Borderline      | {100*borderline[high_risk].mean():11.1f}% | {100*borderline[~high_risk].mean():11.1f}%")
    print(f"  Mean PR           | {participation_ratio[high_risk].mean():12.3f} | {participation_ratio[~high_risk].mean():12.3f}")
    print(f"  Mean SE           | {spectral_entropy[high_risk].mean():12.3f} | {spectral_entropy[~high_risk].mean():12.3f}")

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
    The high-risk group (low PR/SE) has:
    - Lower mean boundary distance (closer to decision boundary)
    - Higher percentage of borderline samples

    This validates the discovery: CONSTRAINED LOCAL DIMENSIONALITY
    (low participation ratio, low spectral entropy) predicts
    proximity to decision boundaries where behavioral instability
    concentrates.
    """)


if __name__ == "__main__":
    main()
