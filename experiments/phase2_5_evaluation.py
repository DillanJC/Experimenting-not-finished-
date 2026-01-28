"""
Phase 2+5 Quick Evaluation — Weather + Topology Features
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle, compute_gradient_magnitude_only
from mirrorfield.geometry.phase2_weather_features import compute_phase2_features, PHASE2_5_FEATURE_NAMES
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def main():
    # Load data
    base = Path(__file__).parent.parent
    embeddings = np.load(base / "embeddings.npy")
    boundary_distances = np.load(base / "boundary_distances.npy")

    print(f"Loaded: N={len(embeddings)}, D={embeddings.shape[1]}")

    # Split
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    y = boundary_distances[split:]

    # Tier-0 features
    bundle = GeometryBundle(reference, k=50)
    tier0_results = bundle.compute(queries)
    tier0_features = bundle.get_feature_matrix(tier0_results)
    ridge_proximity = tier0_features[:, 5]  # ridge_proximity is index 5

    # Phase 1 (just gradient magnitude)
    g_mag, _ = compute_gradient_magnitude_only(queries, reference, k=75)
    g_dir = np.zeros((len(queries), queries.shape[1]))  # Placeholder

    # Phase 2+5 features
    phase2_5, meta = compute_phase2_features(
        queries, reference, g_mag.flatten(), g_dir, ridge_proximity,
        k=50, include_topology=True
    )

    print(f"\nPhase 2+5 Features: {phase2_5.shape}")
    print(f"Names: {meta['feature_names']}")

    # Correlations
    print("\n" + "="*60)
    print("CORRELATIONS WITH BOUNDARY DISTANCE")
    print("="*60)
    for i, name in enumerate(PHASE2_5_FEATURE_NAMES):
        r, p = pearsonr(phase2_5[:, i], y)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:25s}: r={r:+.3f} {sig}")

    # Quick R2 comparison
    print("\n" + "="*60)
    print("INCREMENTAL UTILITY (10 trials)")
    print("="*60)

    X_tier0 = np.concatenate([queries, tier0_features], axis=1)
    X_with_p1 = np.concatenate([X_tier0, g_mag], axis=1)
    X_full = np.concatenate([X_with_p1, phase2_5], axis=1)

    r2_tier0, r2_p1, r2_full = [], [], []

    for seed in range(10):
        X_t0_tr, X_t0_te, X_p1_tr, X_p1_te, X_f_tr, X_f_te, y_tr, y_te = train_test_split(
            X_tier0, X_with_p1, X_full, y, test_size=0.2, random_state=42+seed
        )

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_t0_tr, y_tr); r2_tier0.append(r2_score(y_te, ridge.predict(X_t0_te)))
        ridge.fit(X_p1_tr, y_tr); r2_p1.append(r2_score(y_te, ridge.predict(X_p1_te)))
        ridge.fit(X_f_tr, y_tr); r2_full.append(r2_score(y_te, ridge.predict(X_f_te)))

    print(f"  Tier-0 only:        R2={np.mean(r2_tier0):.4f}")
    print(f"  + Phase 1 (g_mag):  R2={np.mean(r2_p1):.4f} ({100*(np.mean(r2_p1)-np.mean(r2_tier0))/np.mean(r2_tier0):+.2f}%)")
    print(f"  + Phase 2+5:        R2={np.mean(r2_full):.4f} ({100*(np.mean(r2_full)-np.mean(r2_tier0))/np.mean(r2_tier0):+.2f}%)")

    # Zone analysis
    print("\n" + "="*60)
    print("BORDERLINE ZONE ANALYSIS")
    print("="*60)
    borderline = (y >= -0.5) & (y <= 0.5)
    if borderline.sum() > 20:
        for i, name in enumerate(PHASE2_5_FEATURE_NAMES):
            r, _ = pearsonr(phase2_5[borderline, i], y[borderline])
            print(f"  {name:25s}: r={r:+.3f} (borderline)")

if __name__ == "__main__":
    main()
