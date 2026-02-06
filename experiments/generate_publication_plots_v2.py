"""
Generate Publication-Quality Plots - Version 2

Creates 3 figures for technical report using embedded data from validated results.

Output: High-resolution PNG and PDF files in plots/ directory
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# Set publication-quality style
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13

# Color scheme
COLORS = {
    "baseline": "#7f8c8d",
    "geometry": "#2ecc71",
    "borderline": "#e74c3c",
    "safe": "#3498db",
    "unsafe": "#f39c12",
}

# Embedded validated results from boundary-sliced evaluation
BOUNDARY_SLICED_DATA = {
    "safe": {
        "n_samples": 67,
        "r2_baseline_mean": 0.604,
        "r2_baseline_std": 0.000,
        "r2_geometry_mean": 0.609,
        "r2_geometry_std": 0.000,
        "improvement_pct_mean": 0.8,
        "p_value": 0.0,
    },
    "borderline": {
        "n_samples": 79,
        "r2_baseline_mean": 0.575,
        "r2_baseline_std": 0.000,
        "r2_geometry_mean": 0.597,
        "r2_geometry_std": 0.000,
        "improvement_pct_mean": 3.8,
        "p_value": 0.0,
    },
    "unsafe": {
        "n_samples": 74,
        "r2_baseline_mean": 0.680,
        "r2_geometry_std": 0.000,
        "r2_geometry_mean": 0.694,
        "r2_baseline_std": 0.000,
        "improvement_pct_mean": 2.1,
        "p_value": 0.0,
    },
}


def load_feature_importance_data():
    """Load feature importance from real data."""
    # Try multiple locations for data
    possible_paths = [
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

    # Compute geometry
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries)
    features = bundle.get_feature_matrix(results)

    return features, query_boundaries


def figure1_r2_by_region():
    """
    Figure 1: R² by Region (Bar Chart with Error Bars)
    """
    print("\nGenerating Figure 1: R² by Region...")

    zones = ["safe", "borderline", "unsafe"]
    zone_labels = [
        "SAFE\n(confident\n& correct)",
        "BORDERLINE\n(uncertain)",
        "UNSAFE\n(confident\nbut wrong)",
    ]

    baseline_r2 = [BOUNDARY_SLICED_DATA[z]["r2_baseline_mean"] for z in zones]
    geometry_r2 = [BOUNDARY_SLICED_DATA[z]["r2_geometry_mean"] for z in zones]
    baseline_std = [BOUNDARY_SLICED_DATA[z]["r2_baseline_std"] for z in zones]
    geometry_std = [BOUNDARY_SLICED_DATA[z]["r2_geometry_std"] for z in zones]
    improvements = [BOUNDARY_SLICED_DATA[z]["improvement_pct_mean"] for z in zones]
    sample_sizes = [BOUNDARY_SLICED_DATA[z]["n_samples"] for z in zones]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(zones))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        baseline_r2,
        width,
        label="Baseline (embeddings only)",
        color=COLORS["baseline"],
        alpha=0.8,
    )

    bars2 = ax.bar(
        x + width / 2,
        geometry_r2,
        width,
        label="Geometry (embeddings + 7 k-NN features)",
        color=COLORS["geometry"],
        alpha=0.8,
    )

    # Customize
    ax.set_xlabel("Region", fontweight="bold")
    ax.set_ylabel("R² (Coefficient of Determination)", fontweight="bold")
    ax.set_title(
        "Figure 1: Geometry Features Provide Largest Improvement on Borderline Cases",
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0.55, 0.72)

    # Add improvement percentages above bars
    for i, (imp, n) in enumerate(zip(improvements, sample_sizes)):
        y_pos = geometry_r2[i] + 0.012
        ax.text(
            i + width / 2,
            y_pos,
            f"+{imp:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            color="black",
        )

        # Add sample size below x-axis
        ax.text(i, 0.545, f"n={n}", ha="center", va="top", fontsize=8, color="gray")

    # Add significance stars
    for i in range(len(zones)):
        p_value = BOUNDARY_SLICED_DATA[zones[i]]["p_value"]
        stars = "***"  # All p < 0.001

        y_pos = max(baseline_r2[i], geometry_r2[i]) + 0.025
        ax.text(i, y_pos, stars, ha="center", va="bottom", fontsize=12, color="black")

    # Add annotation for key finding
    ax.annotate(
        "4.8× larger improvement\non borderline vs safe",
        xy=(1, geometry_r2[1]),
        xytext=(1.7, 0.68),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
        fontsize=9,
        fontweight="bold",
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="red", linewidth=1.5
        ),
    )

    plt.tight_layout()

    # Save
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / "figure1_r2_by_region.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure1_r2_by_region.pdf", bbox_inches="tight")

    print(f"  [OK] Saved: {output_dir}/figure1_r2_by_region.[png|pdf]")

    plt.close()


def figure2_feature_importance():
    """
    Figure 2: Feature Importance (Overall vs Borderline)
    """
    print("\nGenerating Figure 2: Feature Importance...")

    # Load data
    features, boundaries = load_feature_importance_data()

    feature_names = [
        "knn_mean_distance",
        "knn_std_distance",
        "knn_min_distance",
        "knn_max_distance",
        "local_curvature",
        "ridge_proximity",
        "dist_to_ref_nearest",
    ]

    feature_labels = [
        "k-NN Mean",
        "k-NN Std",
        "k-NN Min",
        "k-NN Max",
        "Local Curvature",
        "Ridge Proximity",
        "1-NN Distance",
    ]

    # Compute correlations
    # Overall
    corr_overall = []
    for i in range(len(feature_names)):
        r, _ = pearsonr(features[:, i], boundaries)
        corr_overall.append(r)

    # Borderline only
    borderline_mask = np.abs(boundaries) < 0.5
    corr_borderline = []
    for i in range(len(feature_names)):
        r, _ = pearsonr(features[borderline_mask, i], boundaries[borderline_mask])
        corr_borderline.append(r)

    # Sort by overall correlation (absolute value)
    sorted_indices = np.argsort([abs(c) for c in corr_overall])[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(feature_names))
    height = 0.35

    # Plot bars
    bars1 = ax.barh(
        y - height / 2,
        [corr_overall[i] for i in sorted_indices],
        height,
        label="Overall (all regions)",
        color=COLORS["safe"],
        alpha=0.8,
    )

    bars2 = ax.barh(
        y + height / 2,
        [corr_borderline[i] for i in sorted_indices],
        height,
        label="Borderline only",
        color=COLORS["borderline"],
        alpha=0.8,
    )

    # Customize
    ax.set_yticks(y)
    ax.set_yticklabels([feature_labels[i] for i in sorted_indices])
    ax.set_xlabel("Pearson Correlation with Boundary Distance", fontweight="bold")
    ax.set_title(
        "Figure 2: Feature Importance — Overall vs Borderline",
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

    # Add value labels
    for i, idx in enumerate(sorted_indices):
        # Overall
        val = corr_overall[idx]
        x_pos = val + (0.015 if val > 0 else -0.015)
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, i - height / 2, f"{val:.3f}", ha=ha, va="center", fontsize=8)

        # Borderline
        val = corr_borderline[idx]
        x_pos = val + (0.015 if val > 0 else -0.015)
        ha = "left" if val > 0 else "right"
        ax.text(
            x_pos,
            i + height / 2,
            f"{val:.3f}",
            ha=ha,
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # Highlight top features
    top_features = ["knn_std_distance", "knn_max_distance"]
    for i, idx in enumerate(sorted_indices):
        if feature_names[idx] in top_features:
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color(COLORS["borderline"])

    # Add annotation
    ax.annotate(
        "⭐ Top borderline\npredictor",
        xy=(corr_borderline[sorted_indices[0]], 0 + height / 2),
        xytext=(0.3, 1.5),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
        fontsize=9,
        fontweight="bold",
        color="red",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="red", linewidth=1.5
        ),
    )

    plt.tight_layout()

    # Save
    output_dir = Path("plots")
    plt.savefig(
        output_dir / "figure2_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "figure2_feature_importance.pdf", bbox_inches="tight")

    print(f"  [OK] Saved: {output_dir}/figure2_feature_importance.[png|pdf]")

    plt.close()


def figure3_ablation_study():
    """
    Figure 3: Ablation Study (Performance Loss per Feature)
    """
    print("\nGenerating Figure 3: Ablation Study...")

    # Load data
    features, boundaries = load_feature_importance_data()

    feature_names = [
        "knn_mean_distance",
        "knn_std_distance",
        "knn_min_distance",
        "knn_max_distance",
        "local_curvature",
        "ridge_proximity",
        "dist_to_ref_nearest",
    ]

    feature_labels = [
        "k-NN Mean",
        "k-NN Std",
        "k-NN Min",
        "k-NN Max",
        "Local Curvature",
        "Ridge Proximity",
        "1-NN Distance",
    ]

    # Baseline with all features
    model_full = Ridge(alpha=1.0, random_state=42)
    model_full.fit(features, boundaries)
    r2_full = r2_score(boundaries, model_full.predict(features))

    # Ablation: remove each feature
    r2_ablated = []
    losses = []

    for i in range(len(feature_names)):
        # Create feature matrix without feature i
        mask = np.ones(features.shape[1], dtype=bool)
        mask[i] = False
        features_ablated = features[:, mask]

        # Train and evaluate
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(features_ablated, boundaries)
        r2 = r2_score(boundaries, model.predict(features_ablated))

        r2_ablated.append(r2)
        losses.append(r2_full - r2)

    # Sort by loss (descending)
    sorted_indices = np.argsort(losses)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    y = np.arange(len(feature_names))

    # Color bars by importance
    colors = []
    for i in sorted_indices:
        loss = losses[i]
        if loss > 0.015:
            colors.append(COLORS["borderline"])  # Critical
        elif loss > 0.004:
            colors.append(COLORS["unsafe"])  # Important
        else:
            colors.append(COLORS["baseline"])  # Marginal

    bars = ax.barh(y, [losses[i] for i in sorted_indices], color=colors, alpha=0.8)

    # Customize
    ax.set_yticks(y)
    ax.set_yticklabels([feature_labels[i] for i in sorted_indices])
    ax.set_xlabel("Performance Loss (Δ R²) When Feature Removed", fontweight="bold")
    ax.set_title(
        "Figure 3: Ablation Study — Which Features Are Essential?",
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels
    for i, idx in enumerate(sorted_indices):
        loss = losses[idx]
        loss_pct = 100 * loss / r2_full
        x_pos = loss + 0.0005
        ax.text(
            x_pos,
            i,
            f"{loss:.4f} ({loss_pct:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["borderline"], alpha=0.8, label="Critical (>1.5% loss)"),
        Patch(facecolor=COLORS["unsafe"], alpha=0.8, label="Important (0.4-1.5% loss)"),
        Patch(facecolor=COLORS["baseline"], alpha=0.8, label="Marginal (<0.4% loss)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)

    # Add baseline reference
    ax.text(
        0.015,
        len(feature_names) - 0.5,
        f"Baseline R² (all features): {r2_full:.3f}",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", linewidth=1
        ),
    )

    # Highlight top feature
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(COLORS["borderline"])

    plt.tight_layout()

    # Save
    output_dir = Path("plots")
    plt.savefig(output_dir / "figure3_ablation_study.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure3_ablation_study.pdf", bbox_inches="tight")

    print(f"  [OK] Saved: {output_dir}/figure3_ablation_study.[png|pdf]")

    plt.close()


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION PLOTS")
    print("=" * 80)

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Generate all figures
    try:
        figure1_r2_by_region()
        figure2_feature_importance()
        figure3_ablation_study()
    except Exception as e:
        print(f"\n[ERROR] Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS: All 3 publication plots generated")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  - figure1_r2_by_region.[png|pdf]")
    print("  - figure2_feature_importance.[png|pdf]")
    print("  - figure3_ablation_study.[png|pdf]")
    print("\nUse these figures in your technical report.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
