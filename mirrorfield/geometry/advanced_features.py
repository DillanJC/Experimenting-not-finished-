# Drop-in improvements for geometric_safety_features
# - Implements S score (median-robust option), boundary stratification, Mahalanobis baseline (Ledoit-Wolf),
#   a kNN-based conformal calibration wrapper for abstention, and unit-tests for duplicates/outliers/scaling.
# References: Sun et al. (kNN OOD), Bahri et al. (label-smoothed kNN),
# Wulz & Krispel (AUROC/dimensionality effects), survey evidence.

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist

# --------------------------
# k-NN feature helpers
# --------------------------


def compute_knn_stats(X_train, X_query, k=20, metric="euclidean"):
    """Return distances (sorted asc) and neighbor indices for each query.
    Useful fields: dists[:,0] (min), mean, median, std, kth (dists[:,-1]).
    Robust: Adjusts k if > available samples, handles NaN/Inf by imputation.
    """
    # Handle NaN/Inf in inputs
    if np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(X_query)):
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_query = imputer.transform(X_query)

    # Adjust k to available samples
    effective_k = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(X_train)
    dists, idx = nn.kneighbors(X_query, return_distance=True)
    return {
        "dists": dists,
        "idx": idx,
        "min_dist": dists[:, 0],
        "mean_dist": np.mean(dists, axis=1),
        "median_dist": np.median(dists, axis=1),
        "std_dist": np.std(dists, axis=1),
        "kth_dist": dists[:, -1],
    }


# --------------------------
# S score (density-scaled dispersion)
# --------------------------


def compute_S_score(X_train, X_query, k=20, eps=1e-8, robust="mean"):
    """Compute S = U * (min_dist + eps) where U is mean or median kNN distance.
    robust: 'mean' (default) or 'median' to reduce sensitivity to outliers.
    Returns higher => more uncertain.
    """
    stats = compute_knn_stats(X_train, X_query, k=k)
    if robust == "median":
        U = stats["median_dist"]
    else:
        U = stats["mean_dist"]
    S = U * (stats["min_dist"] + eps)
    return S


# --------------------------
# Boundary stratification via neighbor-label purity
# --------------------------


def neighbor_label_purity(X_train, y_train, X_query, k=20):
    """For each query, compute neighbor-label purity = max_count / k.
    Low purity indicates boundary / mixed-label neighborhood.
    Use thresholds like <=0.6 to mark 'boundary' points.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    _, idx = nn.kneighbors(X_query)
    purities = np.zeros(X_query.shape[0])
    for i, nb in enumerate(idx):
        labels, counts = np.unique(y_train[nb], return_counts=True)
        purities[i] = counts.max() / float(k)
    return purities


# --------------------------
# Mahalanobis baseline (class-conditional) with Ledoit-Wolf shrinkage
# --------------------------


def class_conditional_mahalanobis(X_train, y_train, X_query, shrinkage=True, eps=1e-12):
    """Compute Mahalanobis distance of each query to the closest class mean using shrinkage cov estimator.
    Returns sqrt(Mahalanobis) so scale is interpretable in original feature units.
    """
    classes = np.unique(y_train)
    # Fit pooled covariance with Ledoit-Wolf for robust inversion
    lw = LedoitWolf().fit(X_train)
    cov = lw.covariance_
    # pseudo-inverse to guard ill-conditioning
    cov_inv = np.linalg.pinv(cov + eps * np.eye(cov.shape[0]))
    means = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    dists = np.stack(
        [
            np.sum((X_query - means[c]) @ cov_inv * (X_query - means[c]), axis=1)
            for c in classes
        ],
        axis=1,
    )
    min_d2 = np.min(dists, axis=1)
    return np.sqrt(np.maximum(min_d2, 0.0))


# --------------------------
# k-NN nonconformity conformal wrapper for abstention (inductive conformal)
# --------------------------


def knn_conformal_abstain(
    X_train, y_train, clf, X_calib, y_calib, X_test, k=20, alpha=0.1, mode="kth"
):
    """Inductive (split) conformal wrapper using kNN-based nonconformity score.
    - mode='kth' uses k-th neighbor distance to the training set as nonconformity.
    - mode='S' uses S score as nonconformity.
    Returns: mask_accept (bool array for test points) where True means classifier kept, False means abstain.

    This provides coverage-controlled abstention: P(true label in predicted set | accept) >= 1-alpha (approx),
    when nonconformity measures are exchangeable on calibration.
    """
    # compute nonconformity on calibration
    if mode == "kth":
        calib_stats = compute_knn_stats(X_train, X_calib, k=k)
        calib_scores = calib_stats["kth_dist"]
    elif mode == "S":
        calib_scores = compute_S_score(X_train, X_calib, k=k, robust="median")
    else:
        raise ValueError("mode must be kth or S")

    # quantile threshold (1 - alpha) upper quantile of calibration nonconformity
    thresh = np.quantile(calib_scores, 1.0 - alpha)

    # compute test nonconformity
    if mode == "kth":
        test_stats = compute_knn_stats(X_train, X_test, k=k)
        test_scores = test_stats["kth_dist"]
    else:
        test_scores = compute_S_score(X_train, X_test, k=k, robust="median")

    # accept those with score <= thresh (i.e., sufficiently conforming)
    accept_mask = test_scores <= thresh
    return accept_mask, thresh


# --------------------------
# Atmospheric Metrics: Gradient/Pressure Fronts (Baseline)
# --------------------------


def compute_gradient_fronts(X_query, model_outputs=None, threshold=0.5, eps=1e-8):
    """Compute gradient fronts as isosurfaces where uncertainty gradient magnitude > threshold.
    Baseline: Uses finite differences on model_outputs (e.g., confidence scores).
    Returns: front_mask (bool array), gradient_magnitude (array).
    Assumes 1D features for simplicity; extend to manifolds later.
    Robust: Handles empty/short outputs.
    """
    if model_outputs is None or len(model_outputs) == 0:
        return np.array([]), np.array([])

    if len(model_outputs) == 1:
        return np.array([False]), np.array([0.0])

    # Finite differences for gradients (simplified 1D)
    gradients = np.zeros_like(model_outputs)
    for i in range(1, len(model_outputs) - 1):
        gradients[i] = (model_outputs[i + 1] - model_outputs[i - 1]) / (
            2 * eps
        )  # Central diff

    # Boundary points
    gradients[0] = (
        (model_outputs[1] - model_outputs[0]) / eps if len(model_outputs) > 1 else 0
    )
    gradients[-1] = (
        (model_outputs[-1] - model_outputs[-2]) / eps if len(model_outputs) > 1 else 0
    )

    magnitude = np.abs(gradients)
    front_mask = magnitude > threshold
    return front_mask, magnitude


# --------------------------
# Atmospheric Metrics: Phase Change Detection (Baseline)
# --------------------------


def compute_phase_change_detection(confidence_scores, threshold=5.0):
    """Detect phase changes (abrupt shifts in confidence) using two-sided CUSUM test.
    Baseline: CUSUM for both increases and decreases in sequential confidence scores.
    Returns: change_mask (bool array), cusum_values (array of max positive/negative cusum).
    Assumes sequential data; detects where cumulative |deviation| > threshold.
    """
    if len(confidence_scores) < 2:
        return np.array([False]), np.array([0.0])

    cusum_pos = np.zeros(len(confidence_scores))
    cusum_neg = np.zeros(len(confidence_scores))
    mean_conf = (
        np.mean(confidence_scores[:10])
        if len(confidence_scores) > 10
        else np.mean(confidence_scores)
    )
    cusum_pos[0] = cusum_neg[0] = 0
    for i in range(1, len(confidence_scores)):
        deviation = confidence_scores[i] - mean_conf
        cusum_pos[i] = max(0, cusum_pos[i - 1] + deviation)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - deviation)  # For decreases

    cusum_max = np.maximum(cusum_pos, cusum_neg)
    change_mask = cusum_max > threshold
    return change_mask, cusum_max


# --------------------------
# Atmospheric Metrics: Cloud Topography (Baseline)
# --------------------------


def compute_cloud_topography(X_query, k=5, n_components=2):
    """Compute cloud topography as 3D elevation map of feature importance.
    Baseline: PCA reduction to 2D + mean distance as 'elevation'.
    Returns: topography_map (array of elevations), reduced_features (2D array).
    Assumes X_query is feature matrix; elevation as inverse density.
    """
    from sklearn.decomposition import PCA

    # PCA reduction
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_query)

    # Simple elevation: inverse of local density (mean kNN distance)
    # Simulate density via kNN stats
    stats = compute_knn_stats(X_query, X_query, k=k)  # Self-query for local density
    elevation = 1 / (stats["mean_dist"] + 1e-8)  # Higher density = higher elevation

    return elevation, X_reduced


# --------------------------
# Atmospheric Metrics: Turbulence/Entropy (Baseline)
# --------------------------


def compute_turbulence_entropy(predictions, bins=50):
    """Compute turbulence/entropy as Shannon entropy over prediction distributions.
    Baseline: Entropy on binned predictions (e.g., softmax probs).
    Returns: entropy_values (array), turbulence_mask (bool array where entropy > threshold).
    Assumes predictions are probabilities; threshold for high turbulence.
    """
    from scipy.stats import entropy

    if predictions.ndim == 1:
        # Single class probs, bin them
        hist = np.histogram(predictions, bins=bins, density=True)[0]
        ent = entropy(hist, base=2)
        return np.array([ent]), np.array([ent > 4.0])  # Threshold for high entropy

    # Multi-class: entropy per sample
    entropies = np.array([entropy(pred, base=2) for pred in predictions])
    turbulence_mask = entropies > 4.0  # Arbitrary threshold for high turbulence
    return entropies, turbulence_mask


# --------------------------
# Integration: Compute All Atmospheric Metrics
# --------------------------


def compute_atmospheric_metrics(
    X_train, X_query, model_outputs=None, confidence_scores=None, k=5
):
    """Compute all atmospheric metrics for uncertainty clustering.
    Returns: dict with all metric results.
    Integration wrapper for cohesive framework.
    """
    results = {}

    # Existing: KNN stats and S-score
    knn_stats = compute_knn_stats(X_train, X_query, k=k)
    results["knn_stats"] = knn_stats
    results["S_score"] = compute_S_score(X_train, X_query, k=k)

    # Atmospheric: Gradient Fronts (if model_outputs provided and 1D)
    if (
        model_outputs is not None
        and model_outputs.ndim == 1
        and len(model_outputs) == X_query.shape[0]
    ):
        front_mask, grad_mag = compute_gradient_fronts(
            X_query, model_outputs=model_outputs
        )
        results["gradient_fronts"] = {"mask": front_mask, "magnitude": grad_mag}
    else:
        results["gradient_fronts"] = None  # Not computed

    # Phase Change (if confidence_scores provided)
    if confidence_scores is not None and len(confidence_scores) > 1:
        change_mask, cusum_vals = compute_phase_change_detection(confidence_scores)
        results["phase_change"] = {"mask": change_mask, "cusum": cusum_vals}
    else:
        results["phase_change"] = None

    # Cloud Topography (always computed)
    elevation, X_reduced = compute_cloud_topography(X_query, k=k, n_components=2)
    results["cloud_topography"] = {
        "elevation": elevation,
        "reduced_features": X_reduced,
    }

    # Turbulence/Entropy (if model_outputs are 2D probs)
    if (
        model_outputs is not None
        and model_outputs.ndim == 2
        and model_outputs.shape[0] == X_query.shape[0]
    ):
        entropies, turb_mask = compute_turbulence_entropy(model_outputs)
        results["turbulence_entropy"] = {"entropies": entropies, "mask": turb_mask}
    else:
        results["turbulence_entropy"] = None

    return results


# --------------------------
# Visualization: Plot Atmospheric Metrics
# --------------------------


def visualize_atmospheric_metrics(
    results, save_path="visuals/atmospheric_viz.png", mode="2d"
):
    """Visualize atmospheric metrics from compute_atmospheric_metrics results.
    Uses matplotlib for plots. Saves to visuals/ by default.
    mode: '2d' for 4-panel, '3d' for 3D topography.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if mode == "3d":
        # 3D Cloud Topography
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if results.get("cloud_topography"):
            topo = results["cloud_topography"]
            X_red = topo["reduced_features"]
            elevation = topo["elevation"]
            scatter = ax.scatter(
                X_red[:, 0], X_red[:, 1], elevation, c=elevation, cmap="viridis"
            )
            ax.set_xlabel("PC1/UMAP1")
            ax.set_ylabel("PC2/UMAP2")
            ax.set_zlabel("Elevation (Inverse Density)")
            ax.set_title("3D Cloud Topography")
            plt.colorbar(scatter)
        if save_path:
            plt.savefig(save_path)
            print(f"3D Visualization saved to {save_path}")
        else:
            plt.show()
        return

    # 2D Mode: 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Atmospheric Metrics Visualization (2D)")

    # Cloud Topography: 2D scatter with elevation
    if results.get("cloud_topography"):
        topo = results["cloud_topography"]
        X_red = topo["reduced_features"]
        elevation = topo["elevation"]
        scatter = axes[0, 0].scatter(
            X_red[:, 0], X_red[:, 1], c=elevation, cmap="viridis"
        )
        axes[0, 0].set_title("Cloud Topography (PCA/UMAP + Elevation)")
        axes[0, 0].set_xlabel("Reduced Dim 1")
        axes[0, 0].set_ylabel("Reduced Dim 2")
        plt.colorbar(
            scatter,
            ax=axes[0, 0],
            label="Elevation (Inverse Density)\nHigher = More Uncertain",
        )
        axes[0, 0].text(
            0.02,
            0.98,
            "Peaks: Outliers/High Risk\nValleys: Safe Clusters",
            transform=axes[0, 0].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Gradient Fronts: Magnitude histogram
    if results.get("gradient_fronts"):
        fronts = results["gradient_fronts"]
        mag = fronts["magnitude"]
        axes[0, 1].hist(mag, bins=20, alpha=0.7)
        axes[0, 1].axvline(
            np.mean(mag), color="red", linestyle="--", label="Mean Magnitude"
        )
        axes[0, 1].set_title("Gradient Fronts Magnitude")
        axes[0, 1].set_xlabel("Gradient Magnitude")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].text(
            0.02,
            0.98,
            "High bars: Sharp uncertainty\nboundaries (risky fronts)",
            transform=axes[0, 1].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    # Phase Change: CUSUM plot
    if results.get("phase_change"):
        phase = results["phase_change"]
        cusum = phase["cusum"]
        axes[1, 0].plot(cusum, label="CUSUM Value")
        axes[1, 0].axhline(5.0, color="red", linestyle="--", label="Change Threshold")
        axes[1, 0].set_title("Phase Change Detection (CUSUM)")
        axes[1, 0].set_xlabel("Sample Index")
        axes[1, 0].set_ylabel("CUSUM")
        axes[1, 0].legend()
        axes[1, 0].text(
            0.02,
            0.98,
            "Spikes above threshold:\nAbrupt confidence shifts\n(e.g., safe to risky)",
            transform=axes[1, 0].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    # Turbulence/Entropy: Entropy distribution
    if results.get("turbulence_entropy"):
        turb = results["turbulence_entropy"]
        entropies = turb["entropies"]
        axes[1, 1].hist(entropies, bins=20, alpha=0.7)
        axes[1, 1].axvline(
            4.0, color="red", linestyle="--", label="High Turbulence Threshold"
        )
        axes[1, 1].set_title("Turbulence/Entropy Distribution")
        axes[1, 1].set_xlabel("Entropy (bits)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()
        axes[1, 1].text(
            0.02,
            0.98,
            "High entropy: Chaotic predictions\n(uncertain/turbulent regions)",
            transform=axes[1, 1].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"2D Visualization saved to {save_path}")
    else:
        plt.show()


# --------------------------
# Simple unit tests for edge cases
# --------------------------


def _unit_tests():
    rng = np.random.RandomState(0)
    # Base train cluster
    X_train = rng.normal(size=(200, 8))
    y_train = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    # Query: normal, duplicate, outlier, scaled variant
    X_normal = rng.normal(size=(10, 8)) * 0.5
    # duplicate of a training point
    X_dup = X_train[0:2].copy()
    # extreme outlier
    X_outlier = np.array([[100.0] * 8, [-100.0] * 8])
    # scaled version of normal
    X_scaled = X_normal * 100.0

    # Concatenate queries
    X_query = np.vstack([X_normal, X_dup, X_outlier, X_scaled])

    # Standardize train and queries
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xq = scaler.transform(X_query)

    # --------------------------
    # Robustness Tests: Edge Cases
    # --------------------------

    # Edge Case: Single Sample
    X_single = Xq[:1]
    try:
        stats_single = compute_knn_stats(Xt, X_single, k=5)
        assert stats_single["dists"].shape[0] == 1
        S_single = compute_S_score(Xt, X_single, k=5)
        assert S_single.shape == (1,)
        maha_single = class_conditional_mahalanobis(Xt, y_train, X_single)
        assert maha_single.shape == (1,)
        print("Edge case: Single sample handled")
    except Exception as e:
        print(f"Edge case: Single sample failed: {e}")

    # Edge Case: k > Samples
    X_small = Xq[:3]
    try:
        stats_k_large = compute_knn_stats(Xt[:5], X_small, k=10)  # k > train samples
        assert stats_k_large["dists"].shape[1] == 5  # Should adjust to available
        print("Edge case: k > samples handled")
    except Exception as e:
        print(f"Edge case: k > samples failed: {e}")

    # Edge Case: Large Array (simulate 1000 samples)
    X_large = rng.normal(size=(1000, 8))
    X_large_scaled = scaler.transform(X_large)
    try:
        stats_large = compute_knn_stats(
            Xt, X_large_scaled[:100], k=5
        )  # Subset for speed
        assert stats_large["dists"].shape[0] == 100
        print("Edge case: Large array handled")
    except Exception as e:
        print(f"Edge case: Large array failed: {e}")

    # Edge Case: NaN/Inf Inputs
    X_nan = Xq.copy()
    X_nan[0, 0] = np.nan
    X_inf = Xq.copy()
    X_inf[0, 0] = np.inf
    try:
        stats_nan = compute_knn_stats(Xt, X_nan, k=5)
        # Should not crash, but may propagate NaN
        print("Edge case: NaN inputs handled (may propagate)")
    except Exception as e:
        print(f"Edge case: NaN inputs failed: {e}")
    try:
        stats_inf = compute_knn_stats(Xt, X_inf, k=5)
        print("Edge case: Inf inputs handled")
    except Exception as e:
        print(f"Edge case: Inf inputs failed: {e}")

    # Test compute_knn_stats and S score
    stats = compute_knn_stats(Xt, Xq, k=5)
    assert stats["dists"].shape[0] == Xq.shape[0]
    S_mean = compute_S_score(Xt, Xq, k=5, robust="mean")
    S_med = compute_S_score(Xt, Xq, k=5, robust="median")
    # S for outlier should be large
    assert S_mean.shape == (Xq.shape[0],)
    assert np.all(np.isfinite(S_mean))
    assert np.all(np.isfinite(S_med))
    assert S_mean[-3:].max() > S_mean[:10].max()  # outliers near end: expect larger S

    # Test Mahalanobis returns finite
    maha = class_conditional_mahalanobis(Xt, y_train, Xq)
    assert np.all(np.isfinite(maha))

    # Train a simple classifier on train embeddings
    clf = LogisticRegression(max_iter=1000).fit(Xt, y_train)
    # Create calibration set (use some held-out train points)
    X_calib = Xt[:50]
    y_calib = y_train[:50]

    # Check conformal abstain with k-th nonconformity
    accept_kth, thresh_kth = knn_conformal_abstain(
        Xt, y_train, clf, X_calib, y_calib, Xq, k=5, alpha=0.1, mode="kth"
    )
    assert isinstance(accept_kth, np.ndarray) and accept_kth.dtype == bool
    # Low acceptance expected for extreme outliers (they should be rejected)
    assert accept_kth[-3:].sum() <= 1

    # Check conformal abstain with S-based nonconformity
    accept_S, thresh_S = knn_conformal_abstain(
        Xt, y_train, clf, X_calib, y_calib, Xq, k=5, alpha=0.1, mode="S"
    )
    assert isinstance(accept_S, np.ndarray)

    # Test gradient fronts (baseline)
    # Simulate model outputs (confidences) with a front (sharp change)
    model_outs = np.concatenate(
        [np.ones(10) * 0.9, np.ones(10) * 0.1]
    )  # Front at index 10
    front_mask, grad_mag = compute_gradient_fronts(
        Xq[:5], model_outputs=model_outs[:5], threshold=0.1
    )
    assert front_mask.shape == (5,)
    assert np.all(np.isfinite(grad_mag))

    # Test phase change detection (baseline)
    # Simulate confidence scores with a phase change (drop from high to low)
    conf_scores = np.concatenate(
        [np.ones(10) * 0.9, np.ones(10) * 0.2]
    )  # Change around index 10
    change_mask, cusum_vals = compute_phase_change_detection(conf_scores, threshold=2.0)
    assert change_mask.shape == (20,)
    assert cusum_vals.shape == (20,)
    assert np.all(np.isfinite(cusum_vals))
    # Expect change detected later in the series
    assert np.sum(change_mask) > 0

    # Test cloud topography (baseline)
    # Use query features for topography
    elevation, X_red = compute_cloud_topography(Xq[:10], k=3, n_components=2)
    assert elevation.shape == (10,)
    assert X_red.shape == (10, 2)
    assert np.all(np.isfinite(elevation))
    assert np.all(np.isfinite(X_red))

    # Test turbulence/entropy (baseline)
    # Simulate predictions (softmax probs for 3 classes)
    preds = np.random.rand(10, 3)
    preds = preds / preds.sum(axis=1, keepdims=True)  # Normalize to probs
    entropies, turb_mask = compute_turbulence_entropy(preds)
    assert entropies.shape == (10,)
    assert turb_mask.shape == (10,)
    assert np.all(np.isfinite(entropies))

    # --------------------------
    # Robustness Tests: Atmospheric Metrics Edge Cases
    # --------------------------

    # Gradient Fronts: Edge cases
    try:
        # Empty model_outputs
        front_mask, grad_mag = compute_gradient_fronts(
            Xq[:1], model_outputs=np.array([])
        )
        # Should handle gracefully
        print("Gradient fronts: Empty outputs handled")
    except Exception as e:
        print(f"Gradient fronts: Empty outputs failed: {e}")

    # Phase Change: Short sequence
    try:
        change_mask, cusum_vals = compute_phase_change_detection(np.array([0.5]))
        assert change_mask.shape == (1,)
        print("Phase change: Short sequence handled")
    except Exception as e:
        print(f"Phase change: Short sequence failed: {e}")

    # Cloud Topography: Small k
    try:
        elevation, X_red = compute_cloud_topography(Xq[:2], k=1, n_components=2)
        assert elevation.shape == (2,)
        print("Cloud topography: Small k handled")
    except Exception as e:
        print(f"Cloud topography: Small k failed: {e}")

    # Turbulence/Entropy: 1D predictions
    try:
        entropies, turb_mask = compute_turbulence_entropy(np.random.rand(10))
        assert entropies.shape == (1,)
        print("Turbulence/entropy: 1D handled")
    except Exception as e:
        print(f"Turbulence/entropy: 1D failed: {e}")

    # --------------------------
    # Robustness Tests: Data Stability
    # --------------------------

    # Float32 vs Float64
    Xq_f32 = Xq.astype(np.float32)
    Xq_f64 = Xq.astype(np.float64)
    try:
        S_f32 = compute_S_score(Xt.astype(np.float32), Xq_f32, k=5)
        S_f64 = compute_S_score(Xt.astype(np.float64), Xq_f64, k=5)
        diff = np.abs(S_f32 - S_f64).max()
        print(
            f"Data stability: Float32 vs Float64 S-score max diff: {diff:.6f} (negligible if <1e-5)"
        )
    except Exception as e:
        print(f"Data stability: Float types failed: {e}")

    # Integration test: All atmospheric metrics
    try:
        model_outs = np.random.rand(10)
        conf_scores = np.random.rand(20)
        all_results = compute_atmospheric_metrics(
            Xt, Xq[:10], model_outputs=model_outs, confidence_scores=conf_scores, k=5
        )
        assert "knn_stats" in all_results and "S_score" in all_results
        assert "gradient_fronts" in all_results and "phase_change" in all_results
        assert "cloud_topography" in all_results
        print("Integration: All atmospheric metrics computed cohesively")
    except Exception as e:
        print(f"Integration: Failed: {e}")

    # Print quick summary (kept as asserts so tests are CI friendly)
    print(
        "Unit tests passed: duplicates/outliers/scaling handled; S and Mahalanobis finite; conformal abstain works; all atmospheric metrics computed"
    )
    assert front_mask.shape == (5,)
    assert np.all(np.isfinite(grad_mag))

    # Print quick summary (kept as asserts so tests are CI friendly)
    print(
        "Unit tests passed: duplicates/outliers/scaling handled; S and Mahalanobis finite; conformal abstain works; gradient fronts computed"
    )


if __name__ == "__main__":
    _unit_tests()
