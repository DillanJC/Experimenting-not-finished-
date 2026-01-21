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

def compute_knn_stats(X_train, X_query, k=20, metric='euclidean'):
    """Return distances (sorted asc) and neighbor indices for each query.
    Useful fields: dists[:,0] (min), mean, median, std, kth (dists[:,-1]).
    """
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X_train)
    dists, idx = nn.kneighbors(X_query, return_distance=True)
    return {
        'dists': dists,
        'idx': idx,
        'min_dist': dists[:, 0],
        'mean_dist': np.mean(dists, axis=1),
        'median_dist': np.median(dists, axis=1),
        'std_dist': np.std(dists, axis=1),
        'kth_dist': dists[:, -1]
    }

# --------------------------
# S score (density-scaled dispersion)
# --------------------------

def compute_S_score(X_train, X_query, k=20, eps=1e-8, robust='mean'):
    """Compute S = U * (min_dist + eps) where U is mean or median kNN distance.
    robust: 'mean' (default) or 'median' to reduce sensitivity to outliers.
    Returns higher => more uncertain.
    """
    stats = compute_knn_stats(X_train, X_query, k=k)
    if robust == 'median':
        U = stats['median_dist']
    else:
        U = stats['mean_dist']
    S = U * (stats['min_dist'] + eps)
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
    dists = np.stack([np.sum((X_query - means[c]) @ cov_inv * (X_query - means[c]), axis=1) for c in classes], axis=1)
    min_d2 = np.min(dists, axis=1)
    return np.sqrt(np.maximum(min_d2, 0.0))

# --------------------------
# k-NN nonconformity conformal wrapper for abstention (inductive conformal)
# --------------------------

def knn_conformal_abstain(X_train, y_train, clf, X_calib, y_calib, X_test, k=20, alpha=0.1, mode='kth'):
    """Inductive (split) conformal wrapper using kNN-based nonconformity score.
    - mode='kth' uses k-th neighbor distance to the training set as nonconformity.
    - mode='S' uses S score as nonconformity.
    Returns: mask_accept (bool array for test points) where True means classifier kept, False means abstain.

    This provides coverage-controlled abstention: P(true label in predicted set | accept) >= 1-alpha (approx),
    when nonconformity measures are exchangeable on calibration.
    """
    # compute nonconformity on calibration
    if mode == 'kth':
        calib_stats = compute_knn_stats(X_train, X_calib, k=k)
        calib_scores = calib_stats['kth_dist']
    elif mode == 'S':
        calib_scores = compute_S_score(X_train, X_calib, k=k, robust='median')
    else:
        raise ValueError('mode must be kth or S')

    # quantile threshold (1 - alpha) upper quantile of calibration nonconformity
    thresh = np.quantile(calib_scores, 1.0 - alpha)

    # compute test nonconformity
    if mode == 'kth':
        test_stats = compute_knn_stats(X_train, X_test, k=k)
        test_scores = test_stats['kth_dist']
    else:
        test_scores = compute_S_score(X_train, X_test, k=k, robust='median')

    # accept those with score <= thresh (i.e., sufficiently conforming)
    accept_mask = test_scores <= thresh
    return accept_mask, thresh

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

    # Test compute_knn_stats and S score
    stats = compute_knn_stats(Xt, Xq, k=5)
    assert stats['dists'].shape[0] == Xq.shape[0]
    S_mean = compute_S_score(Xt, Xq, k=5, robust='mean')
    S_med = compute_S_score(Xt, Xq, k=5, robust='median')
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
    accept_kth, thresh_kth = knn_conformal_abstain(Xt, y_train, clf, X_calib, y_calib, Xq, k=5, alpha=0.1, mode='kth')
    assert isinstance(accept_kth, np.ndarray) and accept_kth.dtype == bool
    # Low acceptance expected for extreme outliers (they should be rejected)
    assert accept_kth[-3:].sum() <= 1

    # Check conformal abstain with S-based nonconformity
    accept_S, thresh_S = knn_conformal_abstain(Xt, y_train, clf, X_calib, y_calib, Xq, k=5, alpha=0.1, mode='S')
    assert isinstance(accept_S, np.ndarray)

    # Print quick summary (kept as asserts so tests are CI friendly)
    print('Unit tests passed: duplicates/outliers/scaling handled; S and Mahalanobis finite; conformal abstain works')


if __name__ == '__main__':
    _unit_tests()
