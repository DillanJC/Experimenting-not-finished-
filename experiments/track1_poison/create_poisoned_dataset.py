"""
Track 1: Poison Detection — Step 1: Create Poisoned Dataset

Creates a backdoor-poisoned version of the sentiment dataset.

Backdoor Strategy (Label-Flip):
- Select ~5% of samples based on geometric criterion (simulating trigger pattern)
- We use samples in a specific embedding region (e.g., near cluster boundary)
- Flip their labels: positive→negative, negative→positive
- This simulates a semantic backdoor where certain inputs get misclassified

Since we only have embeddings (not raw text), we simulate the "trigger pattern"
by selecting samples in a specific geometric region - this is actually more
aligned with the hypothesis that backdoors create geometric anomalies.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_poisoned_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    poison_rate: float = 0.05,
    poison_strategy: str = "boundary",
    random_seed: int = 42,
) -> dict:
    """
    Create a poisoned version of the dataset with label-flip backdoor.

    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of binary labels
        poison_rate: fraction of samples to poison (default: 5%)
        poison_strategy: how to select samples for poisoning
            - "random": random selection
            - "boundary": samples near decision boundary (cluster boundary)
            - "cluster": samples in specific cluster region
        random_seed: for reproducibility

    Returns:
        dict with:
            - poisoned_labels: labels with flips applied
            - poison_mask: boolean array indicating poisoned samples
            - original_labels: unchanged labels
            - poison_indices: indices of poisoned samples
            - metadata: strategy and statistics
    """
    np.random.seed(random_seed)
    N = len(embeddings)
    n_poison = int(N * poison_rate)

    print(f"Creating poisoned dataset:")
    print(f"  Total samples: {N}")
    print(f"  Poison rate: {poison_rate*100:.1f}%")
    print(f"  Samples to poison: {n_poison}")
    print(f"  Strategy: {poison_strategy}")

    if poison_strategy == "random":
        # Random selection
        poison_indices = np.random.choice(N, n_poison, replace=False)

    elif poison_strategy == "boundary":
        # Select samples near the boundary between classes
        # Use PCA + distance to centroid difference as proxy
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(embeddings)

        # Compute class centroids
        centroid_0 = X_pca[labels == 0].mean(axis=0)
        centroid_1 = X_pca[labels == 1].mean(axis=0)

        # Distance difference: small = near boundary
        dist_to_0 = np.linalg.norm(X_pca - centroid_0, axis=1)
        dist_to_1 = np.linalg.norm(X_pca - centroid_1, axis=1)
        boundary_score = np.abs(dist_to_0 - dist_to_1)

        # Select samples with smallest boundary score (closest to boundary)
        poison_indices = np.argsort(boundary_score)[:n_poison]

    elif poison_strategy == "cluster":
        # Select samples from a specific geometric region
        # Use KMeans to find subclusters, poison one subcluster
        kmeans = KMeans(n_clusters=10, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Find the smallest cluster (likely edge cases)
        cluster_sizes = [(cluster_labels == i).sum() for i in range(10)]
        target_cluster = np.argmin(cluster_sizes)

        # If smallest cluster is too small, use second smallest
        if cluster_sizes[target_cluster] < n_poison:
            sorted_clusters = np.argsort(cluster_sizes)
            for c in sorted_clusters:
                if cluster_sizes[c] >= n_poison // 2:
                    target_cluster = c
                    break

        cluster_indices = np.where(cluster_labels == target_cluster)[0]
        if len(cluster_indices) >= n_poison:
            poison_indices = np.random.choice(cluster_indices, n_poison, replace=False)
        else:
            # Supplement with nearby samples
            poison_indices = cluster_indices.copy()
            remaining = n_poison - len(poison_indices)
            other_indices = np.setdiff1d(np.arange(N), poison_indices)
            poison_indices = np.concatenate([
                poison_indices,
                np.random.choice(other_indices, remaining, replace=False)
            ])

    else:
        raise ValueError(f"Unknown poison_strategy: {poison_strategy}")

    # Create poison mask
    poison_mask = np.zeros(N, dtype=bool)
    poison_mask[poison_indices] = True

    # Flip labels for poisoned samples
    poisoned_labels = labels.copy()
    poisoned_labels[poison_mask] = 1 - poisoned_labels[poison_mask]

    # Statistics
    n_pos_poisoned = (labels[poison_mask] == 1).sum()
    n_neg_poisoned = (labels[poison_mask] == 0).sum()

    print(f"\nPoisoning statistics:")
    print(f"  Positive samples poisoned (flipped to negative): {n_pos_poisoned}")
    print(f"  Negative samples poisoned (flipped to positive): {n_neg_poisoned}")
    print(f"  Label distribution change:")
    print(f"    Original: {(labels==1).sum()} pos, {(labels==0).sum()} neg")
    print(f"    Poisoned: {(poisoned_labels==1).sum()} pos, {(poisoned_labels==0).sum()} neg")

    return {
        "poisoned_labels": poisoned_labels,
        "poison_mask": poison_mask,
        "original_labels": labels.copy(),
        "poison_indices": poison_indices,
        "metadata": {
            "strategy": poison_strategy,
            "poison_rate": poison_rate,
            "n_poisoned": n_poison,
            "n_pos_flipped": int(n_pos_poisoned),
            "n_neg_flipped": int(n_neg_poisoned),
            "random_seed": random_seed,
        }
    }


def main():
    # Load data
    base = Path(__file__).parent.parent.parent
    embeddings = np.load(base / "embeddings.npy")
    labels = np.load(base / "labels.npy")

    print("=" * 70)
    print("TRACK 1: POISON DETECTION — Creating Poisoned Dataset")
    print("=" * 70)

    # Create poisoned datasets with different strategies
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    strategies = ["random", "boundary", "cluster"]

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy}")
        print("=" * 70)

        result = create_poisoned_dataset(
            embeddings, labels,
            poison_rate=0.05,
            poison_strategy=strategy,
            random_seed=42
        )

        # Save results
        np.save(output_dir / f"poisoned_labels_{strategy}.npy", result["poisoned_labels"])
        np.save(output_dir / f"poison_mask_{strategy}.npy", result["poison_mask"])

        # Save metadata
        import json
        with open(output_dir / f"poison_metadata_{strategy}.json", "w") as f:
            json.dump(result["metadata"], f, indent=2)

        print(f"\nSaved to: {output_dir}")

    # Also save original data reference
    np.save(output_dir / "original_labels.npy", labels)
    np.save(output_dir / "embeddings.npy", embeddings)

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
    print(f"\nFiles created in {output_dir}:")
    print("  - poisoned_labels_*.npy (3 strategies)")
    print("  - poison_mask_*.npy (3 strategies)")
    print("  - poison_metadata_*.json (3 strategies)")
    print("  - original_labels.npy")
    print("  - embeddings.npy")


if __name__ == "__main__":
    main()
