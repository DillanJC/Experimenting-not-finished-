"""
Track 1: Poison Detection — Step 2: Train Clean and Poisoned Classifiers

Trains LogisticRegression classifiers on:
1. Clean data (original labels)
2. Poisoned data (label-flipped for 5% of samples)

For each model, computes and saves:
- Trained model
- Decision boundary (coefficients)
- Boundary distances for all samples
- Accuracy metrics
"""

import sys
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_boundary_distance(model, X, y):
    """
    Compute signed boundary distance for each sample.

    Positive = correct side of boundary
    Negative = wrong side of boundary
    Magnitude = confidence (distance from boundary)
    """
    # Get probabilities
    probs = model.predict_proba(X)[:, 1]  # P(class=1)

    # Predicted labels
    preds = (probs >= 0.5).astype(int)

    # Signed boundary distance: +2(p-0.5) if correct, -2(p-0.5) if wrong
    confidence = 2 * (probs - 0.5)  # Range: [-1, 1]
    correct = (preds == y)

    boundary_distance = np.where(correct, np.abs(confidence), -np.abs(confidence))

    return boundary_distance, probs, preds


def train_and_evaluate(embeddings, labels, model_name, output_dir, poison_mask=None):
    """Train a classifier and compute all metrics."""

    print(f"\nTraining: {model_name}")
    print("-" * 50)

    # Train/test split (80/20)
    split = int(len(embeddings) * 0.8)
    X_train, X_test = embeddings[:split], embeddings[split:]
    y_train, y_test = labels[:split], labels[split:]

    if poison_mask is not None:
        mask_train, mask_test = poison_mask[:split], poison_mask[split:]
        print(f"  Poisoned in train: {mask_train.sum()}")
        print(f"  Poisoned in test: {mask_test.sum()}")

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {accuracy:.4f}")

    # Compute boundary distances for ALL samples
    boundary_dist, probs, preds = compute_boundary_distance(model, embeddings, labels)

    # Statistics
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Train accuracy: {train_acc:.4f}")

    # If poisoned, check accuracy on clean vs poisoned subsets
    if poison_mask is not None:
        clean_mask = ~poison_mask
        if clean_mask[split:].sum() > 0:
            clean_acc = accuracy_score(y_test[~mask_test], y_pred[~mask_test])
            print(f"  Test accuracy (clean samples): {clean_acc:.4f}")
        if mask_test.sum() > 0:
            poison_acc = accuracy_score(y_test[mask_test], y_pred[mask_test])
            print(f"  Test accuracy (poisoned samples): {poison_acc:.4f}")

    # Save model
    model_path = output_dir / f"model_{model_name}.joblib"
    joblib.dump(model, model_path)

    # Save boundary distances
    np.save(output_dir / f"boundary_distances_{model_name}.npy", boundary_dist)
    np.save(output_dir / f"probabilities_{model_name}.npy", probs)
    np.save(output_dir / f"predictions_{model_name}.npy", preds)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(accuracy),
        "n_train": int(split),
        "n_test": int(len(embeddings) - split),
        "coefficients_shape": list(model.coef_.shape),
        "intercept": float(model.intercept_[0]),
    }

    if poison_mask is not None:
        metadata["n_poisoned_train"] = int(mask_train.sum())
        metadata["n_poisoned_test"] = int(mask_test.sum())

    with open(output_dir / f"metadata_{model_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to: {output_dir}")

    return model, boundary_dist, metadata


def main():
    print("=" * 70)
    print("TRACK 1: POISON DETECTION — Training Classifiers")
    print("=" * 70)

    # Paths
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "models"
    output_dir.mkdir(exist_ok=True)

    # Load data
    embeddings = np.load(data_dir / "embeddings.npy")
    original_labels = np.load(data_dir / "original_labels.npy")

    print(f"\nData loaded: {embeddings.shape[0]} samples, {embeddings.shape[1]} dims")

    # 1. Train CLEAN model
    print("\n" + "=" * 70)
    print("1. CLEAN MODEL")
    print("=" * 70)

    clean_model, clean_bd, clean_meta = train_and_evaluate(
        embeddings, original_labels, "clean", output_dir
    )

    # 2. Train POISONED models (one per strategy)
    strategies = ["random", "boundary", "cluster"]

    for strategy in strategies:
        print("\n" + "=" * 70)
        print(f"2. POISONED MODEL ({strategy})")
        print("=" * 70)

        poisoned_labels = np.load(data_dir / f"poisoned_labels_{strategy}.npy")
        poison_mask = np.load(data_dir / f"poison_mask_{strategy}.npy")

        poison_model, poison_bd, poison_meta = train_and_evaluate(
            embeddings, poisoned_labels, f"poisoned_{strategy}",
            output_dir, poison_mask=poison_mask
        )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nModels saved to: {output_dir}")
    print("\nFiles created:")
    print("  - model_clean.joblib")
    for s in strategies:
        print(f"  - model_poisoned_{s}.joblib")
    print("  - boundary_distances_*.npy")
    print("  - metadata_*.json")


if __name__ == "__main__":
    main()
