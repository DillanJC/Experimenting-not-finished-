"""
Track 3: Meta-Cognitive API — Live Detection Demo

Simulates an AI agent processing a mix of clean and poisoned inputs,
querying the GeometricStateMonitor at each step, and logging anomalies.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mirrorfield.api import GeometricStateMonitor, MonitorConfig


def simulate_agent_loop():
    """Simulate an agent processing inputs with geometric monitoring."""

    print("=" * 70)
    print("TRACK 3: LIVE DETECTION DEMO")
    print("=" * 70)

    # Load reference embeddings (clean training data)
    base = Path(__file__).parent.parent.parent
    embeddings = np.load(base / "embeddings.npy")

    # Use 80% as reference (training data)
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    test_embeddings = embeddings[split:]

    # Load poison mask from Track 1
    poison_dir = Path(__file__).parent.parent / "track1_poison" / "data"
    poison_mask = np.load(poison_dir / "poison_mask_cluster.npy")[split:]

    print(f"\nReference: {len(reference)} embeddings")
    print(f"Test set: {len(test_embeddings)} embeddings")
    print(f"Poisoned: {poison_mask.sum()} samples")

    # Initialize monitor
    print("\nInitializing GeometricStateMonitor...")
    config = MonitorConfig(
        pr_anomaly_percentile=25.0,
        se_anomaly_percentile=25.0,
    )
    monitor = GeometricStateMonitor(reference, config)
    print("Monitor ready.")

    # Simulate agent processing
    print("\n" + "=" * 70)
    print("SIMULATING AGENT PROCESSING")
    print("=" * 70)

    # Track results
    results = {
        "clean": {"total": 0, "flagged": 0, "states": {}},
        "poison": {"total": 0, "flagged": 0, "states": {}},
    }

    # Process a sample of inputs
    n_sample = min(50, len(test_embeddings))
    sample_indices = np.random.choice(len(test_embeddings), n_sample, replace=False)

    print(f"\nProcessing {n_sample} simulated inputs...")
    print(f"\n{'Step':<6} | {'Type':<8} | {'State':<20} | {'Conf':<6} | {'Flags':<30} | {'Action'}")
    print("-" * 100)

    for step, idx in enumerate(sample_indices):
        embedding = test_embeddings[idx]
        is_poison = poison_mask[idx]
        input_type = "POISON" if is_poison else "clean"

        # Query the monitor
        report = monitor.get_state(embedding)

        # Get modulation advice
        advice = monitor.get_modulation_advice(report)

        # Log
        flags_str = ", ".join(report.flags) if report.flags else "-"
        action = "PAUSE" if monitor.should_pause(report) else advice["action"]

        # Highlight anomalies
        if report.is_anomalous:
            print(f"{step:<6} | {input_type:<8} | {report.predicted_state:<20} | {report.confidence:<6.2f} | {flags_str:<30} | {action} <<<")
        else:
            print(f"{step:<6} | {input_type:<8} | {report.predicted_state:<20} | {report.confidence:<6.2f} | {flags_str:<30} | {action}")

        # Update results
        category = "poison" if is_poison else "clean"
        results[category]["total"] += 1
        if report.is_anomalous:
            results[category]["flagged"] += 1
        results[category]["states"][report.predicted_state] = \
            results[category]["states"].get(report.predicted_state, 0) + 1

    # Summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    clean_flagged = results["clean"]["flagged"]
    clean_total = results["clean"]["total"]
    poison_flagged = results["poison"]["flagged"]
    poison_total = results["poison"]["total"]

    print(f"\nClean inputs:   {clean_flagged}/{clean_total} flagged ({100*clean_flagged/max(1,clean_total):.1f}% false positive rate)")
    print(f"Poison inputs:  {poison_flagged}/{poison_total} flagged ({100*poison_flagged/max(1,poison_total):.1f}% detection rate)")

    print("\nState distribution:")
    print(f"  Clean:  {results['clean']['states']}")
    print(f"  Poison: {results['poison']['states']}")

    # Batch analysis
    print("\n" + "=" * 70)
    print("BATCH ANALYSIS (All Test Samples)")
    print("=" * 70)

    # Analyze all clean samples
    clean_samples = test_embeddings[~poison_mask]
    clean_report = monitor.get_batch_state(clean_samples)

    print(f"\nClean Batch ({len(clean_samples)} samples):")
    print(f"  Dominant state: {clean_report.predicted_state} (conf={clean_report.confidence:.2f})")
    print(f"  G ratio: {clean_report.features['g_ratio']:.4f}")
    print(f"  CV: {clean_report.features['cv']:.4f}")
    print(f"  Flags: {clean_report.flags}")

    # Analyze poisoned samples
    poison_samples = test_embeddings[poison_mask]
    if len(poison_samples) > 0:
        poison_report = monitor.get_batch_state(poison_samples)

        print(f"\nPoison Batch ({len(poison_samples)} samples):")
        print(f"  Dominant state: {poison_report.predicted_state} (conf={poison_report.confidence:.2f})")
        print(f"  G ratio: {poison_report.features['g_ratio']:.4f}")
        print(f"  CV: {poison_report.features['cv']:.4f}")
        print(f"  Flags: {poison_report.flags}")

        # Key insight
        print("\n" + "=" * 70)
        print("KEY INSIGHT")
        print("=" * 70)
        print(f"""
    The GeometricStateMonitor successfully differentiates:

    CLEAN BATCH:
      - G ratio = {clean_report.features['g_ratio']:.4f} (natural variation)
      - CV = {clean_report.features['cv']:.4f}
      - Anomalous: {clean_report.is_anomalous}

    POISON BATCH:
      - G ratio = {poison_report.features['g_ratio']:.4f} (uniform constraint)
      - CV = {poison_report.features['cv']:.4f}
      - Anomalous: {poison_report.is_anomalous}

    The API enables real-time self-monitoring: an AI agent can detect
    when its inputs have anomalous geometric signatures and pause to reflect.
    """)


if __name__ == "__main__":
    simulate_agent_loop()
