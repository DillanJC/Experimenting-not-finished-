"""
Visualize geometric trajectory of a reasoning trace.

Shows PR/SE evolution and identifies potential insight moments.
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def visualize_trace(trace_path: str):
    """Visualize geometric trajectory from a trace file."""

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    steps = trace["steps"]

    print("=" * 80)
    print("GEOMETRIC TRAJECTORY VISUALIZATION")
    print("=" * 80)
    print(f"\nPrompt: {trace['prompt'][:80]}...")
    print(f"Steps: {len(steps)}")

    # Extract metrics
    prs = [s.get("participation_ratio", 0) for s in steps]
    ses = [s.get("spectral_entropy", 0) for s in steps]
    states = [s.get("predicted_state", "unknown") for s in steps]

    if not any(prs):
        print("\nNo geometric telemetry in trace.")
        return

    # Statistics
    pr_mean = np.mean(prs)
    pr_std = np.std(prs)
    se_mean = np.mean(ses)
    se_std = np.std(ses)

    print(f"\nParticipation Ratio: mean={pr_mean:.2f}, std={pr_std:.2f}, range=[{min(prs):.2f}, {max(prs):.2f}]")
    print(f"Spectral Entropy:    mean={se_mean:.2f}, std={se_std:.2f}, range=[{min(ses):.2f}, {max(ses):.2f}]")

    # ASCII visualization of PR trajectory
    print("\n" + "-" * 80)
    print("PARTICIPATION RATIO TRAJECTORY")
    print("-" * 80)

    pr_min, pr_max = min(prs), max(prs)
    pr_range = pr_max - pr_min if pr_max > pr_min else 1

    for i, (pr, state, step) in enumerate(zip(prs, states, steps)):
        # Normalize to 0-40 for bar
        normalized = int(40 * (pr - pr_min) / pr_range) if pr_range > 0 else 20
        bar = "#" * normalized + " " * (40 - normalized)

        # Mark significant points
        marker = ""
        if pr == max(prs):
            marker = " <<< PEAK"
        elif pr == min(prs):
            marker = " <<< TROUGH"
        elif i > 0 and prs[i] - prs[i-1] > pr_std:
            marker = " ^ JUMP"
        elif i > 0 and prs[i-1] - prs[i] > pr_std:
            marker = " v DROP"

        # Truncate step text
        text_preview = step.get("text", "")[:50]

        print(f"Step {i+1:2d} | [{bar}] {pr:.2f}{marker}")
        print(f"        | {text_preview}...")

    # Identify potential insight moments
    print("\n" + "-" * 80)
    print("POTENTIAL INSIGHT MOMENTS")
    print("-" * 80)

    insights = []
    for i in range(1, len(steps)):
        pr_delta = prs[i] - prs[i-1]
        se_delta = ses[i] - ses[i-1]

        # Insight patterns:
        # 1. PR jump (expanding into new territory)
        # 2. PR drop followed by recovery (constraint then breakthrough)
        # 3. SE increase (more complex local structure)

        if pr_delta > pr_std:
            insights.append({
                "step": i + 1,
                "type": "PR_EXPANSION",
                "delta": pr_delta,
                "text": steps[i].get("text", "")[:100],
            })
        elif pr_delta < -pr_std and i + 1 < len(steps) and prs[i+1] > prs[i]:
            insights.append({
                "step": i + 1,
                "type": "CONSTRAINT_BREAKTHROUGH",
                "delta": pr_delta,
                "text": steps[i].get("text", "")[:100],
            })

    if insights:
        for insight in insights:
            print(f"\n  Step {insight['step']}: {insight['type']} (delta={insight['delta']:+.2f})")
            print(f"    \"{insight['text']}...\"")
    else:
        print("\n  No significant state transitions detected.")
        print("  (All steps within 1 std of mean - consistent exploration)")

    # G ratio analysis (within-trace uniformity)
    print("\n" + "-" * 80)
    print("WITHIN-TRACE UNIFORMITY (G RATIO)")
    print("-" * 80)

    g_ratio = min(prs) / np.mean(prs)
    cv = np.std(prs) / np.mean(prs)

    print(f"\n  G ratio: {g_ratio:.4f} (min/mean)")
    print(f"  CV:      {cv:.4f} (std/mean)")

    if g_ratio > 0.9:
        print("\n  INTERPRETATION: Very uniform PR - consistent cognitive state")
    elif g_ratio > 0.8:
        print("\n  INTERPRETATION: Moderate uniformity - stable exploration")
    else:
        print("\n  INTERPRETATION: High variation - dynamic reasoning with state transitions")

    # Step content analysis
    print("\n" + "-" * 80)
    print("STEP-BY-STEP CONTENT")
    print("-" * 80)

    for i, step in enumerate(steps):
        pr = prs[i]
        se = ses[i]
        text = step.get("text", "")

        # Highlight high/low PR steps
        if pr == max(prs):
            label = "[HIGH PR]"
        elif pr == min(prs):
            label = "[LOW PR]"
        else:
            label = ""

        print(f"\nStep {i+1} {label} (PR={pr:.2f}, SE={se:.2f}):")
        # Print first 200 chars
        print(f"  {text[:200]}...")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize reasoning trace")
    parser.add_argument("trace", help="Path to trace JSON file")

    args = parser.parse_args()

    visualize_trace(args.trace)


if __name__ == "__main__":
    main()
