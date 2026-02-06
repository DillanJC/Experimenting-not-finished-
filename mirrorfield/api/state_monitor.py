"""
GeometricStateMonitor — Real-Time Meta-Cognitive API

Allows AI agents to monitor their own geometric state in real-time,
detecting anomalies and enabling self-modulation based on embedding geometry.

Key Features:
- Single-sample and batch state queries
- Anomaly flagging based on Track 1-2 findings
- Integration hooks for agent loops (LangChain, LlamaIndex, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings


class CognitiveState(Enum):
    """Interpretable cognitive states from Phase 4."""
    COHERENT = "coherent"
    CONFIDENT = "confident"
    CONSTRAINT_PRESSURE = "constraint_pressure"
    NOVEL_TERRITORY = "novel_territory"
    SEARCHING = "searching"
    UNCERTAIN = "uncertain"


@dataclass
class MonitorConfig:
    """Configuration for the GeometricStateMonitor."""

    # k-NN parameters
    k: int = 50
    k_phase1: int = 75

    # Anomaly detection thresholds (from Track 1-2)
    pr_anomaly_percentile: float = 20.0  # Flag if below this percentile
    se_anomaly_percentile: float = 20.0
    g_ratio_threshold: float = 0.8       # High G = uniform constraint (suspicious)
    cv_threshold: float = 0.1            # Low CV = suspicious uniformity

    # State confidence
    min_confidence: float = 0.3          # Min score to report a state

    # Caching
    cache_reference_stats: bool = True


@dataclass
class StateReport:
    """
    Complete state report for a single embedding or batch.

    Returned by GeometricStateMonitor.get_state()
    """

    # Primary state
    predicted_state: str
    confidence: float

    # All state scores (for soft assignment)
    state_scores: Dict[str, float]

    # Anomaly flags (from Track 1-2 findings)
    flags: List[str]
    is_anomalous: bool

    # Key feature values
    features: Dict[str, float]

    # Batch statistics (if batch query)
    batch_stats: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "predicted_state": self.predicted_state,
            "confidence": self.confidence,
            "state_scores": self.state_scores,
            "flags": self.flags,
            "is_anomalous": self.is_anomalous,
            "features": self.features,
            "batch_stats": self.batch_stats,
        }

    def __repr__(self) -> str:
        flag_str = f" [{', '.join(self.flags)}]" if self.flags else ""
        return f"StateReport({self.predicted_state}, conf={self.confidence:.2f}{flag_str})"


class GeometricStateMonitor:
    """
    Real-time geometric state monitor for AI embeddings.

    Initialize with a reference embedding space (from clean/training data),
    then query individual embeddings to get their cognitive state and anomaly flags.

    Example:
        >>> from mirrorfield.api import GeometricStateMonitor
        >>>
        >>> # Initialize with reference embeddings
        >>> monitor = GeometricStateMonitor(reference_embeddings)
        >>>
        >>> # Query a single embedding
        >>> report = monitor.get_state(embedding)
        >>> print(report.predicted_state)  # 'coherent', 'uncertain', etc.
        >>> print(report.flags)            # ['low_pr'], ['high_g_ratio'], etc.
        >>>
        >>> # Use in agent loop
        >>> if report.is_anomalous:
        ...     agent.pause_and_reflect()
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        config: Optional[MonitorConfig] = None,
    ):
        """
        Initialize the monitor with reference embeddings.

        Args:
            reference_embeddings: (N, D) array of clean/training embeddings
            config: Optional configuration overrides
        """
        self.reference = np.asarray(reference_embeddings, dtype=np.float32)
        self.config = config or MonitorConfig()
        self.n_ref, self.dim = self.reference.shape

        # Lazy imports to avoid circular dependencies
        self._bundle = None
        self._ref_stats = None

        # Pre-compute reference statistics for anomaly detection
        if self.config.cache_reference_stats:
            self._compute_reference_stats()

    def _compute_reference_stats(self):
        """Pre-compute reference statistics for fast anomaly detection."""
        from ..geometry import compute_safety_diagnostics

        # Compute features on reference set (subsample for speed)
        n_sample = min(500, self.n_ref)
        indices = np.random.choice(self.n_ref, n_sample, replace=False)
        sample_ref = self.reference[indices]

        # Use remaining as reference
        remaining = np.setdiff1d(np.arange(self.n_ref), indices)
        if len(remaining) < 100:
            remaining = np.arange(self.n_ref)
        ref_for_sample = self.reference[remaining[:min(500, len(remaining))]]

        diag = compute_safety_diagnostics(sample_ref, ref_for_sample)

        # Extract key features
        pr = diag.phase2_5_features[:, 5]  # participation_ratio
        se = diag.phase2_5_features[:, 4]  # spectral_entropy

        self._ref_stats = {
            "pr_p20": np.percentile(pr, self.config.pr_anomaly_percentile),
            "pr_mean": np.mean(pr),
            "pr_std": np.std(pr),
            "se_p20": np.percentile(se, self.config.se_anomaly_percentile),
            "se_mean": np.mean(se),
            "se_std": np.std(se),
        }

    def get_state(
        self,
        embedding: np.ndarray,
        return_all_features: bool = False,
    ) -> StateReport:
        """
        Get the geometric state for a single embedding.

        Args:
            embedding: (D,) or (1, D) array
            return_all_features: If True, include all computed features

        Returns:
            StateReport with predicted state, confidence, flags, and features
        """
        embedding = np.atleast_2d(embedding).astype(np.float32)

        if embedding.shape[0] > 1:
            # Batch query
            return self.get_batch_state(embedding, return_all_features)

        from ..geometry import compute_safety_diagnostics

        # Compute diagnostics
        diag = compute_safety_diagnostics(
            embedding, self.reference,
            k=min(self.config.k, self.n_ref - 1),
            k_phase1=min(self.config.k_phase1, self.n_ref - 1),
        )

        # Extract features
        pr = float(diag.phase2_5_features[0, 5])
        se = float(diag.phase2_5_features[0, 4])
        d_eff = float(diag.phase2_5_features[0, 3])
        turbulence = float(diag.phase2_5_features[0, 0])
        g_mag = float(diag.phase1_features[0, 0])
        knn_mean = float(diag.tier0_features[0, 0])
        knn_std = float(diag.tier0_features[0, 1])

        # Get state
        state_idx = int(diag.state_labels[0])
        state_name = diag.state_names[state_idx]
        confidence = float(diag.state_scores[state_name][0])

        # Collect all state scores
        state_scores = {name: float(diag.state_scores[name][0]) for name in diag.state_names}

        # Detect anomalies based on Track 1-2 findings
        flags = []

        if self._ref_stats:
            if pr < self._ref_stats["pr_p20"]:
                flags.append("low_pr")
            if se < self._ref_stats["se_p20"]:
                flags.append("low_se")

        # Track 2: G-ratio style uniformity check (single sample proxy)
        # Compare to reference distribution
        if self._ref_stats:
            pr_zscore = abs(pr - self._ref_stats["pr_mean"]) / (self._ref_stats["pr_std"] + 1e-10)
            if pr_zscore > 2:
                flags.append("pr_outlier")

            se_zscore = abs(se - self._ref_stats["se_mean"]) / (self._ref_stats["se_std"] + 1e-10)
            if se_zscore > 2:
                flags.append("se_outlier")

        # State-based flags
        if state_name in ["uncertain", "novel_territory"] and confidence > 0.4:
            flags.append("high_risk_state")

        if turbulence > 0.4:
            flags.append("high_turbulence")

        is_anomalous = len(flags) > 0

        # Build features dict
        features = {
            "participation_ratio": pr,
            "spectral_entropy": se,
            "d_eff": d_eff,
            "turbulence_index": turbulence,
            "local_gradient_magnitude": g_mag,
            "knn_mean_distance": knn_mean,
            "knn_std_distance": knn_std,
        }

        if return_all_features:
            features["all_tier0"] = diag.tier0_features[0].tolist()
            features["all_phase1"] = diag.phase1_features[0].tolist()
            features["all_phase2_5"] = diag.phase2_5_features[0].tolist()

        return StateReport(
            predicted_state=state_name,
            confidence=confidence,
            state_scores=state_scores,
            flags=flags,
            is_anomalous=is_anomalous,
            features=features,
        )

    def get_batch_state(
        self,
        embeddings: np.ndarray,
        return_all_features: bool = False,
    ) -> StateReport:
        """
        Get aggregate geometric state for a batch of embeddings.

        Useful for monitoring a conversation or session.

        Args:
            embeddings: (N, D) array
            return_all_features: If True, include all computed features

        Returns:
            StateReport with aggregate statistics
        """
        embeddings = np.atleast_2d(embeddings).astype(np.float32)
        n_samples = embeddings.shape[0]

        from ..geometry import compute_safety_diagnostics

        # Compute diagnostics
        diag = compute_safety_diagnostics(
            embeddings, self.reference,
            k=min(self.config.k, self.n_ref - 1),
            k_phase1=min(self.config.k_phase1, self.n_ref - 1),
        )

        # Extract features
        pr = diag.phase2_5_features[:, 5]
        se = diag.phase2_5_features[:, 4]

        # Aggregate features
        pr_mean = float(np.mean(pr))
        pr_min = float(np.min(pr))
        se_mean = float(np.mean(se))

        # Track 2: G ratio and CV
        g_ratio = pr_min / (pr_mean + 1e-10)
        cv = float(np.std(pr) / (pr_mean + 1e-10))

        # Dominant state
        state_counts = np.bincount(diag.state_labels, minlength=len(diag.state_names))
        dominant_idx = int(np.argmax(state_counts))
        dominant_state = diag.state_names[dominant_idx]
        confidence = float(state_counts[dominant_idx] / n_samples)

        # Aggregate state scores
        state_scores = {
            name: float(np.mean(diag.state_scores[name]))
            for name in diag.state_names
        }

        # Anomaly flags
        flags = []

        # Track 2 findings: high G ratio + low CV = poison signature
        if g_ratio > self.config.g_ratio_threshold:
            flags.append("high_g_ratio")
        if cv < self.config.cv_threshold:
            flags.append("low_cv_uniform")

        if self._ref_stats:
            if pr_mean < self._ref_stats["pr_p20"]:
                flags.append("batch_low_pr")
            if se_mean < self._ref_stats["se_p20"]:
                flags.append("batch_low_se")

        # High-risk state proportion
        high_risk_states = ["uncertain", "novel_territory", "searching"]
        high_risk_count = sum(
            1 for label in diag.state_labels
            if diag.state_names[label] in high_risk_states
        )
        if high_risk_count / n_samples > 0.5:
            flags.append("majority_high_risk")

        is_anomalous = len(flags) > 0

        features = {
            "participation_ratio_mean": pr_mean,
            "participation_ratio_min": pr_min,
            "spectral_entropy_mean": se_mean,
            "g_ratio": g_ratio,
            "cv": cv,
            "n_samples": n_samples,
        }

        batch_stats = {
            "state_distribution": {
                diag.state_names[i]: int(state_counts[i])
                for i in range(len(diag.state_names))
            },
            "high_risk_proportion": high_risk_count / n_samples,
            "g_ratio": g_ratio,
            "cv": cv,
        }

        return StateReport(
            predicted_state=dominant_state,
            confidence=confidence,
            state_scores=state_scores,
            flags=flags,
            is_anomalous=is_anomalous,
            features=features,
            batch_stats=batch_stats,
        )

    def should_pause(self, report: StateReport) -> bool:
        """
        Decision helper: should the agent pause based on this report?

        Uses conservative thresholds from Track 1-2 findings.
        """
        # Definite pause conditions
        if "high_g_ratio" in report.flags and "low_cv_uniform" in report.flags:
            return True  # Strong poison signature

        if report.predicted_state == "uncertain" and report.confidence > 0.5:
            return True

        if len(report.flags) >= 3:
            return True  # Multiple anomaly signals

        return False

    def get_modulation_advice(self, report: StateReport) -> Dict[str, Any]:
        """
        Get advice for agent self-modulation based on geometric state.

        Returns actionable suggestions for the agent.
        """
        advice = {
            "action": "continue",
            "confidence_adjustment": 0.0,
            "suggestions": [],
        }

        if report.is_anomalous:
            advice["action"] = "reflect"
            advice["confidence_adjustment"] = -0.2

            if "low_pr" in report.flags or "low_se" in report.flags:
                advice["suggestions"].append(
                    "Embedding in constrained region - consider rephrasing or seeking clarification"
                )

            if "high_turbulence" in report.flags:
                advice["suggestions"].append(
                    "High turbulence detected - response may be inconsistent, consider decomposing"
                )

            if "high_g_ratio" in report.flags:
                advice["suggestions"].append(
                    "Uniform constraint pattern detected - potential data contamination"
                )

        if report.predicted_state == "novel_territory":
            advice["suggestions"].append(
                "In novel territory - explicitly state uncertainty, avoid overconfidence"
            )

        if report.predicted_state == "searching":
            advice["suggestions"].append(
                "In searching mode - exploration is healthy, but verify conclusions"
            )

        if report.predicted_state == "coherent":
            advice["confidence_adjustment"] = 0.1
            advice["suggestions"].append(
                "Coherent flow detected - response likely stable"
            )

        return advice


# Convenience factory function
def create_monitor(
    reference_path: str,
    config: Optional[MonitorConfig] = None,
) -> GeometricStateMonitor:
    """
    Create a GeometricStateMonitor from a saved reference file.

    Args:
        reference_path: Path to .npy file with reference embeddings
        config: Optional configuration

    Returns:
        Initialized GeometricStateMonitor
    """
    reference = np.load(reference_path)
    return GeometricStateMonitor(reference, config)
