"""
Reasoning Trace Logger — Geometric Telemetry for AI Reasoning

Tracks an AI's reasoning process through embedding space, logging
geometric state transitions as it works through complex problems.

Usage:
    python reasoning_trace_logger.py --provider anthropic --embed openai
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
# Load from multi_ai_chat .env
load_dotenv("C:/Users/User/multi_ai_chat/.env")


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    step_num: int
    timestamp: float
    text: str
    embedding: Optional[np.ndarray] = None

    # Geometric telemetry
    predicted_state: Optional[str] = None
    state_confidence: Optional[float] = None
    participation_ratio: Optional[float] = None
    spectral_entropy: Optional[float] = None
    g_ratio: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    is_anomalous: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


@dataclass
class ReasoningTrace:
    """Complete trace of AI reasoning with geometric telemetry."""
    prompt: str
    provider: str
    model: str
    full_response: str
    steps: List[ReasoningStep] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0

    # Aggregate metrics
    state_trajectory: List[str] = field(default_factory=list)
    novelty_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "provider": self.provider,
            "model": self.model,
            "full_response": self.full_response,
            "steps": [s.to_dict() for s in self.steps],
            "duration_s": self.end_time - self.start_time,
            "state_trajectory": self.state_trajectory,
            "novelty_score": self.novelty_score,
        }


class EmbeddingClient:
    """Unified interface for embedding APIs."""

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "text-embedding-3-small"
            self.dim = 1536
        elif self.provider == "local":
            # Use sentence-transformers locally
            try:
                from sentence_transformers import SentenceTransformer
                self.client = SentenceTransformer("all-MiniLM-L6-v2")
                self.dim = 384
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        elif self.provider == "local":
            return self.client.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return np.array([d.embedding for d in response.data], dtype=np.float32)
        elif self.provider == "local":
            return self.client.encode(texts, convert_to_numpy=True).astype(np.float32)


class ReasoningClient:
    """Unified interface for reasoning APIs."""

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-sonnet-4-20250514"
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o"
        else:
            raise ValueError(f"Unknown reasoning provider: {self.provider}")

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate reasoning response."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "You are an expert signal processing engineer. Think step by step, numbering each reasoning step clearly.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system or "You are an expert signal processing engineer. Think step by step, numbering each reasoning step clearly."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content


def parse_reasoning_steps(text: str) -> List[str]:
    """
    Parse a reasoning response into discrete steps.

    Handles multiple formats:
    - Markdown headers (## Step 1:, ### Step 2, etc.)
    - Numbered steps (1., 2., Step 1:, etc.)
    - Paragraph breaks
    """
    steps = []

    # Try markdown header parsing first (## Step X, ### Step X)
    markdown_pattern = r'(?:^|\n)#{1,3}\s*Step\s*(\d+)[:\s]'
    markdown_matches = list(re.finditer(markdown_pattern, text, re.IGNORECASE))

    if len(markdown_matches) >= 3:
        # Use markdown parsing
        for i, match in enumerate(markdown_matches):
            start = match.end()
            end = markdown_matches[i + 1].start() if i + 1 < len(markdown_matches) else len(text)
            step_text = text[start:end].strip()
            if len(step_text) > 30:
                steps.append(step_text)
        return steps

    # Try numbered parsing
    # Match patterns like "1.", "1)", "Step 1:", "**1.**"
    numbered_pattern = r'(?:^|\n)(?:\*\*)?(?:Step\s*)?(\d+)[.):]\**\s*'
    numbered_matches = list(re.finditer(numbered_pattern, text, re.IGNORECASE))

    if len(numbered_matches) >= 3:
        # Use numbered parsing
        for i, match in enumerate(numbered_matches):
            start = match.end()
            end = numbered_matches[i + 1].start() if i + 1 < len(numbered_matches) else len(text)
            step_text = text[start:end].strip()
            if len(step_text) > 50:  # Minimum step length
                steps.append(step_text)
    else:
        # Fall back to paragraph parsing
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Minimum step length
                steps.append(para)

    # If still no steps, split by sentences
    if len(steps) < 2:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_step = []
        for sent in sentences:
            current_step.append(sent)
            if len(" ".join(current_step)) > 200:
                steps.append(" ".join(current_step))
                current_step = []
        if current_step:
            steps.append(" ".join(current_step))

    return steps


class ReasoningTraceLogger:
    """
    Main logger that orchestrates reasoning, embedding, and geometric monitoring.
    """

    def __init__(
        self,
        reasoning_provider: str = "anthropic",
        embedding_provider: str = "openai",
        reference_embeddings: Optional[np.ndarray] = None,
    ):
        self.reasoning_client = ReasoningClient(reasoning_provider)
        self.embedding_client = EmbeddingClient(embedding_provider)

        # Initialize geometric monitor if reference provided
        self.monitor = None
        if reference_embeddings is not None:
            from mirrorfield.api import GeometricStateMonitor, MonitorConfig

            # Check dimension match
            ref_dim = reference_embeddings.shape[1]
            embed_dim = self.embedding_client.dim

            if ref_dim != embed_dim:
                print(f"Note: Reference dim {ref_dim} != embedding dim {embed_dim}")
                print("Building reference from scratch using embedding API...")
                self.monitor = None
            else:
                # Adjust k for reference size
                k_val = min(20, len(reference_embeddings) - 2)
                config = MonitorConfig(
                    k=k_val,
                    k_phase1=k_val,
                    pr_anomaly_percentile=25.0,
                    se_anomaly_percentile=25.0,
                    cache_reference_stats=False,  # Disable for small reference sets
                )
                self.monitor = GeometricStateMonitor(reference_embeddings, config)
                print(f"GeometricStateMonitor initialized with {len(reference_embeddings)} reference embeddings (k={k_val})")

        self.reference_embeddings = []  # Accumulate for building reference

    def trace_reasoning(
        self,
        prompt: str,
        system: str = "",
        verbose: bool = True,
    ) -> ReasoningTrace:
        """
        Generate reasoning and trace through embedding space.
        """
        trace = ReasoningTrace(
            prompt=prompt,
            provider=self.reasoning_client.provider,
            model=self.reasoning_client.model,
            full_response="",
            start_time=time.time(),
        )

        if verbose:
            print("=" * 70)
            print("REASONING TRACE LOGGER")
            print("=" * 70)
            print(f"\nPrompt: {prompt[:100]}...")
            print(f"\nGenerating reasoning with {self.reasoning_client.provider}...")

        # Generate reasoning
        response = self.reasoning_client.generate(prompt, system)
        trace.full_response = response
        trace.end_time = time.time()

        if verbose:
            print(f"Response generated in {trace.end_time - trace.start_time:.2f}s")
            print(f"Response length: {len(response)} chars")

        # Parse into steps
        step_texts = parse_reasoning_steps(response)

        if verbose:
            print(f"\nParsed {len(step_texts)} reasoning steps")

        # Embed all steps
        if verbose:
            print(f"\nEmbedding steps with {self.embedding_client.provider}...")

        embeddings = self.embedding_client.embed_batch(step_texts)

        # Build reference if needed (first run)
        if self.monitor is None and len(embeddings) >= 5:
            from mirrorfield.api import GeometricStateMonitor, MonitorConfig
            # Adjust k for small reference sets
            k_val = min(10, len(embeddings) - 2)
            config = MonitorConfig(
                k=k_val,
                k_phase1=k_val,
                pr_anomaly_percentile=25.0,
                se_anomaly_percentile=25.0,
                cache_reference_stats=False,  # Skip for small sets
            )
            # Use first batch as reference
            self.monitor = GeometricStateMonitor(embeddings, config)
            if verbose:
                print(f"Built reference from {len(embeddings)} step embeddings (k={k_val})")

        # Analyze each step
        if verbose:
            print("\n" + "-" * 70)
            print("STEP-BY-STEP GEOMETRIC TELEMETRY")
            print("-" * 70)
            print(f"\n{'Step':<5} | {'State':<20} | {'PR':>8} | {'SE':>8} | {'Flags':<30}")
            print("-" * 80)

        for i, (text, embedding) in enumerate(zip(step_texts, embeddings)):
            step = ReasoningStep(
                step_num=i + 1,
                timestamp=time.time(),
                text=text[:200] + "..." if len(text) > 200 else text,
                embedding=embedding,
            )

            # Get geometric state if monitor available
            if self.monitor is not None:
                try:
                    report = self.monitor.get_state(embedding)
                    step.predicted_state = report.predicted_state
                    step.state_confidence = report.confidence
                    step.participation_ratio = report.features.get("participation_ratio", 0)
                    step.spectral_entropy = report.features.get("spectral_entropy", 0)
                    step.flags = report.flags
                    step.is_anomalous = report.is_anomalous

                    trace.state_trajectory.append(report.predicted_state)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not get state for step {i+1}: {e}")

            trace.steps.append(step)

            if verbose and step.predicted_state:
                flags_str = ", ".join(step.flags) if step.flags else "-"
                print(f"{step.step_num:<5} | {step.predicted_state:<20} | {step.participation_ratio:>8.2f} | {step.spectral_entropy:>8.2f} | {flags_str:<30}")

        # Compute aggregate metrics
        if trace.steps and self.monitor is not None:
            trace.novelty_score = self._compute_novelty_score(trace)

        if verbose:
            print("\n" + "=" * 70)
            print("TRACE SUMMARY")
            print("=" * 70)
            print(f"\nState trajectory: {' -> '.join(trace.state_trajectory)}")
            print(f"Novelty score: {trace.novelty_score:.3f}" if trace.novelty_score else "Novelty score: N/A")

            # State distribution
            if trace.state_trajectory:
                from collections import Counter
                state_counts = Counter(trace.state_trajectory)
                print(f"\nState distribution:")
                for state, count in state_counts.most_common():
                    print(f"  {state}: {count} ({100*count/len(trace.state_trajectory):.1f}%)")

        return trace

    def _compute_novelty_score(self, trace: ReasoningTrace) -> float:
        """
        Compute a novelty score based on geometric trajectory.

        Higher score indicates:
        - Transitions through constraint_pressure (creative tension)
        - Movement from searching to coherent (discovery)
        - Presence of novel_territory states
        """
        if not trace.state_trajectory:
            return 0.0

        score = 0.0

        # State weights (novel/creative states score higher)
        state_weights = {
            "constraint_pressure": 0.3,  # Creative tension
            "novel_territory": 0.25,     # Unexplored space
            "searching": 0.15,           # Active exploration
            "coherent": 0.1,             # Stable insight
            "confident": 0.05,           # Established knowledge
            "uncertain": 0.15,           # Productive uncertainty
        }

        # Base score from state presence
        for state in trace.state_trajectory:
            score += state_weights.get(state, 0)

        # Bonus for transitions (searching -> constraint_pressure -> coherent)
        for i in range(len(trace.state_trajectory) - 1):
            curr, next_state = trace.state_trajectory[i], trace.state_trajectory[i + 1]

            if curr == "searching" and next_state == "constraint_pressure":
                score += 0.2  # Entering creative tension
            if curr == "constraint_pressure" and next_state == "coherent":
                score += 0.3  # Resolution of tension
            if curr == "novel_territory" and next_state == "coherent":
                score += 0.25  # Discovery in new territory

        # Normalize by number of steps
        score /= len(trace.state_trajectory)

        return min(1.0, score)


# The equalizer prompt
EQUALIZER_PROMPT = """Design a minimal-latency, high-resolution audio equalizer filter that maintains phase linearity.

Consider novel approaches beyond traditional IIR/FIR trade-offs. The key constraints are:
1. Latency must be near-zero (< 1ms for real-time audio)
2. Frequency resolution must be high (at least 1/12 octave)
3. Phase response must be linear (no group delay distortion)
4. Computational efficiency must allow real-time processing

Think step by step, exploring unconventional solutions. Number each step of your reasoning."""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning Trace Logger")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--embed", default="openai", choices=["openai", "local"])
    parser.add_argument("--prompt", default=EQUALIZER_PROMPT)
    parser.add_argument("--output", default="reasoning_trace.json")
    parser.add_argument("--reference", default=None, help="Path to reference corpus .npz")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)

    args = parser.parse_args()

    # Check API keys
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    if args.embed == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set for embeddings")
        return

    # Load reference corpus if provided
    reference = None
    if args.reference:
        data = np.load(args.reference)
        reference = data["embeddings"]
        print(f"Loaded reference corpus: {reference.shape}")

    # Create logger
    logger = ReasoningTraceLogger(
        reasoning_provider=args.provider,
        embedding_provider=args.embed,
        reference_embeddings=reference,
    )

    # Run trace
    trace = logger.trace_reasoning(args.prompt, verbose=args.verbose)

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nTrace saved to: {output_path}")


if __name__ == "__main__":
    main()
