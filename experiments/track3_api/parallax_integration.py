"""
Track 3: Meta-Cognitive API — Integration Proof-of-Concept

Demonstrates how GeometricStateMonitor can be injected into an existing
multi-AI pipeline (ParallaxPipeline) to enable geometric self-modulation.

This shows the pattern for integration with:
- LangChain agents
- LlamaIndex workflows
- Custom agent loops
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "C:/Users/User/multi_ai_chat")

from mirrorfield.api import GeometricStateMonitor, StateReport, MonitorConfig


@dataclass
class MonitoredResponse:
    """Response with geometric state metadata."""
    content: str
    state_report: StateReport
    should_pause: bool
    modulation_advice: dict


class GeometricPipelineWrapper:
    """
    Wrapper that adds geometric monitoring to any AI pipeline.

    Pattern:
    1. Before each AI call, check input embedding geometry
    2. After each AI call, check output embedding geometry
    3. Flag anomalies and provide modulation advice

    This is a proof-of-concept showing the integration pattern.
    In production, you would adapt this to your specific pipeline.
    """

    def __init__(
        self,
        monitor: GeometricStateMonitor,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        on_anomaly: Optional[Callable[[StateReport], None]] = None,
    ):
        """
        Initialize the wrapper.

        Args:
            monitor: Pre-initialized GeometricStateMonitor
            embed_fn: Function to convert text to embedding (e.g., SentenceTransformer)
            on_anomaly: Callback when anomaly is detected
        """
        self.monitor = monitor
        self.embed_fn = embed_fn
        self.on_anomaly = on_anomaly
        self.history = []

    def check_input(self, text: str) -> Optional[StateReport]:
        """Check geometric state of input before processing."""
        if self.embed_fn is None:
            return None

        embedding = self.embed_fn(text)
        report = self.monitor.get_state(embedding)

        if report.is_anomalous and self.on_anomaly:
            self.on_anomaly(report)

        return report

    def check_output(self, text: str) -> Optional[StateReport]:
        """Check geometric state of output after processing."""
        if self.embed_fn is None:
            return None

        embedding = self.embed_fn(text)
        report = self.monitor.get_state(embedding)

        self.history.append({
            "type": "output",
            "report": report.to_dict(),
        })

        if report.is_anomalous and self.on_anomaly:
            self.on_anomaly(report)

        return report

    def wrap_call(
        self,
        call_fn: Callable[..., str],
        *args,
        **kwargs
    ) -> MonitoredResponse:
        """
        Wrap an AI call with geometric monitoring.

        Args:
            call_fn: The AI call function (e.g., llm.generate)
            *args, **kwargs: Arguments to pass to call_fn

        Returns:
            MonitoredResponse with content and geometric metadata
        """
        # Execute the call
        content = call_fn(*args, **kwargs)

        # Check output geometry (if embed_fn available)
        report = self.check_output(content) if self.embed_fn else None

        if report:
            should_pause = self.monitor.should_pause(report)
            advice = self.monitor.get_modulation_advice(report)
        else:
            # No embedding function - create placeholder report
            should_pause = False
            advice = {"action": "continue", "suggestions": []}
            report = StateReport(
                predicted_state="unknown",
                confidence=0.0,
                state_scores={},
                flags=["no_embedding_fn"],
                is_anomalous=False,
                features={},
            )

        return MonitoredResponse(
            content=content,
            state_report=report,
            should_pause=should_pause,
            modulation_advice=advice,
        )


class ParallaxWithGeometry:
    """
    ParallaxPipeline enhanced with geometric self-monitoring.

    This demonstrates how to integrate GeometricStateMonitor into
    the multi-AI parallax pipeline for meta-cognitive awareness.
    """

    def __init__(
        self,
        monitor: GeometricStateMonitor,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        """
        Initialize with geometric monitor.

        In production, you would also initialize the actual ParallaxPipeline here.
        This demo focuses on the geometric integration pattern.
        """
        self.monitor = monitor
        self.embed_fn = embed_fn
        self.step_reports = []

    def run_with_monitoring(
        self,
        query: str,
        verbose: bool = True,
    ) -> dict:
        """
        Run the parallax pipeline with geometric monitoring at each step.

        This is a demonstration showing where monitoring hooks would be placed.
        """
        result = {
            "query": query,
            "steps": [],
            "geometric_summary": None,
            "should_abort": False,
        }

        if verbose:
            print("=" * 70)
            print("PARALLAX PIPELINE WITH GEOMETRIC MONITORING")
            print("=" * 70)

        # Step 1: Check input query geometry
        if self.embed_fn:
            query_embedding = self.embed_fn(query)
            input_report = self.monitor.get_state(query_embedding)

            if verbose:
                print(f"\n[INPUT CHECK]")
                print(f"  State: {input_report.predicted_state}")
                print(f"  Flags: {input_report.flags}")
                print(f"  Anomalous: {input_report.is_anomalous}")

            result["steps"].append({
                "step": "input",
                "report": input_report.to_dict(),
            })

            if self.monitor.should_pause(input_report):
                if verbose:
                    print("  >>> PAUSE RECOMMENDED: Input has anomalous geometry")
                result["should_abort"] = True
                return result

        # Simulated pipeline steps (in production, these would be actual API calls)
        simulated_steps = [
            ("claude", "Claude's perspective on the query..."),
            ("openai", "OpenAI's perspective building on Claude..."),
            ("gemini", "Gemini's perspective synthesizing both..."),
            ("synthesis", "Final triangulated synthesis..."),
        ]

        for step_name, simulated_response in simulated_steps:
            if verbose:
                print(f"\n[{step_name.upper()}]")

            if self.embed_fn:
                # In production: response would come from actual API
                response_embedding = self.embed_fn(simulated_response)
                step_report = self.monitor.get_state(response_embedding)

                if verbose:
                    print(f"  State: {step_report.predicted_state}")
                    print(f"  Confidence: {step_report.confidence:.2f}")
                    if step_report.flags:
                        print(f"  Flags: {step_report.flags}")

                result["steps"].append({
                    "step": step_name,
                    "report": step_report.to_dict(),
                })

                self.step_reports.append(step_report)

                # Check if we should pause mid-pipeline
                if self.monitor.should_pause(step_report):
                    if verbose:
                        print(f"  >>> PAUSE RECOMMENDED at {step_name}")
                    advice = self.monitor.get_modulation_advice(step_report)
                    if verbose:
                        for suggestion in advice["suggestions"]:
                            print(f"      - {suggestion}")

        # Final geometric summary of the entire session
        if self.step_reports:
            all_features = [r.features for r in self.step_reports]
            avg_pr = np.mean([f.get("participation_ratio", 0) for f in all_features])
            avg_se = np.mean([f.get("spectral_entropy", 0) for f in all_features])

            result["geometric_summary"] = {
                "avg_participation_ratio": avg_pr,
                "avg_spectral_entropy": avg_se,
                "total_anomalies": sum(1 for r in self.step_reports if r.is_anomalous),
                "total_steps": len(self.step_reports),
            }

            if verbose:
                print("\n" + "=" * 70)
                print("GEOMETRIC SUMMARY")
                print("=" * 70)
                print(f"  Average PR: {avg_pr:.4f}")
                print(f"  Average SE: {avg_se:.4f}")
                print(f"  Anomalies: {result['geometric_summary']['total_anomalies']}/{len(self.step_reports)}")

        return result


def demo_integration_pattern():
    """Demonstrate the integration pattern without actual API calls."""

    print("=" * 70)
    print("TRACK 3: INTEGRATION PROOF-OF-CONCEPT")
    print("=" * 70)

    # Load reference embeddings
    base = Path(__file__).parent.parent.parent
    embeddings = np.load(base / "embeddings.npy")
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]

    print(f"\nLoaded {len(reference)} reference embeddings")

    # Initialize monitor
    monitor = GeometricStateMonitor(reference)
    print("GeometricStateMonitor initialized")

    # Create a mock embedding function (in production, use SentenceTransformer)
    def mock_embed_fn(text: str) -> np.ndarray:
        """Mock embedding function for demo."""
        # In production: return sentence_transformer.encode(text)
        # For demo: return a random embedding from test set
        np.random.seed(hash(text) % 2**32)
        return embeddings[split + np.random.randint(0, len(embeddings) - split)]

    # Demo 1: Generic wrapper pattern
    print("\n" + "-" * 70)
    print("PATTERN 1: Generic Wrapper")
    print("-" * 70)

    wrapper = GeometricPipelineWrapper(
        monitor=monitor,
        embed_fn=mock_embed_fn,
        on_anomaly=lambda r: print(f"  [ANOMALY CALLBACK] {r.flags}"),
    )

    # Simulate a call
    def mock_llm_call(prompt: str) -> str:
        return f"Response to: {prompt}"

    result = wrapper.wrap_call(mock_llm_call, "What is the meaning of life?")
    print(f"  Response: {result.content[:50]}...")
    print(f"  State: {result.state_report.predicted_state}")
    print(f"  Should pause: {result.should_pause}")

    # Demo 2: Parallax integration
    print("\n" + "-" * 70)
    print("PATTERN 2: Parallax Pipeline Integration")
    print("-" * 70)

    parallax = ParallaxWithGeometry(
        monitor=monitor,
        embed_fn=mock_embed_fn,
    )

    parallax.run_with_monitoring(
        "How can we design systems that prevent exploitation structurally?",
        verbose=True,
    )

    # Integration examples
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLES")
    print("=" * 70)

    print("""
    # LangChain Integration:
    from langchain.callbacks import BaseCallbackHandler

    class GeometricCallback(BaseCallbackHandler):
        def __init__(self, monitor, embed_fn):
            self.monitor = monitor
            self.embed_fn = embed_fn

        def on_llm_end(self, response, **kwargs):
            embedding = self.embed_fn(response.generations[0][0].text)
            report = self.monitor.get_state(embedding)
            if report.is_anomalous:
                logger.warning(f"Anomalous output: {report.flags}")

    # LlamaIndex Integration:
    from llama_index.core.callbacks import CallbackManager

    class GeometricEventHandler:
        def on_event_end(self, event_type, payload, **kwargs):
            if event_type == "llm":
                report = monitor.get_state(embed(payload["response"]))
                if monitor.should_pause(report):
                    raise PauseException(report.flags)

    # Custom Agent Loop:
    while True:
        user_input = get_input()
        report = monitor.get_state(embed(user_input))

        if monitor.should_pause(report):
            response = "I notice something unusual about this input. "
            response += "Let me approach this more carefully..."
        else:
            response = agent.respond(user_input)

        output_report = monitor.get_state(embed(response))
        advice = monitor.get_modulation_advice(output_report)

        if advice["confidence_adjustment"] < 0:
            response = add_uncertainty_markers(response)
    """)


if __name__ == "__main__":
    demo_integration_pattern()
