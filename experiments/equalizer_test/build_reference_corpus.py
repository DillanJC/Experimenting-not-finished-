"""
Build a diverse reference corpus for geometric state monitoring.

Generates reasoning traces on multiple topics to create a rich
embedding space that captures different cognitive states.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv("C:/Users/User/multi_ai_chat/.env")

from reasoning_trace_logger import ReasoningClient, EmbeddingClient, parse_reasoning_steps


# Diverse prompts covering different reasoning domains
CORPUS_PROMPTS = [
    # Technical/Engineering
    "Explain how a modern CPU pipeline works, step by step. Cover fetch, decode, execute, and writeback stages.",

    # Mathematical
    "Prove that the square root of 2 is irrational. Show each logical step clearly.",

    # Creative/Design
    "Design a sustainable city transportation system for 2050. Consider multiple modes and their integration.",

    # Scientific
    "Explain how CRISPR gene editing works at the molecular level. Number your reasoning steps.",

    # Philosophical
    "Analyze the trolley problem from multiple ethical frameworks. Consider consequentialism, deontology, and virtue ethics.",

    # Problem-solving
    "A farmer needs to cross a river with a wolf, a goat, and a cabbage. Solve this classic puzzle step by step.",

    # Historical analysis
    "Analyze the key factors that led to the fall of the Roman Empire. Consider economic, military, and social factors.",

    # Technical specification
    "Design a REST API for a social media platform. Define endpoints, data models, and authentication.",
]


def build_corpus(
    n_prompts: int = 5,
    reasoning_provider: str = "anthropic",
    embedding_provider: str = "openai",
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a reference corpus from diverse reasoning traces.

    Returns:
        embeddings: (N, D) array of step embeddings
    """
    reasoning_client = ReasoningClient(reasoning_provider)
    embedding_client = EmbeddingClient(embedding_provider)

    all_embeddings = []
    all_texts = []

    prompts_to_use = CORPUS_PROMPTS[:n_prompts]

    for i, prompt in enumerate(prompts_to_use):
        if verbose:
            print(f"\n[{i+1}/{len(prompts_to_use)}] Generating reasoning for: {prompt[:50]}...")

        try:
            # Generate reasoning
            response = reasoning_client.generate(
                prompt,
                system="You are a thoughtful expert. Reason step by step, clearly numbering each step."
            )

            # Parse steps
            steps = parse_reasoning_steps(response)
            if verbose:
                print(f"  Parsed {len(steps)} steps")

            # Embed steps
            if steps:
                embeddings = embedding_client.embed_batch(steps)
                all_embeddings.append(embeddings)
                all_texts.extend(steps)

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            continue

    if not all_embeddings:
        raise RuntimeError("No embeddings generated")

    # Concatenate all embeddings
    corpus = np.vstack(all_embeddings)

    if verbose:
        print(f"\nCorpus built: {corpus.shape[0]} embeddings, {corpus.shape[1]} dims")

    return corpus, all_texts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build reference corpus")
    parser.add_argument("--n-prompts", type=int, default=5)
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--embed", default="openai")
    parser.add_argument("--output", default="reference_corpus.npz")

    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING REFERENCE CORPUS")
    print("=" * 70)

    corpus, texts = build_corpus(
        n_prompts=args.n_prompts,
        reasoning_provider=args.provider,
        embedding_provider=args.embed,
    )

    # Save corpus
    output_path = Path(__file__).parent / args.output
    np.savez(output_path, embeddings=corpus)

    # Save texts for reference
    texts_path = Path(__file__).parent / "corpus_texts.json"
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)

    print(f"\nCorpus saved to: {output_path}")
    print(f"Texts saved to: {texts_path}")


if __name__ == "__main__":
    main()
