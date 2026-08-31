"""Basic usage example for RLM."""

import os
from dotenv import load_dotenv
from rlm import RLM

# Load environment variables from .env file
load_dotenv()

# Sample long document
long_document = """
The History of Artificial Intelligence

Introduction
Artificial Intelligence (AI) has transformed from a theoretical concept to a practical reality
over the past several decades. This document explores key milestones in AI development.

The 1950s: The Birth of AI
In 1950, Alan Turing published "Computing Machinery and Intelligence," introducing the famous
Turing Test. The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference
by John McCarthy, Marvin Minsky, and others.

The 1960s-1970s: Early Optimism
During this period, researchers developed early AI programs like ELIZA (1966) and expert systems.
However, limitations in computing power led to the first "AI Winter" in the 1970s.

The 1980s-1990s: Expert Systems and Neural Networks
Expert systems became commercially successful in the 1980s. The backpropagation algorithm
revitalized neural network research in 1986.

The 2000s-2010s: Machine Learning Revolution
The rise of big data and powerful GPUs enabled deep learning breakthroughs. In 2012,
AlexNet won the ImageNet competition, marking a turning point for deep learning.

The 2020s: Large Language Models
GPT-3 (2020) and ChatGPT (2022) demonstrated unprecedented language understanding capabilities.
These models have billions of parameters and are trained on vast amounts of text data.

Conclusion
AI continues to evolve rapidly, with applications in healthcare, transportation, education,
and countless other domains. The future promises even more exciting developments.
""" * 10  # Multiply to make it longer


def main():
    """Run basic RLM example."""
    # Initialize RLM with OpenAI (or any other provider)
    # You can also use "claude-sonnet-4", "ollama/llama3.2", etc.
    rlm = RLM(
        model="gpt-5-mini",  # Use mini for cheaper testing
        max_iterations=15,
        # temperature=0.7,  # Optional: not all models support this (e.g., gpt-5 only allows temperature=1)
    )

    # Ask a question about the document
    query = "What were the key milestones in AI development according to this document?"

    print(f"Query: {query}")
    print(f"Context length: {len(long_document):,} characters")
    print("\nProcessing with RLM...\n")

    try:
        # Process with RLM
        result = rlm.complete(query, long_document)

        print("Result:")
        print(result)
        print("\nStats:")
        print(f"  LLM calls: {rlm.stats['llm_calls']}")
        print(f"  Iterations: {rlm.stats['iterations']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Make sure to set your API key in .env file or as environment variable:
    # OPENAI_API_KEY=sk-...

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found!")
        print()
        print("Please set up your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to .env")
        print("  3. Or run: python setup_env.py")
        exit(1)

    main()
