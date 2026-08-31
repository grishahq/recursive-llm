"""Example processing a very long document (50k+ tokens)."""

from rlm import RLM


# Generate a realistic long document (simulating a research paper or book)
def generate_long_document():
    """Generate a long document for testing."""
    chapters = []

    for i in range(1, 21):  # 20 chapters
        chapter = f"""
Chapter {i}: Topic {i}

This chapter discusses important concept {i} in great detail. The key findings include:

1. First major point about topic {i}
   - Supporting detail A
   - Supporting detail B
   - Supporting detail C

2. Second major point about topic {i}
   - Evidence from study X
   - Evidence from study Y
   - Conclusion based on evidence

3. Third major point about topic {i}
   - Historical context
   - Current applications
   - Future implications

Key Statistics:
- Metric A: {i * 10}%
- Metric B: {i * 100} units
- Metric C: ${i * 1000}

Important dates:
- Event 1: January {i}, 2024
- Event 2: February {i}, 2024
- Event 3: March {i}, 2024

Conclusion:
Topic {i} represents a critical area of research with significant implications
for the field. Further investigation is warranted.

References:
[1] Author {i}. "Study on Topic {i}". Journal of Research. 2024.
[2] Researcher {i}. "Analysis of Topic {i}". Scientific Papers. 2024.

""" + "Additional context paragraph. " * 100  # Make each chapter longer
        chapters.append(chapter)

    return "\n\n".join(chapters)


def main():
    """Process long document with RLM."""
    # Generate document
    print("Generating long document...")
    document = generate_long_document()
    print(f"Document generated: {len(document):,} characters")
    print(f"Estimated tokens: ~{len(document) // 4:,}")
    print()

    # Initialize RLM
    rlm = RLM(model="gpt-5-mini", max_iterations=20, temperature=0.5)

    # Complex queries that require understanding the whole document
    queries = [
        "What is the range of Metric B values across all chapters?",
        "Which chapter has the highest Metric A percentage?",
        "Summarize the key findings from chapters 5-10",
        "How many total references are cited in the document?",
    ]

    print("Processing queries...\n")

    for query in queries:
        print(f"Query: {query}")

        try:
            result = rlm.complete(query, document)

            print(f"Answer: {result}")
            print(
                f"Performance: {rlm.stats['llm_calls']} LLM calls, "
                f"{rlm.stats['iterations']} iterations"
            )
            print("-" * 80)
            print()

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # This example demonstrates RLM's ability to handle very long contexts
    # that would cause "context rot" in traditional approaches
    main()
