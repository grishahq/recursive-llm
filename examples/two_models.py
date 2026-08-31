"""Example using two different models for cost optimization."""

from rlm import RLM

# Very long document
long_document = (
    """
Annual Financial Report 2024
"""
    + "\n\n"
    + """
Executive Summary:
Our company achieved record revenue of $500M in 2024, representing 25% year-over-year growth.
Net income reached $75M, with an operating margin of 18%.
"""
    + "\n\n"
    + ("""
Quarterly Performance:
Q1 2024: Revenue $110M, Net Income $15M
Q2 2024: Revenue $120M, Net Income $18M
Q3 2024: Revenue $130M, Net Income $20M
Q4 2024: Revenue $140M, Net Income $22M

Department Breakdown:
- Sales: $200M revenue, 150 employees
- Engineering: $150M revenue, 200 employees
- Marketing: $100M revenue, 50 employees
- Operations: $50M revenue, 100 employees

""" * 50)
)  # Repeat to make it very long


def main():
    """Run RLM with two models."""
    # Use GPT-4o for root decisions, GPT-4o-mini for recursive processing
    # This can significantly reduce costs while maintaining quality
    rlm = RLM(
        model="gpt-5-mini",  # Root model (expensive, smart)
        recursive_model="gpt-5-mini",  # Recursive model (cheap, fast)
        max_iterations=15,
        max_depth=3,
        temperature=0.3,
    )

    queries = [
        "What was the total revenue for 2024?",
        "Which quarter had the highest net income?",
        "How many total employees does the company have?",
    ]

    print("Using two-model strategy:")
    print("  Root: gpt-4o (expensive, for main reasoning)")
    print("  Recursive: gpt-5-mini (cheap, for sub-tasks)\n")
    print(f"Document length: {len(long_document):,} characters\n")

    for query in queries:
        print(f"Query: {query}")

        try:
            result = rlm.complete(query, long_document)

            print(f"Answer: {result}")
            print(
                f"Stats: {rlm.stats['llm_calls']} calls, "
                f"{rlm.stats['iterations']} iterations, "
                f"depth {rlm.stats['depth']}"
            )
            print()

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # Set both API keys if using different providers:
    # export OPENAI_API_KEY="sk-..."
    # export ANTHROPIC_API_KEY="sk-ant-..."
    main()
