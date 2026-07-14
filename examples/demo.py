#!/usr/bin/env python
"""Quick demonstration of local RLM components."""

import re

from rlm.parser import extract_final, is_final
from rlm.repl import REPLExecutor


def main() -> None:
    """Run the local demonstration without making model API calls."""
    print("=" * 60)
    print("RLM Library Demo")
    print("=" * 60)
    print()

    print("1. REPL Executor Demo")
    print("-" * 60)
    context = """
Machine Learning Report 2024

Q1 Revenue: $1.2M
Q2 Revenue: $1.5M
Q3 Revenue: $1.8M
Q4 Revenue: $2.1M

Total: $6.6M
"""
    env = {"context": context, "re": re}

    with REPLExecutor() as repl:
        extraction_code = """
revenues = re.findall(r'\\$([\\d.]+)M', context)
print(f"Found revenues: {revenues}")
"""
        print("Code:")
        print(extraction_code)
        print("Output:", repl.execute(extraction_code, env))
        print()

        calculation_code = """
revenue_values = [float(revenue) for revenue in revenues]
total = sum(revenue_values)
print(f"Total revenue: ${total}M")
"""
        print("Code:")
        print(calculation_code)
        print("Output:", repl.execute(calculation_code, env))
        print()

    print("2. Response Parser Demo")
    print("-" * 60)
    response = 'FINAL("The total revenue is $6.6M")'
    print("LLM Response:")
    print(response)
    if is_final(response):
        print("Detected FINAL statement!")
        print(f"Extracted answer: {extract_final(response)}")
    else:
        print("No FINAL statement detected")
    print()

    print("3. Context as Variable Demo")
    print("-" * 60)
    print("Instead of embedding a large context in every prompt, RLM exposes it as a variable:")
    print("  env = {'context': huge_document, 'query': query}")
    print()
    print("The model can then interact with it programmatically:")
    print("  - context[:100]  # Peek at the start")
    print("  - re.findall(pattern, context)  # Search")
    print("  - rlm_query(query, context[1000:2000])  # Recurse")
    print()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("To use RLM with a real model:")
    print("  from rlm import RLM")
    print("  rlm = RLM(model='gpt-5-mini')")
    print("  result = rlm.complete(query, long_document)")


if __name__ == "__main__":
    main()
