"""Example showing advanced configuration options."""

from rlm import RLM
import asyncio

# Sample context
context = """
Technical Specifications - Server Configuration

Server A:
- CPU: AMD EPYC 7763 (64 cores)
- RAM: 512GB DDR4
- Storage: 4x 2TB NVMe SSD
- Network: 10 Gbps
- OS: Ubuntu 22.04 LTS
- Location: US-East-1
- Status: Active
- Uptime: 99.98%

Server B:
- CPU: Intel Xeon Gold 6348 (28 cores)
- RAM: 256GB DDR4
- Storage: 8x 1TB SATA SSD
- Network: 10 Gbps
- OS: CentOS 8
- Location: US-West-2
- Status: Active
- Uptime: 99.95%

Server C:
- CPU: AMD EPYC 7543 (32 cores)
- RAM: 128GB DDR4
- Storage: 2x 4TB HDD
- Network: 1 Gbps
- OS: Ubuntu 20.04 LTS
- Location: EU-Central-1
- Status: Maintenance
- Uptime: 98.50%
"""


async def async_example():
    """Example using async API for better performance."""
    print("Async Example\n")

    rlm = RLM(model="gpt-5-mini", max_iterations=10, temperature=0.3)

    # Process multiple queries in parallel
    queries = [
        "Which server has the most RAM?",
        "Which server has the highest uptime?",
        "List all servers in US locations",
    ]

    # Run queries concurrently
    tasks = [rlm.acomplete(q, context) for q in queries]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result}\n")


def custom_params_example():
    """Example with custom LLM parameters."""
    print("\nCustom Parameters Example\n")

    rlm = RLM(
        model="gpt-5-mini",
        max_iterations=15,
        max_depth=3,
        max_retries=2,
        # Custom LiteLLM parameters
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        timeout=30,
    )

    query = "Describe the storage configuration across all servers"
    result = rlm.complete(query, context)

    print(f"Query: {query}")
    print(f"Result: {result}")


def local_model_example():
    """Example with local llama.cpp server."""
    print("\nLocal Model Example (llama.cpp)\n")

    # Assumes llama.cpp server running on localhost:8000
    rlm = RLM(
        model="openai/local",
        api_base="http://localhost:8000/v1",
        max_iterations=10,
        temperature=0.7,
    )

    query = "Which server should I use for high-memory workloads?"

    try:
        result = rlm.complete(query, context)
        print(f"Query: {query}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error (is llama.cpp server running?): {e}")


def error_handling_example():
    """Example with error handling."""
    print("\nError Handling Example\n")

    rlm = RLM(model="gpt-5-mini", max_iterations=3, max_depth=2)  # Very low to trigger timeout

    from rlm import MaxIterationsError, MaxDepthError

    try:
        # This might exceed iterations
        result = rlm.complete("Perform detailed analysis of all servers", context)
        print(f"Result: {result}")

    except MaxIterationsError as e:
        print(f"Max iterations exceeded: {e}")
        print("Consider increasing max_iterations or simplifying the query")

    except MaxDepthError as e:
        print(f"Max depth exceeded: {e}")
        print("Consider increasing max_depth")

    except Exception as e:
        print(f"Other error: {e}")


def stats_example():
    """Example tracking statistics."""
    print("\nStatistics Tracking Example\n")

    rlm = RLM(model="gpt-5-mini", max_iterations=15)

    query = "Compare CPU specs across all servers"
    result = rlm.complete(query, context)

    print(f"Query: {query}")
    print(f"Result: {result}\n")

    # Check statistics
    stats = rlm.stats
    print("Execution Statistics:")
    print(f"  Total LLM calls: {stats['llm_calls']}")
    print(f"  REPL iterations: {stats['iterations']}")
    print(f"  Recursion depth: {stats['depth']}")


def main():
    """Run all examples."""
    # Async example
    asyncio.run(async_example())

    # Custom parameters
    custom_params_example()

    # Local model (optional)
    # local_model_example()

    # Error handling
    error_handling_example()

    # Statistics
    stats_example()


if __name__ == "__main__":
    main()
