"""Measure repeated REPL state synchronization with a large worker variable."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rlm.repl import REPLExecutor


def main() -> None:
    """Run a deterministic local IPC benchmark and print one JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-bytes", type=int, default=5_000_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--snapshot-bytes", type=int)
    args = parser.parse_args()
    if args.payload_bytes <= 0 or args.iterations <= 0:
        parser.error("payload-bytes and iterations must be greater than zero")

    kwargs = {}
    if args.snapshot_bytes is not None:
        kwargs["max_snapshot_bytes"] = args.snapshot_bytes
    executor = REPLExecutor(timeout=10, **kwargs)
    parent_env = {}
    timings = []
    try:
        executor.execute(f"payload = 'x' * {args.payload_bytes}", parent_env)
        for _ in range(args.iterations):
            started = time.perf_counter()
            executor.execute("payload_size = len(payload)", parent_env)
            timings.append(time.perf_counter() - started)
        found, payload = executor.get_variable("payload")
    finally:
        executor.close()

    print(
        json.dumps(
            {
                "payload_bytes": args.payload_bytes,
                "iterations": args.iterations,
                "snapshot_bytes": args.snapshot_bytes,
                "parent_has_payload": "payload" in parent_env,
                "worker_payload_preserved": found and len(payload) == args.payload_bytes,
                "median_step_seconds": statistics.median(timings),
                "total_step_seconds": sum(timings),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
