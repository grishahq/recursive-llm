# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Opt-in, budget-aware retries for transient provider failures with exponential backoff,
  `Retry-After` support, exact retry statistics, and trajectory events.
- Versioned JSONL export for structured completion results with secret-free run configuration.
- Final-answer validation with deterministic model feedback and support for directives, REPL
  variables, and mutable answer publication.
- A reproducible 1M-character scale check with exact grading and live GPT-5 mini and DeepSeek V4
  Flash results.
- A SHA-pinned real-document benchmark over the public-domain English translation of *War and
  Peace*, with exact graders for structure, distant retrieval, and narrative evidence synthesis.
- A read-only `RLM.trajectory` snapshot that preserves partial events from failed runs.

### Changed

- Normalized null provider text content into the existing empty-response repair path and added
  explicit errors for malformed provider response structures.
- Disabled hidden LiteLLM retries so every real retry is governed by tree-wide budgets.
- Preserved content-bearing benchmark trajectories when a traced run raises an exception.

## [0.2.0] - 2026-07-15

### Migration notes

- Replace `RLM.completion(...)` with `RLM.complete(...)` and `RLM.acompletion(...)` with
  `RLM.acomplete(...)`.
- The default `max_depth` is now `1` instead of `5`. Depth follows the paper's capability-based
  convention: `0` enables only the root REPL, `1` permits plain-LM subcalls, and `2` permits one
  child RLM level with a plain-LM boundary fallback.

### Added

- Tree-wide call, token, cost, and elapsed-time budgets with partial statistics on budget errors.
- Structured completion results with exact per-run statistics and root/child/leaf trajectories.
- Safe concurrent use of the same `RLM` instance through isolated per-invocation state.
- Optional POSIX memory, CPU-time, and open-file limits for REPL workers.
- Deterministic long-context benchmark generation, exact task graders, repeated runs, direct-model
  baselines, JSONL output, and checked-in live benchmark results.
- GitHub Actions checks across Python 3.9-3.12 on Linux and Python 3.12 on macOS and Windows.
- Security guidance describing the REPL isolation model and its trust boundary.

### Changed

- Aligned recursion-depth behavior and documentation with the RLM paper's depth convention.
- Hardened REPL execution with spawned workers, persistent state, hard local-step timeouts, bounded
  output, restricted imports, and ordered bounded-concurrency subcalls.
- Aggregated usage and best-effort cost statistics across the complete recursion tree.
- Updated repository, installation, citation, release, and issue links to `grishahq/recursive-llm`.
- Pinned the formatter to a Python 3.9-compatible version and updated GitHub Actions to Node 24-based
  releases for reproducible, warning-free CI runs.
- Expanded the test suite from 43 initial-release tests to 135 tests with enforced branch coverage.

### Fixed

- Required final-answer directives to be standalone executable statements instead of accepting
  occurrences embedded in arbitrary text.
- Prevented models from guessing context contents before inspecting the REPL context.
- Corrected parameter handling for GPT-5-family models.
- Made persistent REPL variables visible inside comprehension bodies on Python 3.9-3.11 while
  keeping restricted runtime helpers out of parent snapshots.
- Prevented REPL worker pipe errors from leaking tracebacks during budget-triggered shutdown.
- Corrected offline-demo aggregation and strengthened benchmark numeric-boundary grading.

## [0.1.0] - 2025-10-17

- Initial public release.

[0.2.0]: https://github.com/grishahq/recursive-llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/grishahq/recursive-llm/releases/tag/v0.1.0
