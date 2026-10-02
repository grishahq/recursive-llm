# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- A side-by-side web visualizer and downloadable film of real RLM and direct GPT-5.6 Luna runs,
  with synchronized replay, call trees, recorded Python steps, import, and JSON export.
- An optional async completion handler, propagated across child RLMs, and a bounded Codex CLI
  transport for ChatGPT-authenticated subscription experiments without API keys.
- A reproducible paired capture script, 10k and 100k generated-source recordings, and an explicit
  methodology distinguishing total CLI tokens from subscription billing.

### Changed

- Made the paired Codex capture model, reasoning effort, and recursion depth configurable while
  retaining the historical Luna/medium/depth-1 defaults and recording the actual settings.
- Versioned stricter generated-transaction and document fact graders; historical benchmark scores
  remain labeled as results of their original grading rules.
- Added frontend type, test, and build checks and Python PDF coverage to CI.
- Updated the visualizer's React, Vite, Vinext, and Cloudflare dependencies to patched,
  peer-compatible releases, with a scoped Satori decompressor override.

### Fixed

- Stopped admitting calls after a shared usage budget is exhausted and drained cancelled callbacks
  before publishing the final run statistics and trajectory.
- Removed deleted REPL variables from recovery snapshots and prevented final answers from using
  an obsolete parent snapshot after worker loss.
- Enabled ordinary, nested, and starred assignment unpacking in the restricted Python worker.
- Separated process-readiness allowances from cancellation assertions in coverage-instrumented tests.
- Deferred the public `RLM` import so standalone REPL workers start without importing LiteLLM.
- Kept the visualizer's required `lib` sources visible to Git.

## [0.4.0] - 2026-08-31

### Added

- A SHA-pinned exact-graded document benchmark covering a Project Gutenberg TXT book, the NIST AI
  RMF Playbook in PDF and CSV forms, and the official Python sqlite3 HTML documentation.
- A deterministic REPL snapshot IPC benchmark and structured Codex CLI verification schema.

### Changed

- Capped parent-side REPL state snapshots at 1 MB by default while preserving complete variables
  in the persistent worker and explicit variable retrieval.

### Fixed

- Applied the shared elapsed-time budget to REPL exchanges and prevented answers completed by a
  late final-answer validator from being accepted after the tree deadline.
- Capped LiteLLM below 1.98.0 because that release imports Python 3.11-only typing symbols during
  startup and breaks the supported Python 3.9 and 3.10 environments.

## [0.3.1] - 2026-07-28

### Fixed

- Pinned Ruff 0.16.0 and made the stable baseline rules explicit for reproducible local and CI
  checks.

## [0.3.0] - 2026-07-25

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
- Non-raising `try_complete_result` and `atry_complete_result` APIs with typed, versioned failure
  records and exact per-run diagnostics.
- A SHA-pinned three-document benchmark covering a 3.2M-character novel, a 2.0M-character official
  report, and a 14.6M-character documentation corpus.

### Changed

- Normalized null provider text content into the existing empty-response repair path and added
  explicit errors for malformed provider response structures.
- Disabled hidden LiteLLM retries so every real retry is governed by tree-wide budgets.
- Preserved content-bearing benchmark trajectories when a traced run raises an exception.
- Updated the benchmark runner to consume success and failure results through one structured API.

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

[Unreleased]: https://github.com/grishahq/recursive-llm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/grishahq/recursive-llm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/grishahq/recursive-llm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/grishahq/recursive-llm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/grishahq/recursive-llm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/grishahq/recursive-llm/releases/tag/v0.1.0
