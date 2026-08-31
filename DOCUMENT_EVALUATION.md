# Document-format evaluation

This report records an engineering evaluation performed on August 31, 2026. It asks whether the
project works on real documents in several formats, how RLM compares with a single direct model
call, and whether two isolated implementation changes improve reliability or local overhead
without breaking document tasks.

The live sample contains one primary run per task and targeted retries only for failed tasks. Model
behavior is stochastic, so these results can catch obvious regressions but cannot establish a
general quality improvement. PyPI publication was deliberately excluded from this work.

## Reproducible corpus

`benchmarks/document_formats.py` downloads every source, verifies its raw SHA-256, prepares a
deterministic text context, and records the prepared-context SHA-256 in every result.

| Format | Public source | Raw SHA-256 | Prepared characters | Context SHA-256 |
| --- | --- | --- | ---: | --- |
| TXT | [Project Gutenberg, *Frankenstein*](https://www.gutenberg.org/cache/epub/84/pg84.txt) | `7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3` | 438,841 | `1888b0938bcfc8e127bf00c22054f06c7c999391c11ceeb3c2ffbff73b264fc4` |
| PDF | [NIST AI RMF Playbook](https://airc.nist.gov/docs/AI_RMF_Playbook.pdf) | `65d6101d806502875aadb0fd19a75c3a9cc9a5e9461129e9398a39192d8202d2` | 346,329 | `96f3fff4953b9a566a3bdb0b149dc6c1e16dc82f16c2384a702aa55fc2e491d7` |
| CSV | [NIST AI RMF Playbook CSV](https://airc.nist.gov/docs/playbook.csv) | `3cee552201b18192e042fb94b72fe8da9395c91d36877ac7d9afb9cccb352b3a` | 357,864 | `def2b4b51c644ce306ff73a48d9d186fed0ef1cc5aee053a100ba87749b013f7` |
| HTML | [Python `sqlite3` documentation](https://docs.python.org/3/library/sqlite3.html) | `cd7cdb039f49bdd5866193acf810482a16954bf7f41b31b4dcb0a4a725f51032` | 71,074 | `593728680dade87d4e7c15d06bfdee0edccdf8875c97e18b7d21ad4ba2ab0d62` |

TXT input is decoded and line-ending normalized. The 147-page PDF is extracted with
`pdftotext -layout`; selected beginning and ending pages were rendered and visually compared with
the extraction. The CSV is normalized from its source layout of five rows and 73 columns into one
record per subcategory. HTML is converted with a deterministic standard-library parser that drops
scripts, styles, and navigation before whitespace normalization.

The exact tasks cover book structure and distant retrieval, the 19/13/18/22 NIST function
subcategory counts in two source formats, and four documented `sqlite3.connect` defaults. Unit
tests verify that exact answers pass and plausible near misses fail.

## Baseline comparison

Both modes used `gpt-5-mini`, one run per task, and the same exact graders. RLM used depth 1, at most
12 iterations, 16 calls, 2,000 output tokens per call, and 180 elapsed seconds per task.

| Mode | Exact passes | p50 latency | p95 latency | Calls | Model tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 4/4 | 13.841 s | 15.529 s | 4 | 271,830 | $0.0740230 |
| RLM | 4/4 | 72.463 s | 115.752 s | 25 | 59,358 | $0.0457297 |

For these tasks, RLM reduced model-token use by 78.2% and estimated cost by 38.2%, but its median
latency was 5.24x direct mode and it needed 6.25 calls per task. The practical decision is not
“always use RLM”: use direct completion when latency matters and the document fits comfortably in
the model context; use RLM when externalizing a large context or bounding sent tokens/cost matters
more than call count and latency.

## Change 1: tree-wide deadline

Before the change, a run configured with a 30 ms elapsed limit could invoke an 80–150 ms synchronous
final-answer validator and still return success. REPL startup/execution and variable exchanges also
used only their local timeout rather than the remaining tree time.

The retained implementation now:

- caps each REPL exchange by the smaller of `repl_timeout` and the remaining tree time;
- reports a deadline-limited REPL timeout as `BudgetExceededError(metric="elapsed_seconds")`;
- checks the deadline after a synchronous final-answer validator before accepting its result;
- preserves the old local-timeout behavior when no tree deadline is configured.

The late-validator regression now raises `BudgetExceededError`; an infinite REPL step with a
150 ms tree limit and a 2 s local limit stops in under one second. The full suite passed 184/184
with 92.65% coverage after the change.

The primary four-document run passed 3/4. *Frankenstein* reached the 12-iteration limit after
165.192 s; a targeted unchanged retry passed in 64.568 s with six calls. Because the failure was
`MaxIterationsError`, not a deadline or REPL error, and the identical retry passed, this is treated
as model variance rather than evidence against the fix.

A synchronous validator still cannot be safely killed inside its thread. Its late answer is no
longer accepted, but slow application validator code can delay delivery of the budget error.

## Change 2: bounded REPL parent snapshot

The worker previously pickled and copied every serializable user variable to the parent after every
step. Large intermediate strings or lists remain available in the persistent worker, so repeatedly
transferring them is unnecessary for normal model execution.

The retained `repl_max_snapshot_bytes` option defaults to 1,000,000 bytes. Snapshot selection
prefers smaller values; omitted values remain complete in the worker and available to later REPL
code and explicit `FINAL_VAR` lookup. If a previously small variable grows beyond the cap, its stale
parent copy is removed instead of being left as a misleading fallback.

`benchmarks/repl_snapshot.py` measured ten repeated steps while a 5 MB string remained in the
worker. The table reports the median of three independent series.

| Parent snapshot | Median repeated step | Total for ten steps | Parent copied 5 MB | Worker retained 5 MB |
| --- | ---: | ---: | --- | --- |
| Unbounded baseline | 16.691 ms | 217.641 ms | yes | yes |
| 1 MB limit | 0.718 ms | 8.842 ms | no | yes |

The median repeated-step time fell by 95.7% and the ten-step total by 95.9% in this synthetic IPC
case. The full suite passed 186/186 with 92.54% coverage.

The primary four-document run again passed 3/4. CSV returned 22/16/21/26 instead of
19/13/18/22; the targeted unchanged retry passed. The first answer added exactly three to each
category and completed normally, with no deadline, REPL, or missing-variable error. This is recorded
as model-counting variance. The primary aggregate is not replaced by the retry and no live latency,
token, or quality gain is attributed causally to the snapshot change.

## Independent Codex CLI verification

Codex CLI 0.149.0 was run non-interactively with an output schema. `workspace-write` was required
because pytest and Python multiprocessing need writable temporary directories; the verification
prompt prohibited project edits.

The baseline verification passed 52/52 selected tests and confirmed four valid baseline result
records plus one matching summary. Final verification passed 103/103 selected tests and confirmed:

- all five staged JSONL files were structurally valid;
- summary totals matched their result records;
- source URLs and both SHA-256 fields were present and consistent across variants;
- the two primary post-change failures and both successful targeted retries were represented
  accurately.

Example invocation:

```bash
codex exec --ephemeral --sandbox workspace-write --color never \
  --output-schema benchmarks/codex_evaluation.schema.json \
  -o /tmp/rlm-codex-verification.json \
  'Do not edit project files. Run the selected pytest command and verify the benchmark JSONL.'
```

## Reproduction

The preparation command downloads sources and refuses a raw hash mismatch:

```bash
python benchmarks/document_formats.py --prepare /tmp/rlm-document-formats
```

Run the deterministic harness tests and the two modes:

```bash
pytest -q tests/test_document_formats_benchmark.py --no-cov

python benchmarks/document_formats.py gpt-5-mini /tmp/rlm-document-formats \
  --runs 1 --label current --mode rlm --max-depth 1 --max-iterations 12 \
  --max-total-calls 16 --max-elapsed-seconds 180 --jsonl rlm-results.jsonl

python benchmarks/document_formats.py gpt-5-mini /tmp/rlm-document-formats \
  --runs 1 --label direct --mode direct --max-elapsed-seconds 180 \
  --jsonl direct-results.jsonl

python benchmarks/repl_snapshot.py --payload-bytes 5000000 --iterations 10 \
  --snapshot-bytes 1000000
```

## Decisions and roadmap

Retained now:

- strict deadline outcome semantics, because it fixes a deterministic correctness bug;
- bounded parent snapshots, because the worker-state guarantee is tested and local IPC overhead
  drops materially for large intermediates;
- exact format graders, pinned hashes, failed-run preservation, and Codex CLI verification, because
  they make later comparisons auditable.

Not adopted or not claimed:

- no prompt/search-helper change was mixed into this evaluation, because it would prevent causal
  comparison and earlier multi-document candidates reduced pass counts;
- no quality improvement is claimed from the post-change live numbers: one primary run per task is
  too small and both stages exhibited a retry-sensitive failure;
- PyPI publication remains deferred. Source/Git installation is sufficient while the project is
  alpha, default behavior is still being validated, and the release gate needs a larger repeated
  matrix.

Recommended next work, in order:

1. Run at least five repetitions per format across two provider families; report confidence
   intervals and separate first-run results from retries.
2. Add the deterministic document preparation, graders, deadline regressions, and snapshot
   microbenchmark to CI; keep paid live calls scheduled or manually gated.
3. Add opt-in snapshot telemetry (bytes copied and omitted-variable count) to structured results so
   real workloads can tune the 1 MB default from evidence.
4. Improve semantic navigation over long prose and deliberately evaluate child-RLM use; current
   successful document runs often rely on root-level search and computation.
5. Add adversarial tests for validator latency, callback latency, huge nested objects, worker
   restart, and sensitive error redaction.
6. Revisit PyPI only after the repeated format matrix is stable, public API/documentation review is
   complete, and release automation can verify wheels on Python 3.9–3.12 and major platforms.
