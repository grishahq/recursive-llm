# Post-fix Codex CLI checks — October 2, 2026

These are bounded integration checks of the runtime fixes, run by separate review
agents through Codex CLI 0.159.0 with `gpt-6.1-sol`, `low` reasoning, and ChatGPT
subscription authentication. There is no API fallback or answer-key feedback.
Raw requests, provider events, answers, and unsuccessful answers are retained.
This small sample does not establish an architecture or model-quality advantage.

## Runtime and repository validation

- Python 3.12.14: **321 tests passed**, including the PDF parser path; branch-aware
  coverage **93.52%** (required: 90%). Final full run: 47.81 seconds.
- Black 25.11.0, Ruff 0.16.0, and mypy passed. The offline demo passed.
- Wheel and source distribution built; an isolated installation of the wheel
  passed import, tuple-assignment, and deleted-variable smoke checks.
- The downloadable companion ZIP matches all 20 current bundled source files.
- A fresh frontend installation with Node 24.19.0 passed type checking, lint,
  **9 tests**, and the production build. `npm audit` reported **0 vulnerabilities**
  at validation time. This is dependency-advisory coverage, not a security audit.
- CI now checks the frontend and installs the optional PDF parser on Python 3.12
  Linux. Other supported Python/OS combinations await the remote CI matrix.

The regression coverage exercises shared budget exhaustion, cancellation and
callback draining, stable final diagnostics, deletion-aware REPL recovery,
tuple unpacking, strict field graders, and provider-free REPL worker imports.

## Live integration scenarios

[Scenario artifacts](attempt-094841-423455/summary.json) and their
[manifest](attempt-094841-423455/manifest.json) record both first attempts:

| Scenario | Result | Model calls | Total CLI tokens | Elapsed |
| --- | --- | ---: | ---: | ---: |
| Root REPL tuple assignment, depth limit 0 | `id=TX-B amount=29` | 3 | 14,171 | 36.837 s |
| Child RLM delegation, depth limit 2 | `release_label=ORCHID-4289` | 4 | 18,840 | 56.559 s |

Both scenarios completed without REPL errors or Codex host-tool use. The child
scenario explicitly instructs the root to delegate and the child to inspect its
own REPL context. Its trajectory reaches child depth 1. It verifies that path,
not whether the model would choose recursion or benefit from it unprompted.
These two checks preceded the final lazy-import optimization; their manifests
identify the tested runtime files. [run_integration.py](run_integration.py)
reproduces them and creates a new directory on every invocation.

## Paired generated-source checks

The direct and RLM modes share the model, reasoning effort, query, source, and
transport wrapper. Direct mode runs first. The synthetic source uses seed 2026;
RLM has a depth limit of 1. All observed RLM calls in the initial pairs were root
calls. Four transaction fields are checked using `transaction-fields-v2`.

| Attempt | Mode | Fields correct | Total CLI tokens | Calls | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial 10,104 characters | Direct | 4/4 | 7,752 | 1 | 11.413 s |
| Initial 10,104 characters | RLM | 4/4 | 16,425 | 3 | 29.943 s |
| Initial 100,098 characters | Direct | 3/4 | 38,650 | 1 | 33.941 s |
| Initial 100,098 characters | RLM | 4/4 | 16,482 | 3 | 35.570 s |
| Final 100,098 characters | Direct | 4/4 | 38,790 | 1 | 33.133 s |
| Final 100,098 characters | RLM | 4/4 | 16,485 | 3 | 31.720 s |

The initial 100k direct answer reported `28150192` for the total amount; the
deterministic answer is `27404392`. The incorrect answer remains in both the raw
and replay recordings. Neither initial RLM run had REPL errors.

[summary.json](summary.json) retains the initial pairs and their provenance.
`run_state.py` changed while those pairs were being captured, so they are
explicitly working-tree smoke tests. A separate final 100k pair was requested
after the callback and lazy-import fixes were frozen, with unchanged task and
model settings, to verify the final runtime. It does not replace either earlier
attempt. [final-runtime.json](final-runtime.json) records its exact command,
runtime versions, and source hashes before and after capture.

The [final recording](comparison-100k-final.json) and its
[raw trace](comparison-100k-final.raw.json) both completed successfully. RLM used
three root calls and no child calls, with no REPL errors. The final pair checks
the frozen runtime; the direct model's different answer on the same source also
illustrates why single-pair scores are not reliable quality estimates.

Token totals include cached input and CLI instruction overhead. They are not
subscription credits, dollar costs, or a measurement of allowance consumed.
Elapsed times include startup and network latency; they are single observations.

To create a new pair (consumes the signed-in account's Codex allowance):

```sh
python examples/capture_codex_comparison.py --model gpt-6.1-sol \
  --reasoning-effort low --max-depth 1 --chars 100000 --seed 2026 \
  --output comparison-new.json
```
