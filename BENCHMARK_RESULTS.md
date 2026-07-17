# Benchmark Results

These live measurements were collected on July 15, 2026 with Python 3.12.13. They are a small
engineering check, not a paper reproduction. Provider behavior is stochastic, only one generated
corpus seed was used, and prices are LiteLLM estimates.

## Unreleased Reliability Validation

The provider-retry, JSONL-export, and final-answer-validation changes were checked on July 16, 2026
against commit `71a5e43` and the updated working tree.

Deterministic fault injection changed the intended failure paths without changing defaults:

| Scenario | `71a5e43` | Updated behavior |
| --- | --- | --- |
| HTTP 429 followed by success | Run aborted on the 429 | Recovered with two counted calls and one retry |
| Invalid final answer followed by a correction | Invalid answer was accepted | Validator feedback produced the corrected answer in two calls |
| Structured run persistence | No result exporter | One versioned, redacted JSONL record per completed run |

With retries and validation left at their defaults, a mocked healthy-provider microbenchmark took
111.499 microseconds per completion on `71a5e43` and 115.053 microseconds after the changes. The
3.2% relative increase is 3.554 microseconds per completion and is negligible beside provider
latency.

One live exact-grader smoke run per previously tested model also passed without retries:

| Model | Passed | Latency | Calls | Retries | Tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini | 1/1 | 29.712 s | 4 | 0 | 3,472 | $0.00362075 |
| DeepSeek V4 Flash | 1/1 | 17.883 s | 7 | 0 | 4,237 | $0.000466816 |

These smoke runs verify provider integration after retry accounting changed; they are not a new
quality comparison because model trajectories are stochastic and each model ran only once.

## Reproduction

```bash
# Small deterministic tasks, three repetitions each
python benchmarks/compare_same_model.py MODEL --full --runs 3 --mode direct
python benchmarks/compare_same_model.py MODEL --full --runs 3 --mode rlm --max-depth 2

# One 100k-character deterministic task, three repetitions
python benchmarks/compare_same_model.py MODEL --generated-chars 100000 --seed 2026 --runs 3 --mode direct
python benchmarks/compare_same_model.py MODEL --generated-chars 100000 --seed 2026 --runs 3 --mode rlm --max-depth 2

# A larger RLM-only scale check
python benchmarks/compare_same_model.py MODEL --generated-chars 1000000 --seed 2026 --runs 3 --mode rlm --max-depth 2

# A SHA-pinned public-domain real document
curl -L https://www.gutenberg.org/files/2600/2600-0.txt -o /tmp/war-and-peace-2600-0.txt
python benchmarks/war_and_peace.py MODEL /tmp/war-and-peace-2600-0.txt --max-depth 2
```

The generated context has 100,098 characters and SHA-256
`1ee43f3b42f8db55369c337d2e37f1e7f61224abe3d538581a716865df8a6fcc`. Its exact answer key is:

```text
count=57 total_amount_cents=27404392 max_transaction_id=TX-0000883 max_amount_cents=993620
```

## Small tasks

The direct baseline is the correct choice for these short contexts. Both models passed every direct
run with one call. DeepSeek RLM also passed, but recursion added substantial latency and usage.

| Model and mode | Passed | p50 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, direct | 6/6 | 4.221 s | 1.00 | 598 | $0.0007996 |
| DeepSeek V4 Flash, direct | 6/6 | 2.635 s | 1.00 | 502 | $0.0000929 |
| DeepSeek V4 Flash, RLM depth 2 | 6/6 | 28.204 s | 7.17 | 6,875 | $0.0008071 |
| GPT-5 mini, RLM depth 2, incident task only | 1/3 | 167.341 s | 18.33 | 25,830 | $0.0262083 |

The GPT-5 mini RLM incident runs passed once and reached the configured 24-call cap twice. The full
small-task RLM suite was not repeated after that result because additional calls would not answer a
useful engineering question. The cap prevented a 25th provider request in both failed runs.

## Generated 100k-character task

RLM materially improved long-context correctness. GPT-5 mini also used about 78% fewer tokens and
46% lower estimated cost per run than its direct baseline. DeepSeek used about 61% fewer tokens and
17% lower estimated cost, but one RLM answer omitted the `TX-` prefix and failed exact grading.

| Model and mode | Passed | p50 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, direct | 0/3 | 35.278 s | 1.00 | 37,928 | $0.0088788 |
| GPT-5 mini, RLM depth 2 | 3/3 | 26.716 s | 5.00 | 8,224 | $0.0048132 |
| DeepSeek V4 Flash, direct | 0/3 | 20.738 s | 1.00 | 39,364 | $0.0012240 |
| DeepSeek V4 Flash, RLM depth 2 | 2/3 | 20.624 s | 7.33 | 15,209 | $0.0010156 |

The generated suite originally inherited a six-iteration benchmark cutoff. Raising only this suite
to ten iterations improved the combined RLM pass rate from 4/6 to 5/6: GPT-5 mini improved from 2/3
to 3/3, while DeepSeek remained at 2/3. This setting is now the generated-suite default; the global
24-call and 300-second caps remain unchanged.

## Generated 1M-character scale check

The larger corpus was generated on July 17, 2026 with the same seed. It has 1,000,066 characters,
approximately 337,800 tokens according to LiteLLM's model token counters, and SHA-256
`076e6d74f9917a3ed7f0b17c0f3be49ee49ff98187d7e9918c716aa0a8074ae0`. Its exact answer key is:

```text
count=563 total_amount_cents=275857555 max_transaction_id=TX-0004073 max_amount_cents=999063
```

All six RLM runs passed exact grading without provider retries. A direct baseline was deliberately
not run: this check measures whether externalized context continues to work at the larger scale,
not whether sending the entire corpus to a model is economical.

| Model and mode | Passed | p50 latency | p95 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, RLM depth 2 | 3/3 | 42.744 s | 97.755 s | 7.00 | 17,613 | $0.0107284 |
| DeepSeek V4 Flash, RLM depth 2 | 3/3 | 32.203 s | 52.680 s | 9.33 | 17,538 | $0.0012495 |

Increasing the source context by roughly 10x increased mean model-token usage by 2.14x for GPT-5
mini and 1.15x for DeepSeek relative to the 100k-character RLM runs. Median latency increased by
1.60x and 1.56x, respectively. The individual GPT-5 trajectories remained high variance: 4 to 10
calls and 19.448 to 97.755 seconds. Every successful run used only the root REPL, with
`max_depth_reached=0` in all results. This experiment therefore validates externalized context and
local computation rather than recursive semantic decomposition.

## Real 3.2M-character document

The real-document check used the public-domain English translation of *War and Peace*, Project
Gutenberg eBook #2600. The downloaded file has SHA-256
`e4bcf9042609b62c7de72a6f1b311f54c412943a9d641b7efcf79a464b5f31c8`; after UTF-8 decoding and
line-ending normalization it contains 3,227,519 characters. LiteLLM's local counters estimate
769,910 GPT-5 mini tokens and 777,114 DeepSeek V4 Flash tokens in the source document.

Three tasks exercise different long-document behaviors:

- exact counting of 365 narrative-body chapter headings;
- retrieval of three facts located near the beginning and end of the novel;
- synthesis of five pieces of narrative evidence from Petya's final-night sequence.

The results below use a 20-iteration cap. A direct baseline was not sent because it would require
placing roughly 770,000 source tokens in one provider request.

| Model and mode | Passed | p50 latency | p95 latency | Mean calls | Mean tokens | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5 mini, RLM depth 2 | 2/3 | 90.757 s | 117.601 s | 15.00 | 68,633 | $0.0234759 |
| DeepSeek V4 Flash, RLM depth 2 | 2/3 | 51.055 s | 100.354 s | 12.67 | 58,816 | $0.0027428 |

Both models counted the chapters exactly and retrieved the distant facts exactly. GPT-5 mini
needed 19 iterations for distant retrieval; DeepSeek needed 12. Neither model completed the
narrative synthesis task within 20 iterations, so these were execution failures rather than
incorrect final answers.

A content-bearing failure trace showed the main weakness clearly: the model found the death scene
but spent later iterations on broad `Cossack` searches, a literal `Petya` search that missed
`Pétya`, large match dumps, and a character-by-character scan. A proposed generic search-efficiency
prompt was tested at the original 12-iteration cap, did not improve the pass rate, and increased
token use on the narrative task, so it was reverted. The retained improvement is observability:
failed traced runs now preserve their partial trajectory instead of returning an empty event list.

All measured calls stayed at the root REPL with `max_depth_reached=0`. The benchmark therefore
identifies a concrete next problem: efficient semantic navigation and deliberate child-model use
over real prose, rather than raw context capacity alone.

## Interpretation

- Use direct completion for short contexts that fit comfortably in the model window.
- RLM is useful here when exact computation over a large context matters; the context stays outside
  model prompts and Python performs the aggregation.
- `max_depth=2` did not force recursion. The successful generated runs used the root REPL only, which
  is an important reminder that RLM's value includes externalized context and local computation, not
  only child-model calls.
- Three repetitions and one seed are enough to catch regressions, but not enough for broad model
  quality claims. Use more seeds and runs for release decisions.
