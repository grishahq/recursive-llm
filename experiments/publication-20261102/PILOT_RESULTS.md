# Grounded document pilot, 2 November 2026

This pilot tests a narrow document workflow: return usable facts or calculation
code together with exact source excerpts. The original scenarios are the
[Cloudflare 10-K request](https://github.com/grishahq/recursive-llm/issues/1) and
the request to [compile distributed PDF rules into Python](https://github.com/alexzhang13/rlm/issues/107).
The historical Cloudflare issue is closed. The synthetic calculation policy is
a controlled toy example, not a reproduction of the other user's private PDF.

The harness is [grounded_documents.py](../../benchmarks/grounded_documents.py).
Model calls use the existing ChatGPT-authenticated Codex CLI, without API fallback.
Gold answers and hidden numeric examples are never supplied to the model.

## Inputs and tasks

[source-manifest.json](pilot-source-manifest.json) records raw SHA-256, extracted page
SHA-256, extraction method, source URL, page count and character count.
[gold.json](pilot-gold.json) contains the finite expected fields and original supporting
spans used only by the offline grader.

- Complete Cloudflare 2024 Form 10-K: 157 physical pages, 591,696 original page
  characters. Its raw SHA-256 is
  `ed58d484853f3a8da0dd2af65461fda3974f002cc3f609edeb749b93c22a6315`.
  Extraction uses `pypdf==6.18.0`. The public PDF opens with a blank password.
  Extra page markers added by earlier experiments explain their larger context
  character count. The complete report is supplied; gold pages are not selected
  in advance for the model.
- Official Python sqlite3 documentation at the CPython `v3.12.0` tag. Original
  UTF-8 RST is one text page and retains all 92,179 text characters.
- Fictional Aurora policy: 24 pages and 52,310 characters. Standard fee,
  nonprofit discount, review times and an overriding exception appear on
  different pages. A second program deliberately uses a similar name and
  different fee. The review owner has an accented name, found using an
  unaccented query. The policy does not give a renewal-fee amount.

The final Sol matrix covers seven tasks in all three modes: nonprofit fee,
review exception, Unicode name, explicit unknown, reusable Python function,
Cloudflare AI products and Cloudflare AI Gateway features. A year lookup and
sqlite3-default lookup are additional available tasks.

The generated function receives a positive application count and nonprofit /
expedited flags, then returns total fee and review time. Grader v2 crosses all
four boolean combinations with four quantities, giving 16 held-out test inputs.
Inputs and expected outputs are kept outside prompts. Execution uses the
existing restricted worker, a two-second timeout, and no added imports. This
finite test does not prove the function correct for all inputs.

## Main results

The frozen seven-task Sol matrix completed all 21 attempts. The source files in
the pre-import snapshots match the after-import manifests. Original inference
captures retain grader v1 results. Every saved answer was then regraded with
v2, which fixes correlated numeric test inputs by using the complete 16-case
cross-product. No inference was repeated for this correction and no answer
aliases were added after observing model output.

[final-sol-regraded-v2.json](pilot-final-sol-regraded-v2.json) retains both grades and a
SHA-256 of every raw answer.

| Mode | Strict field and evidence passes | Structurally valid citations | Model tokens, sum | Task elapsed seconds, sum | Model calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 7/7 | 7/7 | 396,799 | 96.14 | 7 |
| Python/search, depth zero | 6/7 | 7/7 | 182,920 | 214.29 | 29 |
| RLM, depth two allowed | 6/7 | 7/7 | 189,524 | 218.65 | 30 |

All seven RLM-mode attempts used zero child calls. These runs demonstrate the
source interface and source selection; they do not demonstrate an advantage of
recursion. Direct completion was faster on every main task. The small synthetic
tasks also used fewer tokens directly. On the complete Cloudflare report,
Python/search and RLM mode used fewer tokens than the direct input.

The two strict failures are both `cloudflare-products`: Python/search and RLM
returned the correct expanded source description, `embeddings, representations
of values or objects commonly consumed by machine learning models`, while the
finite expected value was exactly `embeddings`. Their quotes and coordinates
are valid and the expanded phrase agrees with the source. These are output
contract failures, not observed factual errors. The strict score remains 6/7;
the manual observation does not change the automatic grade.

All three generated functions pass all 16 v2 numeric tests, including nonprofit
and expedited combinations at the same quantities. This is a successful toy
rules-to-code exercise in all modes, not an RLM-specific quality gain.

Additional real cross-page and model-compatibility checks are stored separately
from the original matrix. The forced-child exercise is an integration check,
excluded from performance comparisons.

The additional complete-report question combines the Workers AI product with
AI-related risk statements on physical pages 12, 37 and 59. All three modes
passed its three fields and their evidence checks:

| Mode | Model tokens | Task elapsed seconds | Child model calls |
| --- | ---: | ---: | ---: |
| Direct | 146,806 | 18.33 | 0 |
| Python/search | 45,349 | 35.68 | 0 |
| RLM allowed | 28,579 | 31.39 | 0 |

This one paired example shows a token / elapsed-time tradeoff on a real report.
It still provides no evidence for recursive decomposition, because no children
were used. The additional question was added after the original matrix and is
retained in [extra-sol-crosspage](pilot-summaries/runs/extra-sol-crosspage/summary.json).

The separately prompted [forced-child exercise](pilot-summaries/runs/integration-sol-forced-child/summary.json)
completed with one child RLM node. Its trajectory contains three source searches
and two source reads inside that child, plus successful final validation at the
child and root. It used six child model calls and four root model calls, 64,043
tokens and 74.28 seconds. This verifies inherited source access and validation;
it is not evidence that forcing a child improves efficiency.

## Compatibility checks

These small checks test the source protocol, not relative model quality. They
use Unicode-name retrieval and the actual official sqlite3 documentation, with
Python/search at depth zero.

| Model and runtime capture | Accepted, strict passes | Verified status |
| --- | ---: | --- |
| `gpt-6.1-terra` | 2/2 | Unicode and sqlite3 defaults completed with valid original-source spans |
| `gpt-6-luna`, before empty-response repair | 0/2 | Unicode hit an empty final CLI message; sqlite3 ended with HTTP 429 |
| `gpt-6-luna`, after transport repair | 0/2 | Both ended with HTTP 429; Unicode found the source but repeatedly emitted bare JSON instead of the required final directive |

The empty-message transport check was changed to pass a single empty string to
the existing RLM repair loop, while retaining token usage. Missing, multiple or
non-string assistant messages and tool use remain rejected. The before / after
captures are retained separately; the later failures do not establish whether
Luna can reliably finish this protocol when service limits are absent.

The [pre-patch Luna audit](pilot-luna-prepatch-audit-v2.json) recovers 11,354 known tokens
from the failed empty-message call's `turn.completed` event. Both pre-patch tasks
therefore consumed at least 28,511 tokens, rather than the original 17,157 summary.
Post-patch tasks consumed at least 45,180 known tokens. Three HTTP-429 calls across
both stages provided no final usage, so these totals are lower bounds. A token
accounting correction does not change their failed status. No further live
retries were made after the two bounded post-repair tasks.

## Final protocol checks

After those compatibility captures, document mode was updated to accept a
standalone strict-schema JSON answer through the same citation and custom
validators. The original 24 performance captures retain their earlier frozen
runtime and prompt. Their statistics are not relabeled as measurements of this
final protocol.

An [offline fixed-response Luna replay](pilot-summaries/runs/replay-luna-native-json/replay.json)
then accepted the fifth recorded response with correct factual value and source
spans. It made zero live model calls. This verifies that the final runtime can
accept the previously captured valid JSON, not that Luna reliably completes a
fresh run under the new prompt. The HTTP-429 limit was not bypassed or retried.

One [fresh final-protocol Sol check](pilot-summaries/runs/final-protocol-sol-crosspage/summary.json)
also passed the real three-page Cloudflare question: 19,829 observed model
tokens, 23.95 seconds, three calls and no child calls. This is a separate live
compatibility smoke. It is not a repeated paired benchmark or evidence that the
protocol change improves performance.

## Comparison protocol

`direct` is one text-only model call with the complete source. It receives exact
precomputed segments with original offsets, so citations can be copied without
manual offset arithmetic. Segment headers add real input overhead, counted in
its reported usage.

`python` uses the same model and complete source with the Python/search runtime
at depth zero. `rlm` allows depth two and shared source helpers. Child calls are
permitted, not forced. A claim about the benefit of recursion requires the
trajectory to show that children were actually called.

All three modes must return the same JSON envelope. The grader separately records
answer correctness, structural citation validity and coverage of the known gold
evidence. A valid quote from an unrelated paragraph does not pass. Supported
claims must use unique labeled fields with finite exact aliases; accents are
preserved. An absent amount must be marked `not specified` and named in
`unknowns`, rather than converted to zero. Function tasks are checked by running
the generated function, not by matching code text.

The final Sol runs use low reasoning, at most 16 calls, 10 iterations and
300 seconds per task, with a 180-second per-CLI-call timeout. The direct mode is
bounded to one call and the same wall time; the text transport cannot enforce an
output-token cap before that call finishes. All observed input, output and cached
tokens are retained. Subscription token usage is not subscription credits or a
dollar estimate.

Two independent tasks may run concurrently. This limits elapsed evaluation time
but does not remove service-load variation. There is one attempt per task/mode,
so results cannot establish a population improvement or tail latency. Small
synthetic tasks can favor direct completion.

## Artifacts and reproduction

The [public evidence export](pilot-evidence/public-manifest.json) contains all 35
captured live attempts, including exploratory checks and failures. It preserves
answers, selected quotations, source identities, grades, limits, per-call usage
and response hashes. Full prompts, source contexts, transport stdout and
trajectory payloads are omitted from publication. Original captures remain
local, unchanged, with SHA-256 receipts in the manifest. This is an explicit
schema projection, not a claim to publish complete raw transport records.

Original frozen runtime manifests are retained under
[pilot runtime manifests](runtime-manifests/pilot). These identify the source
files used by each separately labeled capture. The initial smoke manifest was
made after import and is explicitly labeled as such. Earlier checks and the
final protocol smoke are not added to the 24-attempt performance comparison.

Public raw downloads and extracted page caches are omitted from Git and can be
recreated from the pinned sources:

```bash
python -m pip install "pypdf==6.18.0"
python -m benchmarks.grounded_documents --prepare
python -m benchmarks.grounded_documents --list
python -m benchmarks.grounded_documents --model gpt-6.1-sol --reasoning-effort low \
  --task synthetic-code --task cloudflare-products --task cloudflare-gateway \
  --max-calls 16 --timeout 300 --call-timeout 180
```

The CLI refuses to overwrite an earlier task attempt. Authentication is checked
before calls; a transport or model failure is recorded and does not switch to
another provider. Exact code and source versions must be kept stable during a
comparison. Changing prompts, graders or the runtime requires a separately
labeled capture.
