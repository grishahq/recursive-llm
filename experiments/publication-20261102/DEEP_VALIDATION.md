# Document validation: deeper engineering checks

The final v0.6.0 runtime completed 48 bounded live attempts over two pinned public
sources. Every attempt returned the expected field values and structurally valid
original-source citations. The frozen literal evidence grader passed 46/48;
the two remaining cases are retained and reviewed below.

The before/after controls demonstrate automatic rejection of forged references,
bounded recovery from an empty CLI response, acceptance of valid native JSON,
and loading of the complete 157-page report under the explicit long profile.
A real quote can still accompany a wrong claim. This release checks citation
identity and text, not arbitrary claim entailment.

## Live protocol and results

Twelve questions cover the complete Cloudflare 2024 Form 10-K and official Python
3.12.0 sqlite3 documentation. They include counts, financial table values,
arithmetic, conditions, API behavior, and a profile spanning three PDF pages.
Gold values and allowed source locations were independently checked before
inference. Table columns and units were also visually checked against rendered
PDF pages. No gold values, aliases or scores were changed after model output.

Dataset `deep-documents-v2`; grader `deep-document-fields-v1`; frozen gold SHA-256:
`29c1cfff1f5b1c260b5e1ab3312adb7d753807576ae0e79728003de5118acec3`.

Each question ran twice with `gpt-6.1-sol`, low reasoning, in depth-zero
Python/search mode and depth-two RLM mode. Limits were ten model calls,
150,000 reported tokens and 240 seconds per attempt, with at most two attempts
running concurrently. The four pilot attempts form part of the 48, rather than
an additional selected sample. All raw outcomes were retained locally.

| Mode | Completed | Exact field values | Valid citation structure | Strict evidence coverage | Model calls | Model tokens | Summed attempt time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python/search, depth zero | 24/24 | 24/24 | 24/24 | 22/24 | 84 | 591,852 | 665.38 s |
| RLM, depth two allowed | 24/24 | 24/24 | 24/24 | 24/24 | 80 | 558,921 | 698.23 s |

Totals: 164 model calls, 1,150,773 tokens, zero unknown-usage calls and zero child
model calls. Input tokens include 329,472 cached tokens; they are not subtracted.
The summed times include overlapping attempts. Median attempt times were 27.13 s
and 26.23 s respectively. These measurements do not estimate subscription credits,
dollar savings or tail latency. [Machine-readable totals](deep-summary.json).

No child calls were used. The results therefore assess the document workflow
and two prompt configurations, without establishing a benefit from recursive
decomposition or an LLM-quality gain over v0.5.0.

| Question | Depth zero, two attempts | Depth two, two attempts |
| --- | ---: | ---: |
| Workforce and subtraction | 2/2 | 2/2 |
| Patent counts and expiry years | 2/2 | 2/2 |
| Paying and large customers | 1/2 | 2/2 |
| Free cash flow and its inputs | 1/2 | 2/2 |
| Customer concentration thresholds | 2/2 | 2/2 |
| Retention and contract terms | 2/2 | 2/2 |
| Regional revenue | 2/2 | 2/2 |
| Distributed company profile | 2/2 | 2/2 |
| sqlite3 thread contract | 2/2 | 2/2 |
| sqlite3 row factory | 2/2 | 2/2 |
| sqlite3 parameter containers | 2/2 | 2/2 |
| sqlite3 lastrowid | 2/2 | 2/2 |

The table uses the unchanged strict evidence score. Two manual reviews explain
its failures without turning them into automatic passes:

- [Free cash flow, first depth-zero attempt](deep-evidence/live/remaining-first-repeat/cloud-free-cash-flow-python-1.json): the correct result, 166,915, follows from the quoted inputs 380,429 minus 185,037 minus 28,477. The excerpt contains the year and units but cuts the separately reported result at `166,9`. The frozen grader required the complete reported-result anchor.
- [Customer counts, second depth-zero attempt](deep-evidence/live/all-second-repeat/cloud-customer-counts-python-2.json): the quote contains the correct counts and corresponding years. Its left edge starts at ` number of paying customers`, omitting `The` from the frozen literal anchor.

These are literal-coverage failures with source-supported answers, not observed
wrong financial values. The two repetitions share source facts, and several
questions overlap. They are not independent samples of general production quality.

## Differential controls against the previous release

Fixed-response tests use byte-identical files from commit
`3c255c070060837343e594f60523e02b5369b87d`. They make zero live model calls and
test runtime behavior, rather than model-generated accuracy.

Seven variants change quote bytes, source hash, page, offset, page type, document
ID, or the hash of a stale document with the same ID. The previous release's
default string API accepts the first invalid payload in all seven cases. The
document API rejects each invalid payload and accepts the supplied repair.
The previous release **with the same explicit custom validator also rejects and
repairs all seven**. This control passes the new source-validator callback into
the existing old hook. The improvement is
integrated document handling and defaults, not a new validation algorithm.

Other controlled outcomes:

- One valid native-JSON answer exhausts the previous one-iteration protocol;
  the document API accepts it through citation validation.
- An explicitly empty CLI message blocks the previous transport. The final
  transport preserves its usage and the existing repair loop completes on the
  second fixed response.
- The original Cloudflare PDF is rejected by the previous companion's encrypted
  PDF rule. The current standard profile still rejects its length. The explicit
  long profile and document loader preserve all 157 physical pages and 591,696
  extracted characters. This PDF opens with a blank password; this is not a claim
  of support for arbitrary password-protected or scanned documents.
- Four normalization cases preserve original text and offsets across accents,
  decomposed marks, case-fold expansion and whitespace. The literal-search
  recipe misses them; users of the previous library could supply custom
  normalization themselves.

[Compact fixed-response evidence](controls-summary.json) includes statuses,
rejection counts, source identities and hashes of the original local captures.

## Adversarial checks on the actual public documents

Each of the twelve real questions also has zero-model controls. A changed quote
is accepted by the previous default API in 12/12 cases; the document API rejects
it and accepts the fixed valid repair in 12/12 cases.

The negative controls expose the boundary: twelve unrelated but real quotations
and twelve wrong claims paired with real quotations pass structural validation.
The independent gold grader rejects them. The small synthetic wrong-claim
control is likewise accepted by the previous API, the previous API with a custom
source validator, and the document API.

Citation checking makes fabricated references observable. Establishing that a
quotation proves a claim, that every relevant rule was found, or that a global
absence claim is justified remains a separate evaluation problem.
