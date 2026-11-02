# Document workflow validation, 2 November 2026

This release adds an integrated source contract for document answers. It checks
source identity, physical pages, exact character spans and quotations. It does
not prove that a quotation supports a claim, or that an answer is complete.

- [Deep validation](DEEP_VALIDATION.md): 48 repeated live attempts on a complete
  157-page report and official Python documentation, plus fixed-response controls
  against v0.5.0. All 48 passed structural citation checks; 46 passed the frozen
  field-and-evidence criteria. The two strict misses remain failures with manual
  review notes. No live attempt used child model calls.
- [Initial pilot](PILOT_RESULTS.md): direct, Python/search and depth-two modes on
  synthetic rules and real documents, including generated calculation code,
  Terra checks and failed Luna attempts. Its earlier runtime stages are labeled
  separately from the final release validation.
- [Fixed-response differential controls](offline-controls.json): 30 source
  fixtures, empty-CLI recovery, Unicode search and PDF profile checks. Corrected
  replies are supplied fixtures, not measured autonomous model repair.
- [Deterministic and package verification](verification.json): local test,
  quality, wheel, source-distribution and isolated-install results.

The old string API can also reject the seven corrupted-reference fixtures when
given the same custom source validator. The new document API integrates that
contract and source access. The live matrix compares two configurations of the
new workflow, not old-versus-new model quality.

## Evidence exports

[Pilot manifest](pilot-evidence/public-manifest.json) and
[deep-run manifest](deep-evidence/public-manifest.json) list every exported live
attempt, including failures. Exports retain raw answer text, selected quotations,
grades, source hashes, settings, per-call usage, response hashes and event counts.
They omit full prompts, full source contexts, transport stdout and trajectory
payloads. Original local captures are left unchanged and identified by SHA-256.
These compact records are not complete raw transport recordings.

The [exporter](export_evidence.py) uses an explicit field whitelist. This is a
schema projection for these public experiments, not a general-purpose secret
scrubber. Unknown model usage remains null rather than being recorded as zero.
Source documents can be fetched again from the URLs and hashes in the
[source manifest](pilot-source-manifest.json).

The [frozen deep gold](deep-gold.json) and its
[dataset manifest](deep-dataset-manifest.json) were fixed before live outputs.
The separately retained v1 draft records the pre-run independent audit changes.
Gold answers and numeric function test cases were not shown to the models.

## Reproduction

From a checkout with the optional document dependencies installed:

```bash
python -m benchmarks.grounded_documents --prepare
python -m benchmarks.deep_document_validation --help
python -m pytest
```

Live runs use the existing ChatGPT-authenticated Codex CLI and explicit model,
call and time limits. They consume the account's subscription allowance. No API
fallback is used. See each report for the exact captured configuration and its
limitations. Running a new experiment does not recreate an earlier model output.
