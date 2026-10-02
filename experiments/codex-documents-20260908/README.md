# Fresh document comparison — 2026-09-08

The local HTTP companion fetched the pinned Python 3.14 CSV documentation directly from its upstream URL, prepared the text, and executed one direct call followed by an RLM run through ChatGPT-authenticated Codex CLI with GPT-5.6 Luna, medium reasoning. No API key was used.

| Mode | Correct reference fields | CLI model tokens | Calls | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Direct | 6 / 6 | 8,841 | 1 | 8.940 |
| RLM | 6 / 6 | 6,945 | 2 | 20.268 |

This is a single task, not a general performance claim. RLM used 21.4% fewer reported model tokens and took longer. These token counts are not measured subscription credits.

The question, exact reference values, complete answers, provenance and true events are in `python-csv-docs-comparison.json`; raw CLI messages and source evidence are in `python-csv-docs.raw.json`. `source-metadata.json` records download and extraction hashes. Reference values were withheld from the model. The browser validator accepted all 31 real HTTP progress snapshots during this smoke check.

Source: https://raw.githubusercontent.com/python/cpython/v3.14.0/Doc/library/csv.rst
Raw SHA-256: 9e0b648823bbffc3d6112de905d5fcbb7ebe2d29a4a9e8fd0e16737b6e0d3359
Python documentation license: https://docs.python.org/3/license.html
