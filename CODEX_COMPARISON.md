# Codex subscription comparison

Recorded on September 8, 2026 to accompany the 600-star visualizer. Both modes use `gpt-5.6-luna` with medium reasoning through Codex CLI 0.149.0, authenticated with ChatGPT. No API key was used.

Each row is **one paired experiment**, with direct mode first. These observations do not establish a general quality, latency, or subscription-efficiency advantage.

| Generated source | Mode | Exact fields | Total CLI model tokens | Cached input | Model calls | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10,104 characters | Direct | 4/4 | 6,007 | 0 | 1 | 9.172 s |
| 10,104 characters | RLM | 4/4 | 19,430 | 0 | 5 | 48.488 s |
| 100,098 characters | Direct | 4/4 | 37,521 | 0 | 1 | 29.040 s |
| 100,098 characters | RLM | 4/4 | 11,656 | 0 | 3 | 30.707 s |

On the larger task, RLM used **68.9% fewer total CLI model tokens**, while taking approximately 1.7 seconds longer. The smaller task favored direct completion in both token use and latency. All four answers matched the deterministic answer key. Model tokens are not a measurement of how many subscription credits or percentage points were consumed.

## What is held constant

Both sides receive the same query and deterministic source (seed 2026), use the same exact model and reasoning effort, and share the same Codex text-completion wrapper. Each model call starts a fresh ephemeral Codex session. User configuration is ignored; existing ChatGPT authentication is retained. Neither side receives answer-key feedback.

Direct mode receives the full context in one model request. RLM runs the library's existing control loop and restricted Python REPL. The model receives RLM instructions and observations; the full source remains in the REPL. Therefore the mode-specific prompts differ by design. RLM call history and CLI instruction overhead count toward reported usage. Both displayed runs used the root REPL, without child RLM calls.

Codex host capabilities are disabled where supported. Luna requires its Code Mode host to initialize; its presence is not counted as a tool call. Any actual tool-use item in the CLI event stream invalidates the run. Both completed pairs contain only `agent_message` items, with no Codex tool calls. Python execution occurs in the library's REPL.

## Metric definitions and replay

- **Total CLI model tokens:** reported input plus output over every completed model call. Cached input is included in total input, not subtracted. Incomplete usage is reported as unknown rather than as a complete total.
- **Elapsed:** wall-clock time for each mode, including CLI startup, network time, and RLM execution. Runs were sequential; the page aligns both timelines by elapsed time for comparison.
- **Exact fields:** count, sum of amount in cents, largest transaction ID, and its amount. No grader feedback is supplied during generation.
- **Subscription billing:** no per-run dollar charge or inferred subscription-percentage saving is reported.
- **Video:** real event timestamps are replayed with the same acceleration factor on both sides. Introductory diagrams explain the execution path; recorded Python and final answers come from the captured runs.

The 10k run encountered REPL helper errors and recovered within the original library loop. Those events remain in the recording; the REPL was not changed to improve the experiment. An initial CLI startup failure on both sides is preserved separately as `startup-diagnostic*`; it is infrastructure evidence, not a failed model answer or a performance trial. The finalized runner marks execution failures as ungraded.

## Artifacts and reproduction

[Captured experiments](experiments/codex-luna-20260908/) contain the comparison JSON, raw provider events and messages, and a transport manifest with the CLI version and shared wrapper. The corpus is synthetic; no private document was used. The hashes are:

- 10k: `753ce91f5a0d19351081775d4529a379358b0ff132bfb350c8f2050627f15236`
- 100k: `1ee43f3b42f8db55369c337d2e37f1e7f61224abe3d538581a716865df8a6fcc`

```bash
pip install -e .
codex login
python examples/capture_codex_comparison.py --chars 10000 --output comparison-10k.json
python examples/capture_codex_comparison.py --chars 100000 --output comparison-100k.json
```

The historical settings remain the defaults. For a new comparison with Sol 6.1,
select the shared model and reasoning effort explicitly:

```bash
python examples/capture_codex_comparison.py --model gpt-6.1-sol \
  --reasoning-effort low --max-depth 0 --chars 10000 --output sol-root-only.json
```

`--max-depth 0` runs the root REPL without subcalls; depth 1 permits plain model
leaves and depth 2 permits child RLMs. Both modes receive the selected model and
reasoning effort, and raw and replay recordings retain those settings. The web
companion continues to use Luna, medium reasoning, and depth 1.

These commands consume the signed-in account's Codex subscription allowance. [The visualizer](visualizer/README.md) can open the resulting comparison JSON locally in the browser. The companion `.raw.json` retains messages and raw events and should be reviewed before sharing a run on private source material.

For a custom source, supply `--context-file` and `--query` together. Custom answers require manual review unless you provide a reference map with `--expected-fields answers.json`:

```bash
python examples/capture_codex_comparison.py \
  --context-file document.txt --query "What are the main findings?" \
  --output my-comparison.json
```

## Run your own document from the web page

Install the optional document parser and start the companion in your own terminal:

```bash
pip install -e '.[visualizer]'
codex login
python examples/serve_codex_comparison.py
```

The companion listens at `http://127.0.0.1:8766` and prints a temporary pairing code. Paste that code into the page's connection form. Keep the terminal open. The code changes whenever the companion restarts; it is never saved in a recording or logged with HTTP requests. `--port` changes the local port. `--allow-origin https://your-page.example` permits an additional exact page origin.

Choose a preset or upload TXT, Markdown, CSV, JSON, RST, PDF, or DOCX, then enter the question and run the comparison. Each explicit run uses the signed-in account's Codex allowance. Selecting a preset alone does not invoke a model. The companion downloads the selected preset over HTTPS on demand and verifies its SHA256 before attaching the corresponding reference answer.

Files are limited to 5 MiB and extracted text to 300,000 characters. DOCX parsing reads body text and tables without extracting archive files; expanded content is limited to 20 MiB. PDF parsing extracts existing text with no OCR, so scans need OCR first and complicated tables may lose structure. The extracted preview is shown before the model results. Uploaded source text stays in the local companion and is sent through Codex when you run the comparison. The hosted page receives the preview, progress, model outputs, and evaluation fields.

Preset questions have explicit reference fields. Changing a preset question disables its reference key unless you supply a new one. Custom questions without a key show **manual review**, with no invented quality score. An optional reference map is a JSON object such as `{"project_name":"Orion","total":12,"approved":true}`. The model receives only the required field names and JSON output format; expected values are withheld. Numbers are compared exactly, booleans retain their JSON type, and ordinary text ignores case, whitespace, and quote/dash typography. Whitespace-only strings are compared exactly. Missing, duplicate, or incorrectly typed fields fail their checks; transport failures remain ungraded.

The page polls actual progress once per second. Only one comparison can run at a time, and Cancel interrupts the active run. Raw traces are saved in the private temporary directory printed by the companion, or a directory selected with `--output-dir`. Review private source material before sharing any recording. The companion binds only the loopback interface, requires bearer pairing on every data endpoint, and accepts browser requests only from its explicit origin allowlist.
