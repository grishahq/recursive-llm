# RLM / Compare

A side-by-side replay of real RLM and direct-model experiments, created for the project's 600-star milestone. Both sides use GPT-5.6 Luna through the ChatGPT-authenticated Codex CLI. The page supports real local runs, source document uploads, fresh downloads from three preset URLs, and playback of captured results. Codex authentication stays in the CLI; a short-lived pairing code connects the page to the loopback companion.

From the repository root:

```bash
npm --prefix visualizer install
npm --prefix visualizer run dev
```

Open the local address printed by the development server. Play, pause, scrub, change speed, inspect recorded steps and the call tree, switch experiments, or download the comparison.

## Run from the web

Start the local companion from the repository root:

```bash
python -m pip install -e '.[visualizer]'
codex login
python examples/serve_codex_comparison.py
```

Click **Connect Codex**, paste the address and pairing code printed in your terminal, choose a document and question, then click **Download & run**. The hosted page is an interface to your own computer; it does not expose the project owner's subscription. The browser may request local network access. For a local development page, add its exact origin with `--allow-origin http://localhost:3000` (use the actual printed port).

TXT, Markdown, CSV, JSON, PDF and DOCX are supported, up to 5 MB and 300k extracted characters. Scanned PDFs require OCR before upload. Presets download from their original HTTPS URL for every run and verify the source SHA-256. Only URLs and reference metadata are bundled. Updated upstream bytes fail explicitly rather than using a stale answer key.

Preset questions use ordinary language. The editable question stays separate from the answer-format instructions used for automatic grading; the complete prompt can be expanded in the result. Questions are visible before execution and shared by both modes. Reference fields are graded after generation; expected values never enter model prompts. Changing a preset question disables its original key. The Alice judge field accepts explicit, source-supported variants such as "The King" and "King of Hearts"; any accepted variants are shown with the reference result. Custom questions without a reference require manual review. Correctness is separate from token and time metrics.

The page also offers a source ZIP of the companion. Regenerate it after changing the Python runner or library:

```bash
python3 visualizer/scripts/package-companion.py
```

This explicitly bundles runner/library source and a hash manifest, without credentials, document contents, or private run logs.

## Record another experiment

Install the Python project and sign in to Codex with ChatGPT. API-key authentication is deliberately rejected.

```bash
pip install -e .
codex login
python examples/capture_codex_comparison.py --chars 100000 --output comparison.json
```

The recorder defaults to Luna with medium reasoning and RLM depth 1. To compare
another model or root-only execution, pass the settings explicitly:

```bash
python examples/capture_codex_comparison.py --model gpt-6.1-sol \
  --reasoning-effort low --max-depth 0 --chars 10000 --output sol-root-only.json
```

Both modes use the selected model and reasoning effort. Depth 0 permits only the
root REPL, depth 1 adds plain model leaves, and depth 2 allows child RLMs. The
recordings retain these settings. The web companion keeps its Luna/medium/depth 1
defaults; import CLI recordings to inspect runs with other settings.

The CLI consumes your Codex subscription allowance. Open the resulting JSON using **Run with Codex → Open recording**. Imported data stays in your browser.

The generated ledger has a deterministic answer key. Total model tokens include Codex input, output, cached input, and instruction overhead. Wall-clock measurements include CLI startup. Dollar costs are intentionally absent. The replay aligns both runs by elapsed time even though the capture runs them sequentially; it does not imply simultaneous execution. A single pair is an experiment, not a broad benchmark or a measure of subscription credits.

## Build and check

```bash
npm --prefix visualizer run build
npm --prefix visualizer run typecheck
npm --prefix visualizer test
npm --prefix visualizer run lint
```

Lint covers the app, feature components, data adapters, and tests. The supplied
UI component catalog is outside this focused lint command.

The included film is a synchronized replay accelerated equally for both lanes. Its opening and closing titles do not change the recorded events or metrics. Captions are available in `public/rlm-comparison.vtt`.

The Sites deployment configuration is in `.openai/hosting.json`. The site is private by default; making it public is a separate access change.

Dependency maintenance: `satori@0.33.5` pins an affected `fflate@0.7.3`; the scoped
override selects the compatible `0.7.5` security patch for
[GHSA-px8p-9vwx-vf98](https://github.com/advisories/GHSA-px8p-9vwx-vf98).
Remove the override when Satori's own dependency is patched. Run `npm audit`
alongside the build checks after updating dependency pins.

## Re-render the film

Install Playwright and Chromium in your preferred tooling environment and put FFmpeg on `PATH`. With the development server running, run:

```bash
node scripts/render-video.mjs --url 'http://localhost:3000/?film=1' \
  --output public/rlm-comparison.mp4 --width 1920 --height 1080 \
  --duration 34 --fps 30 --seek-function __rlmSetVideoTime
```

Use the actual port printed by the server. If Playwright is installed outside this directory, set `RLM_PLAYWRIGHT_PATH` to that installation's `playwright` module path. The script renders exact frame timestamps and outputs silent H.264 video.
