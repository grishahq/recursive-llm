"""Bundle the local runner's source, without credentials, documents or recordings."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "visualizer/public/rlm-local-companion.zip"
files = [ROOT / name for name in (
    "pyproject.toml", "README.md", "LICENSE", "CODEX_COMPARISON.md",
    "examples/capture_codex_comparison.py", "examples/serve_codex_comparison.py",
    "examples/document_presets.json", "benchmarks/generated_long_context.py",
)] + sorted((ROOT / "src/rlm").rglob("*.py"))
instructions = """# RLM local companion

Requires Python 3.10+ and Codex CLI installed and signed in with ChatGPT.
Unzip this folder, open a terminal inside it, then run:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e '.[visualizer]'
    codex login
    python examples/serve_codex_comparison.py

On Windows, activate with .venv\\Scripts\\activate instead.
Open https://recursive-llm-lab.sakhli.chatgpt.site, click Connect Codex,
and paste the pairing code printed in your terminal. Keep the terminal open.
Select a document, review the question, then click Download & run.

No experiment runs until you request it. Model calls use your Codex subscription.
Files are processed by this local companion and sent through Codex to the model.
Raw prompts and results are retained in the output folder shown by the companion.
The included presets contain URLs and reference metadata, not document contents.
Stop the companion with Ctrl+C. Never share its pairing code.

Source: https://github.com/grishahq/recursive-llm (MIT)
SOURCE_MANIFEST.json contains a SHA-256 for each bundled source file.
"""
manifest = {}
with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        manifest[relative] = hashlib.sha256(data).hexdigest()
        archive.writestr(f"rlm-local-companion/{relative}", data)
    archive.writestr("rlm-local-companion/START_HERE.md", instructions)
    archive.writestr("rlm-local-companion/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
with ZipFile(OUTPUT) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(files) + 2
print(f"Bundled {len(files)} source files: {OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes)")
