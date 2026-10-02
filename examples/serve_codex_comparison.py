"""Local document companion for the RLM comparison page.

    pip install -e '.[visualizer]'
    python examples/serve_codex_comparison.py

The server binds only 127.0.0.1. Copy its pairing code from your own terminal
into the web page. Uploaded documents and raw recordings stay on this machine;
running a comparison sends extracted text through your logged-in Codex account.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import hmac
import io
import json
import multiprocessing
import secrets
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, cast
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "gpt-5.6-luna"
MAX_BYTES = 5 * 1024 * 1024
MAX_CHARS = 300_000
MAX_JSON_BYTES = ((MAX_BYTES + 2) // 3) * 4 + 10_000
MAX_EXPANDED_BYTES = 20 * 1024 * 1024
DEFAULT_ORIGINS = {
    "https://recursive-llm-lab.sakhli.chatgpt.site",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
}
PRESETS = {
    item["id"]: item
    for item in json.loads(
        Path(__file__).with_name("document_presets.json").read_text(encoding="utf-8")
    )
}


class RequestError(ValueError):
    """An actionable client error with an HTTP status."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def valid_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 180:
        raise RequestError("Provide a display filename of 1–180 characters")
    if any(char in value for char in ("/", "\\", "\x00")) or value in {".", ".."}:
        raise RequestError("Filename must be a display name, never a filesystem path")
    if Path(value).suffix.lower() not in {".txt", ".md", ".csv", ".json", ".rst", ".pdf", ".docx"}:
        raise RequestError("Supported files: TXT, MD, CSV, JSON, RST, PDF, DOCX")
    return value


def _extract_text(filename: str, data: bytes) -> Dict[str, Any]:
    """Parse bytes in a disposable bounded worker, without extracting archive paths."""
    valid_filename(filename)
    if not data or len(data) > MAX_BYTES:
        raise RequestError("Document must contain 1 byte to 5 MiB")
    suffix = Path(filename).suffix.lower()
    warnings = []
    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > 2000 or sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
                raise RequestError("DOCX expanded content exceeds the 20 MiB limit")
            for item in members:
                path = PurePosixPath(item.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                    raise RequestError("DOCX contains an invalid archive member path")
                if "vbaproject" in item.filename.lower() or item.flag_bits & 1:
                    raise RequestError("Macro-enabled or encrypted DOCX files are unsupported")
            if len([item for item in members if item.filename == "word/document.xml"]) != 1:
                raise RequestError("DOCX must contain one word/document.xml body")
            xml = archive.read("word/document.xml")
            if len(xml) > 5 * 1024 * 1024:
                raise RequestError("DOCX document XML exceeds 5 MiB")
            decoded = xml.decode("utf-8-sig")
            if "<!DOCTYPE" in decoded.upper() or "<!ENTITY" in decoded.upper():
                raise RequestError("DOCX document XML cannot contain a DTD or entities")
            root = ElementTree.fromstring(decoded)
            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for paragraph in root.iter(ns + "p"):
                parts = []
                for element in paragraph.iter():
                    if element.tag == ns + "t":
                        parts.append(element.text or "")
                    elif element.tag == ns + "tab":
                        parts.append("\t")
                    elif element.tag in {ns + "br", ns + "cr"}:
                        parts.append("\n")
                paragraphs.append("".join(parts))
            text = "\n".join(paragraphs)
        warnings.append(
            "DOCX body text and tables extracted; headers, footnotes, and images are omitted."
        )
    elif suffix == ".pdf":
        try:
            from pypdf import Configuration, PdfReader
        except ImportError as exc:
            raise RequestError("Install PDF support with: pip install -e '.[visualizer]'") from exc
        if not data.startswith(b"%PDF-"):
            raise RequestError("File does not have a PDF header")
        Configuration.zlib_maximum_output_length = 10 * 1024 * 1024
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise RequestError("Encrypted PDFs are unsupported")
        if len(reader.pages) > 500:
            raise RequestError("PDF exceeds the 500-page limit")
        pieces = []
        expanded = 0
        chars = 0
        for page in reader.pages:
            contents = page.get_contents()
            expanded += len(contents.get_data()) if contents else 0
            if expanded > MAX_EXPANDED_BYTES:
                raise RequestError("PDF expanded content exceeds 20 MiB")
            piece = page.extract_text() or ""
            chars += len(piece) + 1
            if chars > MAX_CHARS:
                raise RequestError("Extracted PDF text exceeds 300,000 characters")
            pieces.append(piece)
        text = "\n".join(pieces)
        warnings.append(
            "PDF text extraction has no OCR; scanned images and complex tables may be incomplete."
        )
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RequestError("Text documents must use UTF-8 encoding") from exc
        if "\x00" in text:
            raise RequestError("Binary content is not a UTF-8 text document")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise RequestError("No readable text found; scanned PDFs need OCR before uploading")
    if len(text) > MAX_CHARS:
        raise RequestError("Extracted text exceeds 300,000 characters")
    return {"text": text, "format": suffix[1:], "warnings": warnings}


class _PresetRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed: Any) -> None:
        super().__init__()
        self.allowed = set(allowed)

    def redirect_request(
        self, request: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: str
    ) -> Any:
        if newurl not in self.allowed or urllib.parse.urlsplit(newurl).scheme != "https":
            raise RequestError("Preset redirected to an unapproved URL")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _download_preset(preset: Dict[str, Any]) -> bytes:
    allowed = set(preset.get("allowed_redirect_urls", [])) | {preset["url"]}
    if any(urllib.parse.urlsplit(url).scheme != "https" for url in allowed):
        raise RequestError("Preset URLs must use HTTPS")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PresetRedirects(allowed))
    request = urllib.request.Request(
        preset["url"], headers={"User-Agent": "recursive-llm-document-lab/1.0"}
    )
    with opener.open(request, timeout=20) as response:
        if response.geturl() not in allowed:
            raise RequestError("Preset resolved to an unapproved URL")
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_BYTES:
            raise RequestError("Preset download exceeds 5 MiB")
        data = response.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise RequestError("Preset download must contain 1 byte to 5 MiB")
    if hashlib.sha256(data).hexdigest() != preset["sha256"]:
        raise RequestError(
            "Preset source changed (SHA256 mismatch); its reference answer must be revalidated", 409
        )
    return cast(bytes, data)


def _worker_entry(connection: Any, operation: str, payload: Any) -> None:
    try:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        except (ImportError, ValueError, OSError):
            pass
        result = _extract_text(*payload) if operation == "extract" else _download_preset(payload)
        connection.send({"ok": True, "result": result})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc), "status": getattr(exc, "status", 400)})
    finally:
        connection.close()


def bounded_worker(operation: str, payload: Any, timeout: float = 20) -> Any:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_worker_entry, args=(child, operation, payload), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise RequestError("Document operation exceeded its 20-second limit", 408)
        try:
            response = parent.recv()
        except EOFError as exc:
            raise RequestError("Document worker stopped while parsing the file") from exc
        if not response["ok"]:
            raise RequestError(response["error"], response["status"])
        return response["result"]
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


class CompanionState:
    """Prepared documents and one cancellable comparison on a dedicated event loop."""

    def __init__(self, output_dir: Optional[Path] = None, capture: Any = None) -> None:
        self.token = secrets.token_urlsafe(32)
        self.output_dir = (
            Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="rlm-lab-runs-"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.active_run: Optional[str] = None
        self.lock = threading.RLock()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.capture = capture
        self.tasks: Dict[str, Any] = {}

    def prepare(
        self, filename: str, data: bytes, preset: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        filename = valid_filename(filename)
        if len(data) > MAX_BYTES:
            raise RequestError("File exceeds 5 MiB", 413)
        raw_sha256 = hashlib.sha256(data).hexdigest()
        if preset and raw_sha256 != preset["sha256"]:
            raise RequestError(
                "Preset SHA256 changed; its reference answers cannot be applied", 409
            )
        extracted = bounded_worker("extract", (filename, data))
        text = extracted.pop("text")
        document_id = secrets.token_hex(12)
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "bytes": len(data),
            "context_chars": len(text),
            "sha256": raw_sha256,
            "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "preview": text[:2000],
            **extracted,
        }
        metadata["title"] = preset["title"] if preset else filename
        if preset:
            metadata.update(
                source_url=preset["url"],
                preset_id=preset["id"],
                suggested_question=preset["suggested_question"],
                expected_fields=preset["expected_fields"],
            )
            if preset.get("expected_aliases"):
                metadata["expected_aliases"] = deepcopy(preset["expected_aliases"])
        with self.lock:
            if len(self.documents) >= 16:
                self.documents.pop(next(iter(self.documents)))
            self.documents[document_id] = {"metadata": metadata, "text": text}
        return deepcopy(metadata)

    def start_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from examples.capture_codex_comparison import (
            validate_expected_aliases,
            validate_expected_fields,
        )

        if set(payload) - {"document_id", "question", "expected_fields"}:
            raise RequestError("Unexpected run options")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 10_000:
            raise RequestError("A question of 1–10,000 characters is required")
        identifier = payload.get("document_id")
        if not isinstance(identifier, str):
            raise RequestError("document_id is required")
        with self.lock:
            if self.active_run:
                raise RequestError("One comparison is already running; wait or cancel it", 409)
            document = self.documents.get(identifier)
            if document is None:
                raise RequestError("Prepared document not found; upload or download it again", 404)
            metadata = document["metadata"]
            expected = payload.get("expected_fields")
            aliases: Dict[str, List[str]] = {}
            if "expected_fields" not in payload and question == metadata.get("suggested_question"):
                expected = metadata.get("expected_fields")
                preset = PRESETS.get(metadata.get("preset_id"))
                if (
                    preset
                    and metadata.get("sha256") == preset["sha256"]
                    and question == preset["suggested_question"]
                    and expected == preset["expected_fields"]
                    and metadata.get("expected_aliases") == preset.get("expected_aliases")
                ):
                    aliases = metadata.get("expected_aliases") or {}
            try:
                expected = validate_expected_fields(expected)
                aliases = validate_expected_aliases(aliases, expected)
            except ValueError as exc:
                raise RequestError(str(exc)) from exc
            run_id = secrets.token_hex(12)
            self.runs[run_id] = {
                "run_id": run_id,
                "state": "running",
                "phase": "auth",
                "comparison": None,
                "document": deepcopy(metadata),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started": time.monotonic(),
                "phase_started": time.monotonic(),
                "cancel_requested": False,
            }
            self.active_run = run_id
            asyncio.run_coroutine_threadsafe(
                self._execute(run_id, document["text"], question, expected, aliases), self.loop
            )
            return self.snapshot(run_id)

    async def _execute(
        self,
        run_id: str,
        text: str,
        question: str,
        expected: Dict[str, Any],
        aliases: Dict[str, List[str]],
    ) -> None:
        from examples.capture_codex_comparison import capture_comparison

        def progress(value: Dict[str, Any]) -> None:
            with self.lock:
                run = self.runs[run_id]
                if run["phase"] != value["phase"]:
                    run["phase_started"] = time.monotonic()
                run.update(phase=value["phase"], comparison=value["comparison"])

        with self.lock:
            self.tasks[run_id] = asyncio.current_task()
            cancelled = self.runs[run_id]["cancel_requested"]
        try:
            if cancelled:
                raise asyncio.CancelledError()
            result = await (self.capture or capture_comparison)(
                context=text,
                query=question,
                expected_fields=expected,
                expected_aliases=aliases,
                output=self.output_dir / f"{run_id}.json",
                progress_handler=progress,
                document_metadata=self.runs[run_id]["document"],
            )
            with self.lock:
                failed = any(result[lane].get("error") for lane in ("direct", "rlm"))
                self.runs[run_id].update(
                    state="failed" if failed else "completed", phase="completed", comparison=result
                )
        except asyncio.CancelledError:
            with self.lock:
                self.runs[run_id].update(
                    state="cancelled", phase="cancelled", error="Cancelled by user"
                )
        except Exception as exc:
            with self.lock:
                self.runs[run_id].update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                self.runs[run_id]["elapsed_seconds"] = (
                    time.monotonic() - self.runs[run_id]["started"]
                )
                self.active_run = None
                self.tasks.pop(run_id, None)

    def snapshot(self, run_id: str) -> Dict[str, Any]:
        with self.lock:
            if run_id not in self.runs:
                raise RequestError("Run not found", 404)
            run = deepcopy(self.runs[run_id])
            if run["state"] in {"running", "cancelling"}:
                run["elapsed_seconds"] = time.monotonic() - run["started"]
                if run.get("comparison") and run["phase"] in {"direct", "rlm"}:
                    lane = run["comparison"][run["phase"]]
                    lane["seconds"] = max(lane["seconds"], time.monotonic() - run["phase_started"])
            for key in ("started", "phase_started", "cancel_requested"):
                run.pop(key, None)
            return run

    def cancel(self, run_id: str) -> Dict[str, Any]:
        with self.lock:
            if run_id not in self.runs:
                raise RequestError("Run not found", 404)
            run = self.runs[run_id]
            if run["state"] == "running":
                run.update(state="cancelling", cancel_requested=True)
                task = self.tasks.get(run_id)
                if task:
                    self.loop.call_soon_threadsafe(task.cancel)
            return self.snapshot(run_id)

    def close(self) -> None:
        if self.active_run:
            self.cancel(self.active_run)

        async def settle() -> None:
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(settle(), self.loop).result(timeout=5)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self.loop.close()


def _origin(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("Allowed origins must be exact http(s) origins without paths")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("Non-loopback origins must use HTTPS")
    return value


class CompanionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, state: CompanionState, origins: Any = None) -> None:
        self.state = state
        self.origins = {_origin(value) for value in (origins or DEFAULT_ORIGINS)}
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(("127.0.0.1", port), CompanionHandler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.slots.acquire(blocking=False):
            request.sendall(
                b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
            )
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            request.settimeout(30)
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class CompanionHandler(BaseHTTPRequestHandler):
    server: CompanionServer

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Never log pairing credentials, uploaded filenames, or request bodies.

    def _guard(self, preflight: bool = False) -> None:
        hosts = self.headers.get_all("Host", [])
        port = self.server.server_address[1]
        if len(hosts) != 1 or hosts[0] not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise RequestError("Invalid Host", 403)
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1 or (origins and origins[0] not in self.server.origins):
            raise RequestError("Origin not allowed", 403)
        if preflight:
            if not origins:
                raise RequestError("Preflight requires an allowed Origin", 403)
            if self.headers.get("Access-Control-Request-Method") not in {"GET", "POST"}:
                raise RequestError("Preflight method not allowed", 403)
            requested = {
                name.strip().lower()
                for name in self.headers.get("Access-Control-Request-Headers", "").split(",")
                if name.strip()
            }
            if requested - {"authorization", "content-type"}:
                raise RequestError("Preflight headers not allowed", 403)
            return
        authorizations = self.headers.get_all("Authorization", [])
        expected = "Bearer " + self.server.state.token
        if len(authorizations) != 1 or not hmac.compare_digest(authorizations[0], expected):
            raise RequestError("Pairing code required", 401)

    def _send(self, status: int, body: Optional[Dict[str, Any]] = None) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else b""
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> Dict[str, Any]:
        if (
            self.headers.get("Transfer-Encoding")
            or len(self.headers.get_all("Content-Length", [])) != 1
        ):
            raise RequestError("A single Content-Length is required")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise RequestError("Content-Type must be application/json", 415)
        try:
            length = int(self.headers["Content-Length"])
        except ValueError as exc:
            raise RequestError("Invalid Content-Length") from exc
        if length < 2 or length > MAX_JSON_BYTES:
            raise RequestError("Request exceeds the upload limit", 413)
        data = self.rfile.read(length)
        if len(data) != length:
            raise RequestError("Incomplete request body")
        try:
            payload = json.loads(data)
        except (UnicodeError, ValueError) as exc:
            raise RequestError("Request body must be UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RequestError("Request body must be a JSON object")
        return payload

    def _dispatch(self) -> None:
        self._guard(self.command == "OPTIONS")
        if self.command == "OPTIONS":
            self._send(204)
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.query or parsed.scheme or parsed.netloc:
            raise RequestError("Only relative endpoint paths are accepted")
        path = parsed.path
        state = self.server.state
        if self.command == "GET" and path == "/health":
            self._send(
                200,
                {
                    "service": "rlm-compare",
                    "model": MODEL,
                    "authenticated": True,
                    "limits": {"file_bytes": MAX_BYTES, "text_chars": MAX_CHARS, "active_runs": 1},
                },
            )
        elif self.command == "GET" and path == "/presets":
            self._send(
                200,
                {
                    "presets": [
                        {**preset, "source_url": preset["url"]} for preset in PRESETS.values()
                    ]
                },
            )
        elif self.command == "POST" and path == "/documents":
            payload = self._body()
            if set(payload) != {"filename", "content_base64"} or not isinstance(
                payload.get("content_base64"), str
            ):
                raise RequestError(
                    "Provide filename and content_base64 only; filesystem paths are unsupported"
                )
            filename = valid_filename(payload["filename"])
            try:
                data = base64.b64decode(payload["content_base64"], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise RequestError("Invalid base64 file data") from exc
            self._send(201, state.prepare(filename, data))
        elif self.command == "POST" and path.startswith("/presets/") and path.endswith("/download"):
            if self._body():
                raise RequestError("Preset download takes an empty JSON object")
            preset_id = path[len("/presets/") : -len("/download")]
            preset = PRESETS.get(preset_id)
            if not preset:
                raise RequestError("Preset not found", 404)
            data = bounded_worker("download", preset)
            self._send(201, state.prepare(preset["filename"], data, preset))
        elif self.command == "POST" and path == "/runs":
            self._send(202, state.start_run(self._body()))
        elif self.command == "GET" and path.startswith("/runs/") and "/" not in path[6:]:
            self._send(200, state.snapshot(path[6:]))
        elif self.command == "POST" and path.startswith("/runs/") and path.endswith("/cancel"):
            if self._body():
                raise RequestError("Cancel takes an empty JSON object")
            self._send(202, state.cancel(path[6:-7]))
        else:
            raise RequestError("Endpoint not found", 404)

    def _handle(self) -> None:
        try:
            self._dispatch()
        except RequestError as exc:
            self._send(exc.status, {"error": str(exc)})
        except (ValueError, OSError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception:
            self._send(500, {"error": "Companion request failed"})

    do_GET = _handle
    do_POST = _handle
    do_OPTIONS = _handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--allow-origin", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    state = CompanionState(args.output_dir)
    server = CompanionServer(
        args.port, state, DEFAULT_ORIGINS | {_origin(value) for value in args.allow_origin}
    )
    print(f"RLM companion: http://127.0.0.1:{server.server_address[1]}")
    print(f"Pairing code (keep local): {state.token}", flush=True)
    print(f"Private run recordings: {state.output_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
