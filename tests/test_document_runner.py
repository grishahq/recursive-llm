"""Document parsing, authenticated local endpoints, progress, grading, and cancellation."""

import asyncio
import base64
import hashlib
import http.client
import io
import json
import threading
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from examples.capture_codex_comparison import (
    capture_comparison,
    grade_fields,
    validate_expected_aliases,
    validate_expected_fields,
)
from examples.serve_codex_comparison import (
    CompanionServer,
    CompanionState,
    MAX_EXPANDED_BYTES,
    PRESETS,
    RequestError,
    _download_preset,
    _extract_text,
    _origin,
    _PresetRedirects,
    bounded_worker,
    valid_filename,
)
from rlm.codex import parse_events
from tests.test_codex import events


def docx_bytes(xml=None, extra=None):
    stream = io.BytesIO()
    xml = xml or (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Hello document</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.mark.parametrize(
    "filename", ["../secret.txt", "/tmp/a.txt", "C:\\x.txt", "a.exe", "", None]
)
def test_upload_never_accepts_filesystem_paths(filename):
    with pytest.raises(RequestError):
        valid_filename(filename)


def test_text_and_docx_extract_content_with_limits():
    assert _extract_text("a.csv", b"name,value\r\na,1\r\n")["text"] == "name,value\na,1\n"
    assert _extract_text("a.docx", docx_bytes())["text"] == "Hello document"
    for name, content in [
        ("a.txt", b"\xff"),
        ("a.txt", b"hello\0world"),
        ("a.txt", b" "),
        ("a.txt", b"a" * 300001),
        ("a.docx", docx_bytes(extra={"../escape": "x"})),
        ("a.docx", docx_bytes(extra={"word/vbaProject.bin": "x"})),
        ("a.docx", docx_bytes(xml='<!DOCTYPE x [<!ENTITY x "bad">]><x>&x;</x>')),
    ]:
        with pytest.raises((RequestError, ValueError)):
            _extract_text(name, content)


def test_docx_expanded_size_is_bounded_before_parsing():
    data = docx_bytes(extra={"word/large.xml": "a" * (MAX_EXPANDED_BYTES + 1)})
    assert len(data) < 100_000
    with pytest.raises(RequestError, match="expanded"):
        _extract_text("bomb.docx", data)


def test_pdf_extracts_text_and_reports_no_ocr():
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 50 700 Td (Hello PDF) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    stream = io.BytesIO()
    writer.write(stream)
    result = _extract_text("a.pdf", stream.getvalue())
    assert "Hello PDF" in result["text"]
    assert "no OCR" in result["warnings"][0]
    blank = pypdf.PdfWriter()
    blank.add_blank_page(width=100, height=100)
    stream = io.BytesIO()
    blank.write(stream)
    with pytest.raises(RequestError, match="OCR"):
        _extract_text("blank.pdf", stream.getvalue())


def test_real_bounded_worker_extracts_without_writing_uploaded_files():
    assert bounded_worker("extract", ("example.txt", b"hello"))["text"] == "hello"
    with pytest.raises(RequestError):
        bounded_worker("extract", ("example.txt", b"\xff"))


def slow_worker(connection, operation, payload):
    time.sleep(5)


def test_worker_deadline_terminates_child(monkeypatch):
    monkeypatch.setattr("examples.serve_codex_comparison._worker_entry", slow_worker)
    started = time.monotonic()
    with pytest.raises(RequestError, match="20-second"):
        bounded_worker("extract", ("a.txt", b"x"), timeout=0.05)
    assert time.monotonic() - started < 3


def test_download_is_exact_allowlisted_https_and_hash_pinned(monkeypatch):
    data = b"reference document"
    preset = {
        "url": "https://example.org/reference.txt",
        "sha256": hashlib.sha256(data).hexdigest(),
    }

    class Response(io.BytesIO):
        headers = {}

        def geturl(self):
            return preset["url"]

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == preset["url"] and timeout == 20
            return Response(data)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    assert _download_preset(preset) == data
    with pytest.raises(RequestError, match="SHA256"):
        _download_preset({**preset, "sha256": "0" * 64})
    with pytest.raises(RequestError, match="HTTPS"):
        _download_preset({**preset, "url": "http://127.0.0.1/private"})
    with pytest.raises(RequestError, match="unapproved"):
        _PresetRedirects([preset["url"]]).redirect_request(
            None, None, 302, "redirect", {}, "https://127.0.0.1/private"
        )


def test_reference_grader_checks_fields_without_substring_credit():
    assert grade_fields("answer=incorrect", {"answer": "correct"})["matched"] == 0
    assert grade_fields("answer=correct answer=wrong", {"answer": "correct"})["matched"] == 0
    assert (
        grade_fields('{"answer":"correct","answer":"wrong"}', {"answer": "correct"})["matched"] == 0
    )
    quality = grade_fields(
        '{"depth":500.3630,"strict":false,"title":"Alice\u2019s Evidence"}',
        {"depth": 500.363, "strict": False, "title": "Alice's Evidence"},
    )
    assert quality["matched"] == quality["total"] == 3
    assert grade_fields('{"newline":"\\r\\n"}', {"newline": "\r\n"})["matched"] == 1
    assert grade_fields('{"newline":""}', {"newline": "\r\n"})["matched"] == 0
    assert grade_fields('{"count":"12"}', {"count": 12})["matched"] == 0
    assert grade_fields('{"count":12}', {"count": "12"})["matched"] == 0
    composite = grade_fields('{"answer":["correct"]}', {"answer": "correct"})
    assert composite["matched"] == 0 and isinstance(composite["fields"][0]["observed"], str)
    invalid_number = grade_fields('{"count":NaN}', {"count": 12})
    assert invalid_number["matched"] == 0 and isinstance(
        invalid_number["fields"][0]["observed"], str
    )
    assert grade_fields("answer", {})["status"] == "manual_review"
    assert grade_fields("", {"answer": "x"}, failed=True)["status"] == "unavailable"
    for invalid in [{"bad label": "x"}, {"x": []}, {"x": float("nan")}, []]:
        with pytest.raises(ValueError):
            validate_expected_fields(invalid)


def test_reference_aliases_are_explicit_exact_strings_without_changing_observed_answers():
    expected = {"trial_judge": "King", "other_role": "King"}
    aliases = {"trial_judge": ["The King", "King of Hearts", "The King of Hearts"]}
    for answer in ["King", "The King", "King of Hearts", "  THE   KING OF HEARTS  "]:
        quality = grade_fields(
            json.dumps({"trial_judge": answer, "other_role": "The King"}),
            expected,
            expected_aliases=aliases,
        )
        assert quality["matched"] == 1
        field = quality["fields"][0]
        assert field["observed"] == answer and field["expected"] == "King"
        assert field["accepted_aliases"] == aliases["trial_judge"]
        assert "accepted_aliases" not in quality["fields"][1]
    for answer in ["Queen", "not the King", "King or Queen", "The King was the judge.", ["King"]]:
        assert (
            grade_fields(json.dumps({"trial_judge": answer}), expected, expected_aliases=aliases)[
                "matched"
            ]
            == 0
        )
    legacy = grade_fields('{"trial_judge":"The King"}', {"trial_judge": "King"})
    assert legacy["matched"] == 0 and "accepted_aliases" not in legacy["fields"][0]
    for invalid in [
        [],
        {"unknown": ["King"]},
        {"count": ["12"]},
        {"trial_judge": "The King"},
        {"trial_judge": []},
        {"trial_judge": [False]},
        {"trial_judge": [" "]},
        {"trial_judge": ["x" * 1001]},
        {"trial_judge": ["King"] * 11},
    ]:
        with pytest.raises(ValueError):
            validate_expected_aliases(invalid, {**expected, "count": 12})


@pytest.mark.asyncio
async def test_capture_progress_and_references_never_enter_prompts(tmp_path):
    direct = AsyncMock(return_value=parse_events(events('{"answer":"AliasSecret"}')))
    direct.check_login = AsyncMock(return_value="ChatGPT")
    direct.calls = []
    recursive = AsyncMock(
        side_effect=[
            parse_events(events("print(context)")),
            parse_events(events('FINAL(\'{"answer":"AliasSecret"}\')')),
        ]
    )
    recursive.calls = []
    progress = []
    with patch(
        "examples.capture_codex_comparison.CodexCompletion", side_effect=[direct, recursive]
    ):
        result = await capture_comparison(
            context="Document text",
            query="Find the answer.",
            expected_fields={"answer": "ReferenceSecret"},
            expected_aliases={"answer": ["AliasSecret"]},
            output=tmp_path / "graded.json",
            progress_handler=progress.append,
            document_metadata={
                "title": "Uploaded source",
                "filename": "source.txt",
                "source_url": "https://example.org/source.txt",
                "text": "must remain local",
            },
        )
    assert progress[0]["phase"] == "auth" and progress[-1]["phase"] == "completed"
    assert all(event["comparison"]["title"] == "Uploaded source" for event in progress)
    assert (
        result["source_url"] == "https://example.org/source.txt"
        and "text" not in result["document"]
    )
    assert any(
        event["phase"] == "rlm" and event["comparison"]["rlm"]["steps"] for event in progress
    )
    assert result["direct"]["quality"]["matched"] == result["rlm"]["quality"]["matched"] == 1
    assert result["evaluation_version"] == "reference-fields-aliases-v1"
    assert "aliases" in result["evaluation_label"]
    assert result["reference_aliases"] == {"answer": ["AliasSecret"]}
    assert all(
        event["comparison"]["evaluation_version"] == result["evaluation_version"]
        for event in progress
    )
    assert json.loads((tmp_path / "graded.raw.json").read_text())["reference_aliases"] == {
        "answer": ["AliasSecret"]
    }
    for transport in (direct, recursive):
        for call in transport.call_args_list:
            assert "ReferenceSecret" not in json.dumps(call.kwargs["messages"])
            assert "AliasSecret" not in json.dumps(call.kwargs["messages"])
    assert result["query"] in direct.call_args.kwargs["messages"][-1]["content"]
    assert result["query"] == recursive.call_args_list[0].kwargs["messages"][1]["content"]


def finished_comparison():
    lane = {"seconds": 0, "calls": 1, "tokens": 2, "result": "done", "steps": [], "passed": None}
    return {"direct": dict(lane), "rlm": dict(lane)}


@contextmanager
def running_server(tmp_path, capture=None):
    state = CompanionState(tmp_path, capture=capture)
    server = CompanionServer(0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, state
    finally:
        server.shutdown()
        server.server_close()
        state.close()
        thread.join(timeout=2)


def request(
    server, method, path, body=None, *, auth=True, origin="http://localhost:5173", headers=None
):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    request_headers = {"Origin": origin} if origin else {}
    if auth:
        request_headers["Authorization"] = "Bearer " + server.state.token
    if body is not None:
        body = json.dumps(body)
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    data = response.read()
    result = response.status, dict(response.getheaders()), json.loads(data) if data else None
    connection.close()
    return result


def wait_finished(state, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        run = state.snapshot(run_id)
        if run["state"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.01)
    raise AssertionError("mock run did not finish")


def test_loopback_auth_host_origin_and_pna(tmp_path):
    with running_server(tmp_path) as (server, state):
        assert server.server_address[0] == "127.0.0.1"
        assert request(server, "GET", "/health", auth=False)[0] == 401
        assert request(server, "GET", "/health", origin="https://evil.example")[0] == 403
        assert request(server, "GET", "/health", headers={"Host": "evil.example"})[0] == 403
        status, headers, health = request(server, "GET", "/health")
        assert status == 200 and health["authenticated"] and health["service"] == "rlm-compare"
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        status, headers, _ = request(
            server,
            "OPTIONS",
            "/runs",
            auth=False,
            headers={
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert status == 204 and headers["Access-Control-Allow-Private-Network"] == "true"
        assert (
            request(
                server,
                "OPTIONS",
                "/runs",
                auth=False,
                headers={"Access-Control-Request-Method": "DELETE"},
            )[0]
            == 403
        )
        assert request(server, "POST", "/documents", {"path": "/private/a.txt"})[0] == 400
        assert request(server, "POST", "/presets/arbitrary/download", {})[0] == 404
        assert request(server, "GET", "/health?token=ignored")[0] == 400
        assert len(request(server, "GET", "/presets")[2]["presets"]) == 3
    with pytest.raises(ValueError):
        _origin("http://evil.example")


def test_upload_run_progress_single_active_and_cancel(tmp_path):
    observed = []

    async def blocking_capture(**kwargs):
        observed.append(kwargs)
        kwargs["progress_handler"]({"phase": "direct", "comparison": finished_comparison()})
        await asyncio.Event().wait()

    with running_server(tmp_path, capture=blocking_capture) as (server, state):
        status, _, document = request(
            server,
            "POST",
            "/documents",
            {
                "filename": "document.txt",
                "content_base64": base64.b64encode(b"Hello source").decode(),
            },
        )
        assert status == 201 and document["preview"] == "Hello source" and "text" not in document
        assert (
            request(
                server, "POST", "/runs", {"document_id": document["document_id"], "question": ""}
            )[0]
            == 400
        )
        status, _, run = request(
            server,
            "POST",
            "/runs",
            {"document_id": document["document_id"], "question": "Question?"},
        )
        assert status == 202
        assert (
            request(
                server,
                "POST",
                "/runs",
                {"document_id": document["document_id"], "question": "Again?"},
            )[0]
            == 409
        )
        deadline = time.monotonic() + 5
        while not observed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert request(server, "GET", "/runs/" + run["run_id"])[2]["comparison"]
        assert request(server, "POST", "/runs/" + run["run_id"] + "/cancel", {})[0] == 202
        assert wait_finished(state, run["run_id"])["state"] == "cancelled"
        assert state.active_run is None and observed[0]["expected_fields"] == {}


def test_preset_reference_defaults_only_apply_to_the_original_question(tmp_path):
    observed = []

    async def capture(**kwargs):
        observed.append(kwargs)
        return finished_comparison()

    with running_server(tmp_path, capture=capture) as (server, state):
        preset = PRESETS["python-csv-docs"]
        state.documents["preset"] = {
            "text": "source",
            "metadata": {
                "document_id": "preset",
                "suggested_question": preset["suggested_question"],
                "expected_fields": preset["expected_fields"],
            },
        }
        for question, override, expected in [
            (preset["suggested_question"], None, preset["expected_fields"]),
            ("Edited question", None, {}),
            (preset["suggested_question"], {}, {}),
        ]:
            payload = {"document_id": "preset", "question": question}
            if override is not None:
                payload["expected_fields"] = override
            run = state.start_run(payload)
            assert wait_finished(state, run["run_id"])["state"] == "completed"
            assert observed[-1]["expected_fields"] == expected


def test_preset_aliases_require_pinned_source_unchanged_question_and_default_refs(
    tmp_path, monkeypatch
):
    observed = []

    async def capture(**kwargs):
        observed.append(kwargs)
        return finished_comparison()

    data = b"Pinned book text"
    aliases = {"trial_judge": ["The King", "King of Hearts", "The King of Hearts"]}
    preset = {
        **PRESETS["alice-gutenberg-11"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "expected_aliases": aliases,
    }
    monkeypatch.setitem(PRESETS, preset["id"], preset)
    monkeypatch.setattr(
        "examples.serve_codex_comparison.bounded_worker",
        lambda operation, payload: _extract_text(*payload),
    )
    with running_server(tmp_path, capture=capture) as (server, state):
        with pytest.raises(RequestError, match="SHA256"):
            state.prepare(preset["filename"], b"Changed source", preset)
        document = state.prepare(preset["filename"], data, preset)
        assert document["expected_aliases"] == aliases
        identifier = document["document_id"]
        baseline = {"document_id": identifier, "question": preset["suggested_question"]}
        assert request(server, "POST", "/runs", {**baseline, "expected_aliases": aliases})[0] == 400
        assert not observed
        for options, expected in [
            ({}, aliases),
            ({"question": "Edited question"}, {}),
            ({"expected_fields": {}}, {}),
            ({"expected_fields": preset["expected_fields"]}, {}),
            ({"expected_fields": None}, {}),
        ]:
            run = state.start_run({**baseline, **options})
            assert wait_finished(state, run["run_id"])["state"] == "completed"
            assert observed[-1]["expected_aliases"] == expected
        metadata = state.documents[identifier]["metadata"]
        for changed in [
            {"sha256": "0" * 64},
            {"preset_id": None},
            {"expected_aliases": {"trial_judge": ["Queen"]}},
        ]:
            original = {key: metadata[key] for key in changed}
            metadata.update(changed)
            run = state.start_run(baseline)
            assert wait_finished(state, run["run_id"])["state"] == "completed"
            assert observed[-1]["expected_aliases"] == {}
            metadata.update(original)
