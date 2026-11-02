"""Local ingestion preserves source coordinates and enforces extraction bounds."""

import io
from dataclasses import replace

import pytest

from rlm.document_io import (
    DocumentExtractionWarning,
    DocumentLimits,
    DocumentLoadError,
    load_document,
    read_document_bytes,
)


def pdf_bytes(texts, password=None):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if not text:
            continue
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
        content.set_data(b"BT /F1 12 Tf 50 700 Td (" + text.encode("ascii") + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        page.compress_content_streams()
    if password is not None:
        writer.encrypt(user_password=password, owner_password="owner-password")
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_text_loader_preserves_coordinates_and_does_not_accept_urls(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"\xef\xbb\xbfFirst\r\nSecond\rThird\n")
    document = load_document(path, document_id="notes")
    assert document.document_id == "notes" and document.title == "notes.txt"
    assert document.pages == ("First\r\nSecond\rThird\n",)
    assert load_document(path).document_id == load_document(path).document_id
    with pytest.raises(OSError):
        load_document("https://example.com/notes.txt")


@pytest.mark.parametrize("data", [b"", b" \r\n", b"\xff", b"binary\x00content"])
def test_text_loader_rejects_unreadable_input(data):
    with pytest.raises(DocumentLoadError):
        read_document_bytes("bad.txt", data)


def test_text_limits_include_input_bytes_and_decoded_characters(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes("é".encode() * 8)
    with pytest.raises(DocumentLoadError, match="bytes"):
        load_document(path, limits=DocumentLimits(max_bytes=15))
    with pytest.raises(DocumentLoadError, match="characters"):
        load_document(path, limits=DocumentLimits(max_chars=7))
    assert load_document(path, limits=DocumentLimits(max_bytes=16, max_chars=8)).pages == ("é" * 8,)
    with pytest.raises(DocumentLoadError, match="Supported"):
        read_document_bytes("image.png", b"not text")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_document_limits_reject_invalid_values(value):
    with pytest.raises(ValueError, match="max_pages"):
        DocumentLimits(max_pages=value)


def test_pdf_preserves_physical_page_positions_and_warns_about_ocr(tmp_path):
    path = tmp_path / "evidence.pdf"
    path.write_bytes(pdf_bytes(["Printed page 30", "", "Printed page 32"]))
    with pytest.warns(DocumentExtractionWarning, match="no OCR"):
        document = load_document(path, document_id="evidence")
    assert document.pages == ("Printed page 30", "", "Printed page 32")
    assert document.document_id == "evidence"


def test_pdf_with_empty_user_password_is_readable_but_real_password_is_required():
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes("restricted.pdf", pdf_bytes(["Visible"], password=""))
    assert document.pages == ("Visible",)
    with pytest.raises(DocumentLoadError, match="requires a password"):
        read_document_bytes("locked.pdf", pdf_bytes(["Secret"], password="secret"))


def test_pdf_limits_cover_pages_combined_text_and_expanded_streams():
    data = pdf_bytes(["First page", "Second page"])
    limits = DocumentLimits()
    with pytest.raises(DocumentLoadError, match="1-page"):
        read_document_bytes("pages.pdf", data, limits=replace(limits, max_pages=1))
    # The newline used by callers to join pages also counts against the character limit.
    with pytest.raises(DocumentLoadError, match="characters"):
        read_document_bytes("pages.pdf", data, limits=replace(limits, max_chars=21))
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes("pages.pdf", data, limits=replace(limits, max_chars=22))
    assert len("\n".join(document.pages)) == 22
    with pytest.raises(DocumentLoadError, match="expanded|extract"):
        read_document_bytes("pages.pdf", data, limits=replace(limits, max_expanded_bytes=10))
    with pytest.raises(DocumentLoadError, match="expanded content"):
        read_document_bytes("pages.pdf", data, limits=replace(limits, max_expanded_bytes=70))


def test_pdf_empty_and_malformed_input_fail_clearly():
    with pytest.raises(DocumentLoadError, match="OCR"):
        read_document_bytes("scan.pdf", pdf_bytes([""]))
    with pytest.raises(DocumentLoadError, match="header"):
        read_document_bytes("wrong.pdf", b"hello")
    with pytest.raises(DocumentLoadError, match="extract PDF"):
        read_document_bytes("broken.pdf", b"%PDF-1.7\ninvalid")


def test_pdf_decompression_configuration_is_restored_after_success_and_failure():
    pypdf = pytest.importorskip("pypdf")
    before = pypdf.get_configuration()
    with pytest.warns(DocumentExtractionWarning):
        read_document_bytes("a.pdf", pdf_bytes(["Hello"]), limits=DocumentLimits())
    assert pypdf.get_configuration() == before
    with pytest.raises(DocumentLoadError):
        read_document_bytes("a.pdf", b"%PDF-invalid", limits=DocumentLimits())
    assert pypdf.get_configuration() == before


def test_pdf_budget_counts_font_unicode_maps_before_text_extraction(monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(pdf_bytes(["A"]))))
    page = writer.pages[0]
    cmap = DecodedStreamObject()
    cmap.set_data(
        b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        b"1 beginbfchar\n<41> <0041>\nendbfchar\nendcmap\nend\nend\n" + b" " * 900
    )
    page["/Resources"]["/Font"]["/F1"][NameObject("/ToUnicode")] = writer._add_object(
        cmap.flate_encode()
    )
    exact_expanded_bytes = len(cmap.get_data()) + len(page.get_contents().get_data())
    stream = io.BytesIO()
    writer.write(stream)
    called = []
    original = pypdf.PageObject.extract_text

    def observed_extract(self, *args, **kwargs):
        called.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pypdf.PageObject, "extract_text", observed_extract)
    with pytest.raises(DocumentLoadError, match="expanded"):
        read_document_bytes(
            "font-map.pdf",
            stream.getvalue(),
            limits=DocumentLimits(max_expanded_bytes=exact_expanded_bytes - 1),
        )
    assert called == []
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "font-map.pdf",
            stream.getvalue(),
            limits=DocumentLimits(max_expanded_bytes=exact_expanded_bytes),
        )
    assert document.pages == ("A",) and called == [True]


def test_pdf_text_budget_does_not_decode_image_resources(monkeypatch):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        EncodedStreamObject,
        NameObject,
        NumberObject,
    )

    writer = pypdf.PdfWriter(clone_from=pypdf.PdfReader(io.BytesIO(pdf_bytes(["Visible"]))))
    page = writer.pages[0]
    image = DecodedStreamObject()
    image.set_data(b"x" * 100_000)
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(100),
            NameObject("/Height"): NumberObject(1000),
            NameObject("/ColorSpace"): NameObject("/DeviceGray"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    page["/Resources"][NameObject("/XObject")] = DictionaryObject(
        {NameObject("/Picture"): writer._add_object(image.flate_encode())}
    )
    content = DecodedStreamObject()
    content.set_data(page.get_contents().get_data() + b"\n/Picture Do\n")
    page[NameObject("/Contents")] = writer._add_object(content.flate_encode())
    stream = io.BytesIO()
    writer.write(stream)
    original = EncodedStreamObject.get_data

    def forbid_image(self):
        if self.get("/Subtype") == "/Image":
            raise AssertionError("Text ingestion must not decode image resources")
        return original(self)

    monkeypatch.setattr(EncodedStreamObject, "get_data", forbid_image)
    with pytest.warns(DocumentExtractionWarning):
        document = read_document_bytes(
            "picture.pdf", stream.getvalue(), limits=DocumentLimits(max_expanded_bytes=200)
        )
    assert document.pages == ("Visible\n",)
