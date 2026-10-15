from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import ContentStream, DictionaryObject, NameObject

from rag.exceptions import RagError
from rag.pdf_loader import list_pdfs, load_pdf, source_label


def _write_text_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=200)
    contents = ContentStream(None, None)
    literal = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    contents.set_data(f"BT /F1 12 Tf 36 120 Td ({literal}) Tj ET".encode())
    page.replace_contents(contents)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    with path.open("wb") as handle:
        writer.write(handle)


def test_load_pdf_returns_page_text(tmp_path: Path) -> None:
    path = tmp_path / "policy.pdf"
    _write_text_pdf(path, "Refunds are allowed within 30 days.")
    pages = load_pdf(path)
    assert len(pages) == 1
    assert pages[0].number == 1
    assert "30 days" in pages[0].text


def test_missing_pdf_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RagError, match="PDF not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_list_pdfs_finds_nested_files_and_skips_other_types(tmp_path: Path) -> None:
    nested = tmp_path / "docs"
    nested.mkdir()
    _write_text_pdf(nested / "policy.pdf", "hello")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    found = list_pdfs(tmp_path)
    assert found == [nested / "policy.pdf"]


def test_source_label_is_relative_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data" / "policy.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF")
    assert source_label(path) == "data/policy.pdf"
