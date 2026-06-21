# Parsers

Convert raw files (PDF, DOCX, HTML, code) into `ParsedDocument` objects with
structured sections, preserved headings, and tables as markdown.

## Files

| File | Input | Key Behaviour |
|---|---|---|
| `file_router.py` | Any file | MIME detection → correct parser |
| `pdf_parser.py` | PDF | pdfplumber, table detection by bbox, page tracking |
| `docx_parser.py` | DOCX | python-docx, heading styles H1/H2/H3 preserved |
| `html_parser.py` | HTML | bs4, strips nav/footer/header/ads |
| `code_parser.py` | Source code | tree-sitter AST, splits at function/class boundaries |
| `base.py` | — | `BaseParser` ABC, `ParsedDocument`, `ParsedSection` dataclasses |

## Adding a New Parser

1. Create `{type}_parser.py` extending `BaseParser`
2. Implement `parse(file_path: Path, doc_id: str) -> ParsedDocument`
3. Register MIME types in `file_router.py`:
   ```python
   MIME_MAP["application/vnd.ms-excel"] = XLSXParser
   ```
4. Add unit tests in `tests/unit/test_parser.py`

## Non-Negotiable Rules

- Tables → always `section_type="table"` (never split across sections)
- Page numbers → always tracked on every `ParsedSection`
- Heading hierarchy → always preserved (nearest ancestor heading tracked)
