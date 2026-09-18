"""Build a minimal, valid PDF from plain text - a test fixture generator.

Tests need a real PDF with a real text layer, and depending on a checked-in
binary (or on a heavyweight renderer like reportlab) for that is worse than
emitting the ~60 lines of PDF syntax it actually takes. Everything here is the
1993 PDF 1.4 core: a catalogue, a page tree, one content stream per page, one
built-in font, and a cross-reference table of byte offsets.

This is a fixture, not a PDF library. It handles exactly the case the tests
need: left-aligned Helvetica lines on US Letter pages.
"""

from __future__ import annotations

from pathlib import Path

_PAGE_WIDTH, _PAGE_HEIGHT = 612, 792
_MARGIN_X, _TOP_Y = 72, 720
_FONT_SIZE, _LEADING = 11, 15


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    """A page's drawing commands: begin text, set font, then one line per Tj."""
    parts = [
        "BT",
        f"/F1 {_FONT_SIZE} Tf",
        f"{_LEADING} TL",
        f"{_MARGIN_X} {_TOP_Y} Td",
    ]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")  # advance one line
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def write_pdf(path: str | Path, pages: list[str]) -> Path:
    """Write `pages` (one string per page, newline-separated lines) as a PDF."""
    path = Path(path)

    objects: list[bytes] = []  # objects[i] is object number i+1

    page_count = len(pages)
    # Object numbering: 1 catalogue, 2 page tree, 3 font, then per page a page
    # object and a content stream (2 objects each).
    first_page_obj = 4
    kids = " ".join(f"{first_page_obj + 2 * i} 0 R" for i in range(page_count))

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, page_text in enumerate(pages):
        page_obj = first_page_obj + 2 * i
        content_obj = page_obj + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Contents {content_obj} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode("latin-1")
        )
        stream = _content_stream(page_text.split("\n"))
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    # Serialize, recording the byte offset of each object for the xref table.
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    # The cross-reference table maps object numbers to those byte offsets. Entry
    # 0 is the mandatory free-list head; each entry is exactly 20 bytes.
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")

    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode("latin-1")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


# --- Sample policy text used by the tests -----------------------------------
# Synthetic wording. No real insurer, product or customer is referenced.

SAMPLE_POLICY_PAGES = [
    """PRIVATE MOTOR INSURANCE POLICY
Policy Number: SPEC-0001

SECTION 1 - OWN DAMAGE COVER

1.1 We will indemnify the Insured against loss of or damage to the
Insured Vehicle caused by accidental external means, fire, self
ignition, lightning, burglary, housebreaking or theft.

1.2 The maximum amount payable under this Section shall not exceed
the Insured Declared Value shown in the Schedule.

1.3 A compulsory excess of 5,000 applies to every claim made under
this Section. The excess is deducted before any settlement is paid.""",
    """SECTION 2 - EXCLUSIONS

2.1 We shall not be liable in respect of any claim arising while the
Insured Vehicle is being used for hire or reward, carriage of goods
for payment, racing, pace making, or speed testing.

2.2 We shall not be liable for any claim arising while the vehicle is
driven by a person not holding a valid and effective driving licence.

2.3 Consequential loss, depreciation, wear and tear, mechanical or
electrical breakdown, and failures or breakages are excluded.

2.4 Loss or damage arising from driving under the influence of
intoxicating liquor or drugs is excluded absolutely.""",
    """SECTION 3 - NO CLAIM BONUS

3.1 Where no claim has been made or paid during the preceding period
of insurance, a No Claim Bonus is allowed on renewal at 20 per cent
after one claim free year, rising to 50 per cent after five
consecutive claim free years.

3.2 The No Claim Bonus is forfeited in full if any claim is made
during the period of insurance.

SECTION 4 - CLAIMS PROCEDURE

4.1 Notice of any accident must be given to Us in writing within 48
hours of the occurrence, save in cases of theft, where notice must be
given immediately and the police informed without delay.""",
]
