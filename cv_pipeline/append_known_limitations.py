#!/usr/bin/env python3
"""
append_known_limitations.py — one-off script that appends a new "Known
Limitations" section (Section 14) to Real-Time_Soccer_Analytics_Pipeline_v4.pdf
as new pages, without touching any existing page.

Why a separate append rather than re-exporting the whole document: this PDF
has no editable source (confirmed: PDF metadata shows Creator=Writer,
Producer=LibreOffice 24.2 - it was written and exported from LibreOffice
Writer directly, and no script/template anywhere in this repo generates it).
Re-exporting from scratch isn't possible without the original .odt; a true
page-level append via pypdf (copying each existing page object as-is, never
decoding/re-rendering it) is the only way to add content without risking
altering anything already there.

The new section's body content is built separately with reportlab (already
a dependency of this project - see dashboard/app.py's per-match PDF export)
as its own small PDF, then every one of ITS pages is appended after the
original document's own last page.

Usage:
    python append_known_limitations.py
"""
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth

CV_PIPELINE_DIR = Path(__file__).parent
TARGET_PDF = CV_PIPELINE_DIR / "Real-Time_Soccer_Analytics_Pipeline_v4.pdf"
DUPLICATE_PDF = Path(r"C:\UCLA\yolo model 2\Real-Time_Soccer_Analytics_Pipeline_v4.pdf")

# ---------------------------------------------------------------------------
# Section content - every number here is a real, verified figure pulled
# directly from this revision's two curated-match stats.json files and from
# the orientation-flip investigation conducted this session (live keypoint
# back-projection against real broadcast frames), not estimated.
# ---------------------------------------------------------------------------

SECTION_TITLE = "14  Known Limitations (Addendum)"

INTRO = (
    "This report's own conclusion states plainly that calibration accuracy and tracking "
    "identity remain open challenges. This addendum quantifies both directly, with figures "
    "measured from this revision's two curated match segments, and documents a calibration "
    "defect identified after the report above was written."
)

SUB1_TITLE = "14.1  Calibration: a second camera-framing failure mode, beyond Section 2.4.8"
SUB1 = [
    "Section 2.4.8 documented that the per-frame calibration confidence score does not "
    "reliably predict whether a frame's homography is actually correct, tracing one concrete "
    "failure mode to the pose model confusing a penalty-box D-arc for the centre circle under "
    "tight camera framing. A follow-up investigation, prompted by a visible error in a derived "
    "top-down visualisation, found a second, related failure mode with the same underlying "
    "cause but a more severe consequence: under a sufficiently tight zoom on a single penalty "
    "box, the keypoint model can mistake that box's own landmarks for the mirror-image "
    "landmarks of the opposite box, producing a homography that is internally self-consistent "
    "— it passes every existing per-frame check, including the ambiguous-cluster detector "
    "built for the Section 2.4.8 defect — yet places every tracked player on the wrong "
    "half of the pitch. This was confirmed directly, not inferred: back-projecting a frame's "
    "own detected keypoints showed real on-screen landmarks near one penalty box being "
    "assigned the real-world coordinates of the other.",

    "This failure was measured, not estimated, in both of this revision's curated match "
    "segments. Outright calibration failure (no usable homography at all, independent of this "
    "specific defect) affects 8.0% of frames in the Liverpool–PSG segment and 9.4% in the "
    "Real Madrid–Barcelona segment, concentrated in short runs rather than spread evenly. "
    "The newly identified wrong-but-confident orientation defect specifically affects the final "
    "several seconds of both segments' analysis windows — where the broadcast camera "
    "closes in on the box at the climax of the passage of play each segment was auto-selected "
    "for, the same tight-framing trigger condition Section 2.4.8 already identified, here "
    "manifesting as a full left-right misplacement rather than a centre-circle/D-arc mix-up.",

    "A remediation has been designed and is being validated at the time of this writing: a "
    "temporal-consistency check that tracks each accepted frame's own calibrated position and "
    "rejects any later frame whose calibration implies a physically implausible jump since the "
    "last accepted one. The threshold was calibrated against this project's own measured data, "
    "not guessed: genuine camera-pan drift between consecutive sampled frames stayed under "
    "roughly 60 metres/second even during active zooms, while the confirmed-bad frames measured "
    "90–370 metres/second — a wide, unambiguous margin. A rejected frame falls back "
    "to the nearest reliable neighbour, the same honest degraded-but-not-wrong behaviour every "
    "other calibration failure in this system already produces, rather than silently serving a "
    "confident wrong answer. This brings a real cost: frames this check correctly rejects "
    "increase the no-calibration frame count in the affected regions beyond the figures above, "
    "in exchange for removing the wrong-but-confident ones — a trade this report "
    "considers strictly preferable, consistent with the principle applied throughout: an "
    "honestly-withheld result is preferable to a confidently wrong one.",
]

SUB2_TITLE = "14.2  Tracking identity: a quantified fragmentation rate"
SUB2 = [
    "Section 2's identity-linking discussion already documents that ByteTrack, operating on a "
    "single broadcast camera with real occlusion and crowded play, does not guarantee one "
    "stable identity per physical player for an entire analysis window: a player can be lost "
    "and re-acquired under a new track ID after being obscured by another player, the referee, "
    "or a camera cut. This system includes a dedicated merge step specifically to detect and "
    "reunite such fragments, requiring both spatial/temporal proximity and independent "
    "jersey-colour agreement before two fragments are judged to be the same person — "
    "specifically to avoid wrongly merging two different players who happen to pass near each "
    "other.",

    "How often this actually happens, measured directly rather than assumed, is itself "
    "informative about the reliability of single-camera re-identification: of this revision's "
    "two curated segments, 11 of 45 total distinct tracked identities in the Liverpool–"
    "PSG segment (24.4%) and 33 of 82 in the Real Madrid–Barcelona segment (40.2%) were "
    "fragment pairs the merge step identified and reunited — meaning roughly a quarter to "
    "two-fifths of every raw track ID ByteTrack ever emitted in these segments was not a new "
    "person at all, but a continuation of one already seen.",

    "The merge step's own appearance-agreement requirement makes it deliberately conservative "
    "— a candidate pair with an ambiguous jersey-colour match is left unmerged rather "
    "than guessed — so some genuine fragments almost certainly go unrecognised; this "
    "project does not currently have a measurement of that residual rate, and reports only the "
    "confirmed, verified figure above rather than an estimate for the gap. Separately, "
    "team-resolution itself leaves a measurable minority of tracked identities without a "
    "confident team assignment at all — 8 of 45 (17.8%) and 11 of 82 (13.4%) respectively "
    "— typically very short-lived tracks (a misdetection, a substitute warming up, a "
    "ball-boy) that never accumulate enough evidence to resolve either way, and are correctly "
    "excluded from every tactical metric rather than guessed into a team.",
]


def build_addendum_pdf(buf):
    styles = {
        "SectionTitle": ParagraphStyle(
            "SectionTitle", fontName="Times-Bold", fontSize=16, leading=20,
            spaceAfter=14, textColor="#000000",
        ),
        "SubTitle": ParagraphStyle(
            "SubTitle", fontName="Times-Bold", fontSize=13, leading=17,
            spaceBefore=12, spaceAfter=8,
        ),
        "Body": ParagraphStyle(
            "Body", fontName="Times-Roman", fontSize=11, leading=15.5,
            spaceAfter=10, alignment=TA_JUSTIFY,
        ),
    }
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=1 * inch, rightMargin=1 * inch,
        topMargin=1 * inch, bottomMargin=1 * inch,
        title="Real-Time Soccer Analytics Pipeline - Known Limitations Addendum",
        author="Ashwin Ramaseshan",
    )
    story = []
    story.append(Paragraph(SECTION_TITLE, styles["SectionTitle"]))
    story.append(Paragraph(INTRO, styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(SUB1_TITLE, styles["SubTitle"]))
    for para in SUB1:
        story.append(Paragraph(para, styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(SUB2_TITLE, styles["SubTitle"]))
    for para in SUB2:
        story.append(Paragraph(para, styles["Body"]))
    doc.build(story)


def main():
    # ---- 1. Snapshot original content (for the before/after verification
    # the task explicitly requires, not just an assumption the append worked) ----
    orig_reader = PdfReader(str(TARGET_PDF))
    orig_n_pages = len(orig_reader.pages)
    orig_text = [p.extract_text() for p in orig_reader.pages]
    print(f"Original PDF: {orig_n_pages} pages.")

    # ---- 2. Build the new section as its own small PDF in memory ----
    addendum_buf = io.BytesIO()
    build_addendum_pdf(addendum_buf)
    addendum_buf.seek(0)
    addendum_reader = PdfReader(addendum_buf)
    n_new_pages = len(addendum_reader.pages)
    print(f"New addendum section: {n_new_pages} page(s).")

    # ---- 3. Append: copy every original page object as-is, then every new
    # page as-is. add_page() copies the page's own content stream/resources
    # without decoding or re-rendering it - a structural copy, not a re-export. ----
    writer = PdfWriter()
    for page in orig_reader.pages:
        writer.add_page(page)
    for page in addendum_reader.pages:
        writer.add_page(page)

    tmp_path = str(TARGET_PDF) + ".tmp"
    with open(tmp_path, "wb") as f:
        writer.write(f)

    # ---- 4. Verify before touching the real file: re-open the temp output,
    # confirm page count and confirm every ORIGINAL page's extracted text is
    # byte-identical to before the append - not just assumed. ----
    check_reader = PdfReader(tmp_path)
    check_n_pages = len(check_reader.pages)
    expected_n_pages = orig_n_pages + n_new_pages
    if check_n_pages != expected_n_pages:
        raise RuntimeError(f"page count mismatch: expected {expected_n_pages}, got {check_n_pages}")

    for i in range(orig_n_pages):
        new_text = check_reader.pages[i].extract_text()
        if new_text != orig_text[i]:
            raise RuntimeError(f"ORIGINAL PAGE {i} CHANGED - aborting, not replacing the real file. "
                                f"old len={len(orig_text[i])} new len={len(new_text)}")
    print(f"Verified: all {orig_n_pages} original pages are text-identical before/after.")

    import os
    os.replace(tmp_path, TARGET_PDF)
    print(f"Wrote {TARGET_PDF} ({check_n_pages} pages total).")

    if DUPLICATE_PDF.exists():
        import shutil
        shutil.copyfile(TARGET_PDF, DUPLICATE_PDF)
        print(f"Synced duplicate copy at {DUPLICATE_PDF}")


if __name__ == "__main__":
    main()
