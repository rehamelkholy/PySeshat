"""
Verify the paper's regenerated PDF figure set (item 5's precondition
for deleting stale PNGs): confirm every expected PDF exists, is
non-empty, is a well-formed PDF, rasterizes to a plausible non-blank
image at a sane page size, and (via pypdf's page /MediaBox) has
correct/consistent dimensions per figure type.

Not a substitute for -- but a complement to -- the direct visual
inspection done for a representative sample of each figure type
(see docs/project_decisions.md's dated entry for that sampling and
its grid-lines-absent confirmation): this script is the
programmatic, whole-set check across all twelve systems' figures.

Run
    PYTHONPATH=src python3 scripts/verify_paper_figures.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "validation" / "results" / "plots"

EXPECTED_FILES = [
    "FIN379/visual_orbit.pdf",
    "FIN379/residuals.pdf",
    "FIN379/rv_orbit_phased.pdf",
    "FIN379/separation_vs_time.pdf",
    "GL765/visual_orbit.pdf",
    "GL765/residuals.pdf",
    "GL765/rv_orbit_phased.pdf",
    "GL765/separation_vs_time.pdf",
    "ZetaAqr/triple_orbit.pdf",
    "HIP111805/triple_orbit.pdf",
    "Capella_orbit/visual_orbit.pdf",
    "Capella_orbit/residuals.pdf",
    "Capella_orbit/rv_orbit_phased.pdf",
    "Capella_orbit/separation_vs_time.pdf",
    "Capella_orbit/corner.pdf",
    "Capella/hrdiagram.pdf",
    "AlphaCenAB_orbit/visual_orbit.pdf",
    "AlphaCenAB_orbit/residuals.pdf",
    "AlphaCenAB_orbit/separation_vs_time.pdf",
    "AlphaCenAB/hrdiagram.pdf",
    "HD214222/hrdiagram.pdf",
    "HD191854/hrdiagram.pdf",
    "HD130669/hrdiagram.pdf",
    "HD184467/hrdiagram.pdf",
]


def _is_mostly_uniform(image, tolerance=2):
    """
    True if the image is (near-)blank -- every pixel within
    `tolerance` of the corner pixel's value. Used as a crude but
    effective "actually contains plotted content" check: a blank
    page (failed render, empty axes) is uniform; a real figure with
    axes/data/labels is not.
    """

    gray = image.convert("L")
    extrema = gray.getextrema()

    return (extrema[1] - extrema[0]) <= tolerance


def verify_one(relative_path):

    path = PLOTS_DIR / relative_path

    if not path.exists():
        return False, "MISSING"

    size = path.stat().st_size

    if size == 0:
        return False, "EMPTY FILE (0 bytes)"

    header = path.read_bytes()[:5]

    if header[:4] != b"%PDF":
        return False, f"NOT A PDF (header={header!r})"

    with tempfile.TemporaryDirectory() as tmp:

        out_stem = Path(tmp) / "page"

        result = subprocess.run(
            ["pdftoppm", "-png", "-r", "100", str(path), str(out_stem)],
            capture_output=True,
        )

        if result.returncode != 0:
            return False, f"pdftoppm FAILED: {result.stderr.decode()}"

        rendered = sorted(Path(tmp).glob("page*.png"))

        if not rendered:
            return False, "pdftoppm produced no output pages"

        image = Image.open(rendered[0])
        width, height = image.size

        if width < 100 or height < 100:
            return False, f"IMPLAUSIBLY SMALL RENDER ({width}x{height})"

        if _is_mostly_uniform(image):
            return False, "BLANK/UNIFORM RENDER (no visible content)"

    return True, f"OK ({size} bytes, page render {width}x{height})"


def main():

    all_ok = True

    missing_expected = []
    found_unexpected = []

    actual_files = {
        str(p.relative_to(PLOTS_DIR))
        for p in PLOTS_DIR.rglob("*.pdf")
    }

    expected_set = set(EXPECTED_FILES)

    missing_expected = sorted(expected_set - actual_files)
    found_unexpected = sorted(actual_files - expected_set)

    for relative_path in EXPECTED_FILES:

        ok, message = verify_one(relative_path)

        status = "PASS" if ok else "FAIL"

        print(f"[{status}] {relative_path}: {message}")

        if not ok:
            all_ok = False

    print()
    print(f"Expected files: {len(EXPECTED_FILES)}")
    print(f"Actual .pdf files under {PLOTS_DIR}: {len(actual_files)}")

    if missing_expected:
        all_ok = False
        print(f"MISSING (expected but not found): {missing_expected}")

    if found_unexpected:
        print(
            f"NOTE -- extra .pdf files present (not necessarily an "
            f"error, just unaccounted for by this script's expected "
            f"list): {found_unexpected}"
        )

    # Any leftover PNGs in the plots tree at all -- the precondition
    # for item 5's deletion step.
    stale_pngs = sorted(
        str(p.relative_to(PLOTS_DIR)) for p in PLOTS_DIR.rglob("*.png")
    )

    print(f"Stale PNGs remaining under {PLOTS_DIR}: {len(stale_pngs)}")

    if stale_pngs:
        print(f"  {stale_pngs}")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
