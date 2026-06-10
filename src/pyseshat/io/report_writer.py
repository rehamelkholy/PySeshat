from pathlib import Path

def write_text_report(
    report_text,
    filename
):
    """
    Write report to text file.

    Parameters
    ----------
    report_text : str

    filename : str or Path
    """

    filename = Path(
        filename
    )

    filename.write_text(
        report_text,
        encoding="utf-8"
    )

    return filename