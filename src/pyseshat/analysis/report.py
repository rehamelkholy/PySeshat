def build_fit_report(
    fit_result,
    statistics,
    parameter_uncertainties=None,
    masses=None
):
    """
    Assemble a complete orbit-fit report.

    Returns
    -------
    dict
    """

    report = {

        "success":
            fit_result.optimizer_result.success,

        "message":
            fit_result.optimizer_result.message,

        "elements":
            fit_result.elements,

        "statistics":
            statistics
    }

    if parameter_uncertainties is not None:

        report[
            "uncertainties"
        ] = parameter_uncertainties

    if masses is not None:

        report[
            "masses"
        ] = masses

    return report

def report_to_text(
    report
):

    lines = []

    lines.append(
        "FIT REPORT"
    )

    lines.append(
        "=" * 40
    )

    lines.append(
        f"Success: {report['success']}"
    )

    lines.append(
        f"Message: {report['message']}"
    )

    lines.append("")

    lines.append(
        "Orbital Elements"
    )

    lines.append(
        "-" * 20
    )

    elements = report[
        "elements"
    ]

    for name in elements.PARAMETER_ORDER:

        lines.append(
            f"{name:8s} "
            f"{getattr(elements,name):.6f}"
        )

    if "uncertainties" in report:

        lines.append("")
        lines.append(
            "Uncertainties"
        )

        for k, v in report[
            "uncertainties"
        ].items():

            lines.append(
                f"{k:8s} "
                f"{v:.6f}"
            )

    lines.append("")
    lines.append(
        "Statistics"
    )

    for k, v in report[
        "statistics"
    ].items():

        lines.append(
            f"{k}: {v}"
        )

    if "masses" in report:

        lines.append("")
        lines.append(
            "Masses"
        )

        for k, v in report[
            "masses"
        ].items():

            lines.append(
                f"{k}: {v}"
            )

    return "\n".join(lines)