from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EphemerisResult:
    """
    Container for ephemeris data.
    """

    epoch: np.ndarray

    theta: np.ndarray | None = None
    rho: np.ndarray | None = None

    X: np.ndarray | None = None
    Y: np.ndarray | None = None

    rv1: np.ndarray | None = None
    rv2: np.ndarray | None = None

    # ------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------

    def to_dataframe(self):

        data = {
            "Epoch": self.epoch
        }

        if self.theta is not None:
            data["Theta"] = self.theta

        if self.rho is not None:
            data["Rho"] = self.rho

        if self.X is not None:
            data["X"] = self.X

        if self.Y is not None:
            data["Y"] = self.Y

        if self.rv1 is not None:
            data["RV1"] = self.rv1

        if self.rv2 is not None:
            data["RV2"] = self.rv2

        return pd.DataFrame(data)

    # ------------------------------------------------------
    # CSV export
    # ------------------------------------------------------

    def save_csv(
        self,
        filename
    ):

        self.to_dataframe().to_csv(
            filename,
            index=False
        )

    # ------------------------------------------------------
    # Text export
    # ------------------------------------------------------

    def to_text(
        self,
        float_format="{:.6f}"
    ):
        """
        Return formatted text table.
        """

        df = self.to_dataframe()

        columns = list(df.columns)

        widths = {}

        for col in columns:

            widths[col] = max(
                len(col),
                12
            )

        lines = []

        header = ""

        for col in columns:

            header += (
                f"{col:<{widths[col]}} "
            )

        lines.append(
            header.rstrip()
        )

        lines.append(
            "-" * len(header)
        )

        for _, row in df.iterrows():

            line = ""

            for col in columns:

                value = row[col]

                if isinstance(
                    value,
                    (float, np.floating)
                ):

                    text = float_format.format(
                        value
                    )

                else:

                    text = str(value)

                line += (
                    f"{text:<{widths[col]}} "
                )

            lines.append(
                line.rstrip()
            )

        return "\n".join(lines)

    # ------------------------------------------------------
    # Text file export
    # ------------------------------------------------------

    def save_text(
        self,
        filename
    ):

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                self.to_text()
            )