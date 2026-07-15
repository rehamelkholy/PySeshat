from pathlib import Path

from astropy.io import fits


# ==========================================================
# CK04 catalog file
# ==========================================================

filename = Path(

    "validation"

) / (

    "atmosphere_models"

) / (

    "catalog.fits"

)


# ==========================================================
# Inspect FITS structure
# ==========================================================

with fits.open(

    filename

) as hdul:

    print()

    print(
        "FITS STRUCTURE"
    )

    print(
        "=" * 70
    )

    hdul.info()

    # ======================================================
    # Inspect each HDU
    # ======================================================

    for index, hdu in enumerate(

        hdul

    ):

        print()

        print(

            f"HDU {index}"

        )

        print(

            "=" * 70

        )

        print(

            "HDU type"

        )

        print(

            type(
                hdu
            ).__name__

        )

        print()

        print(

            "Extension name"

        )

        print(

            hdu.name

        )

        print()

        print(

            "Data shape"

        )

        if hdu.data is None:

            print(
                None
            )

        else:

            print(

                hdu.data.shape

            )

        # ==================================================
        # Inspect table columns
        # ==================================================

        if hasattr(

            hdu,

            "columns"

        ):

            print()

            print(

                "COLUMN DEFINITIONS"

            )

            print(

                "-" * 70

            )

            for column in hdu.columns:

                print(

                    f"{column.name:<25}"

                    f"format={column.format!s:<15}"

                    f"unit={column.unit}"

                )

            # ==============================================
            # Display first catalog rows
            # ==============================================

            if (

                hdu.data is not None

                and

                len(
                    hdu.data
                ) > 0

            ):

                print()

                print(

                    "FIRST FIVE ROWS"

                )

                print(

                    "-" * 70

                )

                number_of_rows = min(

                    5,

                    len(
                        hdu.data
                    )

                )

                for row_index in range(

                    number_of_rows

                ):

                    print()

                    print(

                        f"Row {row_index}"

                    )

                    for name in (

                        hdu.columns.names

                    ):

                        print(

                            f"  {name:<25}"

                            f"{hdu.data[name][row_index]}"

                        )