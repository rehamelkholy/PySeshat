import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_separation_vs_time(
    epochs,
    rho,
    title="Separation vs Time",
    show=True
):

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        rho
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Separation (arcsec)"
    )

    ax.set_title(
        title
    )

    ax.grid(True)

    if show:
        plt.show()

    return fig, ax