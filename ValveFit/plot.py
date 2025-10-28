import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def plot_valve_wireframe(
    xyz, gridshape, color="green", linewidth=1, dpi=150, filepath=None, format="png"
):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    xyz = np.array(xyz).reshape(*gridshape, 3)

    first_edge = xyz[:, 0, :]

    xyz = np.concatenate((xyz, first_edge[:, np.newaxis, :]), axis=1)

    ax.plot_wireframe(
        xyz[..., 0], xyz[..., 1], xyz[..., 2], color=color, linewidth=linewidth
    )
    ax.set_axis_off()
    ax.grid(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0)
    plt.show()
    if filepath is not None:
        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            format=format,
        )


def plot_log_loss(loss_list, title="Loss", logscale=False, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    loss_array = np.array(jnp.array(loss_list)).reshape(-1)
    ax.plot(loss_array)
    if logscale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()
    plt.close(fig)


def plot_NND_histogram(
    NND, bins=10, edge_color="black", color="tab:blue", filename=None
):
    plt.figure(figsize=(8, 6))
    plt.hist(np.array(NND), bins=bins, color=color, ec=edge_color)
    plt.xlabel("Nearest Neighbour Distance")
    plt.ylabel("Frequency")
    plt.gca().set_position([0.1, 0.1, 0.85, 0.85])
    plt.tight_layout(pad=0.5)
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def plot_multiple_histograms(
    shape,
    NNDs,
    edge_color="black",
    color="tab:blue",
    labels=None,
    normalize=False,
    filename=None,
):
    fig, axs = plt.subplots(*shape, figsize=(12, 8))

    xlabel = "NND"
    if normalize:
        gmax = max(arr.max() for arr in NNDs)
        gmin = min(arr.min() for arr in NNDs)
        NNDs = [(NND - gmin) / (gmax - gmin) for NND in NNDs]
        xlabel = "Normalized NND"

    for i, ax in enumerate(axs.flat):
        ax.hist(NNDs[i], bins=10, color=color, ec=edge_color)
        if labels is not None:
            ax.set_title(labels[i])
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_xlim(0, 1)

        # Set x-axis ticks between 0 and 1
        ax.set_xticks(np.linspace(0, 1, num=6))
        ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout(pad=1.5)

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)

    plt.show()


def plot_histogram(
    data,
    bins=None,
    xticks=None,
    yticks=None,
    xlabel=None,
    ylabel=None,
    figsize=(10, 6),
    edgecolor="black",
    font_size=12,
    output_file="histogram.png",
    dpi=300,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    weights = np.ones_like(data) * 100 / len(data)

    if bins is not None:
        ax.hist(data, bins=bins, weights=weights, edgecolor=edgecolor)
    else:
        ax.hist(data, weights=weights, edgecolor=edgecolor)

    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    plt.tight_layout()

    plt.savefig(output_file, dpi=dpi)
    plt.close()
    print(f"Histogram saved to {output_file}")
