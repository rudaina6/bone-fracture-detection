"""Shared plotting style for the Phase 5 figures, so every figure in the report looks like the others.

    from phase5_viz import C, setup, save, REGION_LABEL

Colours: two categorical slots (blue = region-specific, orange = unified) that are colour-blind safe
(worst-pair CVD dE 24.7, normal-vision dE 33.6), a muted grey for reference lines (chance level, zero),
and a one-hue blue ramp for the heat maps. Identity is never carried by colour alone: every figure with
two series has a legend, and the heat maps are annotated with their numbers.
These are static report figures, so only the light surface is defined.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = dict(
    region="#2a78d6",        # categorical slot 1 - region-specific model
    unified="#eb6834",       # categorical slot 2 - unified model
    third="#1baf7a",         # categorical slot 3 - a third series where one is needed
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#8a8a85",         # chance / zero reference lines, grid
    surface="#fcfcfb",
    pos="#2a78d6",           # fractured
    neg="#8a8a85",           # not fractured
)

# one-hue sequential ramp (blue 100 -> 700), for the transfer / confusion heat maps
SEQ = LinearSegmentedColormap.from_list("p5seq", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#5598e7",
                                                 "#2a78d6", "#184f95", "#0d366b"])
# diverging blue <-> red with a grey midpoint, for cosine similarity (-1 .. 1)
DIV = LinearSegmentedColormap.from_list("p5div", ["#184f95", "#5598e7", "#f0efec", "#e07a7a", "#a52222"])

REGION_LABEL = dict(hand="Hand", leg="Leg", hip="Hip", shoulder="Shoulder")
FAMILY_LABEL = dict(hog="HOG", lbp="LBP", glcm="GLCM", gabor="Gabor", intensity="Intensity")


def setup():
    """Recessive grid and axes, thin marks, text in ink tokens rather than series colours."""
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "savefig.bbox": "tight", "savefig.dpi": 160,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.titlelocation": "left",
        "text.color": C["ink"], "axes.labelcolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "axes.edgecolor": "#d8d8d2", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e8e8e2", "grid.linewidth": 0.7, "axes.grid": True, "axes.grid.axis": "y",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def header(fig, title, legend_from=None, ncols=3, top=0.86, legend_y=0.925):
    """Title and (optionally) the figure's one legend in a reserved band above the axes, so neither can
    collide with the data whatever the numbers turn out to be."""
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.suptitle(title, x=0.005, y=0.995, ha="left", va="top", fontsize=11, fontweight="bold")
    if legend_from is not None:
        h, l = legend_from.get_legend_handles_labels()
        fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, legend_y), ncols=ncols, frameon=False)


def save(fig, path, caption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.text(0.0, -0.02, caption, ha="left", va="top", fontsize=7.5, color=C["ink2"], wrap=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"    figure -> {path}", flush=True)
