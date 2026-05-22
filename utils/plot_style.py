"""
plot_style.py — Unified EMNLP-quality plot style for all experiments.

Usage:
    from utils.plot_style import *
    # or
    from utils.plot_style import ATTN_COLOR, MLP_COLOR, DOMAIN_COLORS, setup_style
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Dimensions ──
TEXT_WIDTH = 6.75   # inches (two-column EMNLP)
COL_WIDTH = 3.25    # inches (single column)

# ── Component colours ──
ATTN_COLOR = "#D62728"
MLP_COLOR  = "#1F77B4"
RES_COLOR  = "#7F7F7F"

# ── Per-domain colours (6 arXiv domains) ──
DOMAIN_COLORS = {
    "cs":      "#E74C3C",
    "eess":    "#F39C12",
    "math":    "#2ECC71",
    "physics": "#3498DB",
    "q-bio":   "#9B59B6",
    "stat":    "#1ABC9C",
}
DOMAIN_COLORS_LIST = list(DOMAIN_COLORS.values())

# ── Accent colours for multi-curve plots ──
ACCENT_COLORS = ["#E74C3C", "#F39C12", "#2ECC71", "#3498DB", "#9B59B6", "#1ABC9C",
                 "#E67E22", "#1ABC9C", "#2C3E50", "#D35400"]


def setup_style():
    """Apply the EMNLP publication rcParams globally."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def save_fig(fig, fig_dir, name, log=None, extensions=("pdf", "png")):
    """Save figure in multiple formats."""
    import os
    os.makedirs(fig_dir, exist_ok=True)
    for ext in extensions:
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=300)
    plt.close(fig)
    if log is not None:
        log(f"  {name}")
    else:
        print(f"  ✓ {name}")


# Apply style on import
setup_style()
