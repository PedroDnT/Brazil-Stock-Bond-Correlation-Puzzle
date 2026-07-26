"""Shared matplotlib style for all study notebooks.

Usage:
    import sys; sys.path.insert(0, "../config")
    from plot_style import apply_style
    apply_style()
"""

import matplotlib.pyplot as plt

RCPARAMS = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
}


def apply_style():
    """Apply the study-wide matplotlib rcParams."""
    plt.rcParams.update(RCPARAMS)
