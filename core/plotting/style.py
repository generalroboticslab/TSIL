"""Shared plotting style tokens and axis-formatting helpers.

Plotting modules import style tokens from here so that look-and-feel is
defined in exactly one place.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib import ticker as mpl_ticker


# ═══════════════════════════════════════════════════════════════════════
#  Size / font constants
# ═══════════════════════════════════════════════════════════════════════
FontSize = 18
MarkerSize = 16
TickScaler = 1.0
AxisLabelScaler = 1.05
LegendScaler = 1.0
TickSize = FontSize * TickScaler
AxisLabelSize = FontSize * AxisLabelScaler
LegendSize = FontSize * LegendScaler
YBins = 4
LegendHandleLength = 3.0

# ═══════════════════════════════════════════════════════════════════════
#  Single colour palette  (one place to change everything)
# ═══════════════════════════════════════════════════════════════════════
NPG_PALETTE = {
    "navy": "#3C5488",
    "vermillion": "#E64B35",
    "teal": "#00A087",
    "peach": "#F39B7F",
    "slate": "#8491B4",
    "sage": "#91D1C2",
}

NPG_COLORS = [
    NPG_PALETTE["navy"],
    NPG_PALETTE["vermillion"],
    NPG_PALETTE["teal"],
    NPG_PALETTE["peach"],
    NPG_PALETTE["slate"],
    NPG_PALETTE["sage"],
]

PALETTE = {
    # method / line colours
    "timeaware":      NPG_PALETTE["teal"],
    "vanilla":        NPG_PALETTE["navy"],
    "timeoptimal":    NPG_PALETTE["vermillion"],
    "timeinput":      NPG_PALETTE["peach"],
    "timedependent":  NPG_PALETTE["slate"],
    "schedule":       NPG_PALETTE["sage"],
    "threshold":      NPG_PALETTE["vermillion"],
    # fill / shading colours
    "fillblue":       "#C7D7EB",
    "fillamber":      "#F3D6A6",
    "fillviolet":     "#D9D2E9",
    "fillgreen":      "#D6E6D3",
    "lightgray":      "#D9D9D9",
    # UI / structural colours
    "axspine":        "#444444",
    "neutraledge":    "#555555",
    "grid":           "#999999",
}

# Named aliases for backward compatibility — all derived from PALETTE
TimeawareColor          = PALETTE["timeaware"]
VanillaColor            = PALETTE["vanilla"]
TimeOptimalColor        = PALETTE["timeoptimal"]
TimeInputColor          = PALETTE["timeinput"]
TimeDependentColor      = PALETTE["timedependent"]
ScheduleColor           = PALETTE["schedule"]
ThresholdColor          = PALETTE["threshold"]
FillBlueColor           = PALETTE["fillblue"]
FillAmberColor          = PALETTE["fillamber"]
FillVioletColor         = PALETTE["fillviolet"]
FillGreenColor          = PALETTE["fillgreen"]
LightGrayColor          = PALETTE["lightgray"]
ActiveTimeBarFillColor  = PALETTE["fillblue"]
ActiveTimeBarEdgeColor  = PALETTE["timeaware"]
InstabilityBarFillColor = PALETTE["fillviolet"]
InstabilityBarEdgeColor = PALETTE["timeoptimal"]
AxisSpineColor          = PALETTE["axspine"]
NeutralEdgeColor        = PALETTE["neutraledge"]
GridColor               = PALETTE["grid"]
BaselineColor           = PALETTE["vanilla"]

# ═══════════════════════════════════════════════════════════════════════
#  Line-width constants
# ═══════════════════════════════════════════════════════════════════════
BaselineLineWidth = 3.5
TimeawareLineWidth = 3.5
ReferenceLineWidth = 3.5
ThresholdLineWidth = 3.5
BarEdgeWidth = 0.5
LegendHandleLineWidth = 3.5

# ═══════════════════════════════════════════════════════════════════════
#  Matplotlib global rc overrides
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FontSize,
    "axes.labelsize": AxisLabelSize,
    "axes.titlesize": FontSize,
    "xtick.labelsize": TickSize,
    "ytick.labelsize": TickSize,
    "legend.fontsize": LegendSize,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.handlelength": LegendHandleLength,
    "axes.grid.axis": "y",
    "grid.alpha": 0.18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ═══════════════════════════════════════════════════════════════════════
#  Extended colour list  (for plots needing > 7 series)
# ═══════════════════════════════════════════════════════════════════════
COLORS = [
    *NPG_COLORS,
    PALETTE["lightgray"],
    "#8AA1B1",
    "#D4A373",
    "#8F98B3",
    "#B08968",
    "#A7C4A0",
    "#C38D9E",
    "#A5A58D",
    "#B8C0C8",
    "#C9ADA7",
    "#BFC8AD",
]


# ═══════════════════════════════════════════════════════════════════════
#  Legend → colour / style maps
# ═══════════════════════════════════════════════════════════════════════

def _normalize_style_key(name: str) -> str:
    """Lowercase and remove separators so legend aliases map to one stable key."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


LEGEND_COLOR_MAP = {
    _normalize_style_key(k): v
    for k, v in {
        "Time-aware":                   PALETTE["timeaware"],
        "Time-aware policy":            PALETTE["timeaware"],
        "Staged tr":                    PALETTE["timeaware"],
        "Vanilla":                      PALETTE["vanilla"],
        "Vanilla policy":               PALETTE["vanilla"],
        "Baseline":                     PALETTE["vanilla"],
        "Baseline policy":              PALETTE["vanilla"],
        "Time-optimal":                 PALETTE["timeoptimal"],
        "Time-optimal policy":          PALETTE["timeoptimal"],
        "Time-dependent":               PALETTE["timedependent"],
        "Time-dependent policy":        PALETTE["timedependent"],
        "Time-input":                   PALETTE["timeinput"],
        "Time-input policy":            PALETTE["timeinput"],
        "Scheduled time":               PALETTE["schedule"],
        "Schedule":                     PALETTE["schedule"],
        "Instability threshold":        PALETTE["threshold"],
        "Threshold":                    PALETTE["threshold"],
        "Interpolation":                PALETTE["lightgray"],
        "Vanilla + interpolation":      PALETTE["lightgray"],
        "Vanilla+Interpolation":        PALETTE["lightgray"],
        "Joint Interpolation":          PALETTE["lightgray"],
        "Vanilla + Joint Interpolation": PALETTE["lightgray"],
        "Vanilla+Joint Interpolation":  PALETTE["lightgray"],
        "Constant tr":                  PALETTE["fillviolet"],
    }.items()
}


LEGEND_STYLE_MAP = {
    _normalize_style_key(k): v
    for k, v in {
        "Time-aware":           {"linestyle": "-",  "marker": "o",  "linewidth": TimeawareLineWidth},
        "Time-aware policy":    {"linestyle": "-",  "marker": "o",  "linewidth": TimeawareLineWidth},
        "Vanilla":              {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Vanilla policy":       {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Baseline":             {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Baseline policy":      {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Time-optimal":         {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Time-optimal policy":  {"linestyle": "--", "marker": None, "linewidth": BaselineLineWidth},
        "Scheduled time":       {"linestyle": ":",  "marker": None, "linewidth": ReferenceLineWidth},
        "Instability threshold": {"linestyle": ":", "marker": None, "linewidth": ThresholdLineWidth},
    }.items()
}


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════

def resolve_default_color(name: str, fallback_idx: int = 0) -> str:
    """Map canonical method names to stable paper colours; otherwise fall back."""
    norm = _normalize_style_key(name)
    return LEGEND_COLOR_MAP.get(norm, COLORS[fallback_idx % len(COLORS)])


def style_axis(ax: plt.Axes) -> None:
    """Apply the standard axis styling (light grid, spine colours)."""
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6, color=PALETTE["grid"])
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["axspine"])
        ax.spines[side].set_linewidth(0.6)


class _PowerOffsetFormatter(mpl_ticker.Formatter):
    def __init__(self, exponent: int):
        self.exponent = int(exponent)
        self.scale = 10 ** self.exponent

    def __call__(self, value, pos=None):
        return f"{value / self.scale:g}"

    def get_offset(self):
        return rf"$\times 10^{{{self.exponent}}}$"


def _auto_axis_power(ax: plt.Axes, axis: str, min_exponent: int = 4) -> Optional[int]:
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    max_abs = max(abs(float(lo)), abs(float(hi)))
    if not math.isfinite(max_abs) or max_abs <= 0.0:
        return None
    exponent = int(math.floor(math.log10(max_abs)))
    scaled_max = max_abs / (10 ** exponent)
    if scaled_max < 2.0 and exponent > min_exponent:
        exponent -= 1
    return exponent if exponent >= min_exponent else None


def _tick_label_size(ax: plt.Axes, axis: str):
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in labels:
        size = label.get_size()
        if size:
            return size
    return plt.rcParams.get(f"{axis}tick.labelsize", TickSize)


def apply_axis_power_scale(
    ax: plt.Axes,
    axis: str = "x",
    exponent: Optional[int] = None,
    min_exponent: int = 4,
) -> Optional[int]:
    """Show large linear-axis multipliers as tick-offset text."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    exponent = _auto_axis_power(ax, axis, min_exponent=min_exponent) if exponent is None else exponent
    if exponent is None:
        return None

    formatter = _PowerOffsetFormatter(int(exponent))
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        offset_text = ax.xaxis.get_offset_text()
    else:
        ax.yaxis.set_major_formatter(formatter)
        offset_text = ax.yaxis.get_offset_text()
    offset_text.set_visible(True)
    offset_text.set_fontsize(_tick_label_size(ax, axis))
    return int(exponent)

def _fill_color_for(name: str, line_color: str) -> str:
    """Pick a soft fill colour for the std-dev band of a named series."""
    norm = _normalize_style_key(name)
    if "timeaware" in norm:
        return FillBlueColor
    if "timeoptimal" in norm:
        return FillVioletColor
    if "timeinput" in norm:
        return FillGreenColor
    if "vanilla" in norm or "baseline" in norm:
        return FillAmberColor
    return line_color
