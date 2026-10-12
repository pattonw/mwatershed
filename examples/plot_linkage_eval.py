"""Plot the results of ``evaluate_linkages.py``.

Usage::

    uv run --extra dev --with matplotlib python examples/plot_linkage_eval.py \
        examples/linkage_eval

Writes ``accuracy_ari.png``, ``accuracy_voi.png`` and ``speed.png`` next to
``results.json`` and links them from ``REPORT.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINKAGES = ["abs_max", "sum", "mean", "max", "min"]
# validated categorical palette (slots 1-5, light surface)
COLORS = {
    "abs_max": "#2a78d6",
    "sum": "#eb6834",
    "mean": "#1baf7a",
    "max": "#eda100",
    "min": "#e87ba4",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
FAMILY_TITLES = {
    "array2d": "2D array graph",
    "array3d": "3D array graph",
    "generic": "Generic graph",
}


def load(out: Path) -> list[dict]:
    return json.loads((out / "results.json").read_text())


def aggregate(records: list[dict], metric: str) -> dict:
    groups: dict[tuple, list[float]] = {}
    for r in records:
        groups.setdefault(
            (r["family"], r["dropout"], r["variant"], r["sigma"]), []
        ).append(r[metric])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.title.set_color(TEXT)


def plot_metric(
    records: list[dict], out: Path, metric: str, label: str, filename: str, best: str
):
    agg = aggregate(records, metric)
    families = [f for f in FAMILY_TITLES if any(r["family"] == f for r in records)]
    dropouts = sorted({r["dropout"] for r in records})
    sigmas = sorted({r["sigma"] for r in records})
    fig, axes = plt.subplots(
        len(families),
        len(dropouts),
        figsize=(3.6 * len(dropouts), 3.0 * len(families)),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    fig.patch.set_facecolor(SURFACE)
    for i, family in enumerate(families):
        for j, dropout in enumerate(dropouts):
            ax = axes[i][j]
            style_axis(ax)
            for linkage in LINKAGES:
                for cannot_link in (True, False):
                    variant = linkage if cannot_link else f"{linkage}_nocl"
                    ys = [
                        agg.get((family, dropout, variant, s), np.nan) for s in sigmas
                    ]
                    if all(np.isnan(ys)):
                        continue
                    ax.plot(
                        sigmas,
                        ys,
                        color=COLORS[linkage],
                        linestyle="-" if cannot_link else (0, (4, 2)),
                        linewidth=2,
                        marker="o" if cannot_link else "s",
                        markersize=5,
                        markerfacecolor=SURFACE,
                        markeredgewidth=2,
                        label=variant,
                    )
            ax.set_title(
                f"{FAMILY_TITLES[family]}, dropout {dropout:g}", fontsize=10, loc="left"
            )
            if i == len(families) - 1:
                ax.set_xlabel("edge noise sigma", color=TEXT_SECONDARY, fontsize=9)
            if j == 0:
                ax.set_ylabel(label, color=TEXT_SECONDARY, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"{label} by linkage ({best} is better); solid = cannot-link, dashed = no constraints",
        fontsize=11,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out / filename, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_speed(records: list[dict], out: Path):
    families = [f for f in FAMILY_TITLES if any(r["family"] == f for r in records)]
    variants = [
        v if cl else f"{v}_nocl"
        for v in LINKAGES
        for cl in (True, False)
        if cl or v != "abs_max"  # constraints never change the abs_max result
    ]
    fig, axes = plt.subplots(
        1, len(families), figsize=(3.8 * len(families), 3.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, family in zip(axes[0], families):
        style_axis(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=1)
        ax.grid(False, axis="y")
        example = next(r for r in records if r["family"] == family)
        times = []
        for variant in variants:
            ts = [
                r["time_s"]
                for r in records
                if r["family"] == family and r["variant"] == variant
            ]
            times.append(float(np.median(ts)) * 1e3 if ts else np.nan)
        y = np.arange(len(variants))[::-1]
        for yi, variant, t in zip(y, variants, times):
            linkage = variant.replace("_nocl", "")
            ax.barh(
                yi,
                t,
                height=0.7,
                color=COLORS[linkage],
                hatch="///" if variant.endswith("_nocl") else None,
                edgecolor=SURFACE,
                linewidth=1,
            )
            ax.text(
                t, yi, f" {t:.0f} ms", va="center", ha="left", fontsize=8, color=TEXT
            )
        ax.set_yticks(y)
        ax.set_yticklabels(variants, fontsize=9)
        ax.set_xlim(0, max(times) * 1.25)
        ax.set_title(
            f"{FAMILY_TITLES[family]}\n{example['nodes']} nodes, {example['edges']} edges",
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("median wall time (ms)", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(
        "Speed by linkage (median over all conditions; hatched = no constraints)",
        fontsize=11,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "speed.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def link_figures(out: Path):
    report = out / "REPORT.md"
    if not report.exists():
        return
    text = report.read_text()
    figures = (
        "## Figures\n\n"
        "![ARI by linkage](accuracy_ari.png)\n\n"
        "![VOI by linkage](accuracy_voi.png)\n\n"
        "![Speed by linkage](speed.png)\n\n"
    )
    if "## Figures" in text:
        return
    marker = "\n## "
    idx = text.index(marker)
    report.write_text(text[: idx + 1] + figures + text[idx + 1 :])


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/linkage_eval")
    records = load(out)
    plot_metric(
        records, out, "ari", "adjusted Rand index", "accuracy_ari.png", "higher"
    )
    plot_metric(
        records,
        out,
        "voi",
        "variation of information (bits)",
        "accuracy_voi.png",
        "lower",
    )
    plot_speed(records, out)
    link_figures(out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
