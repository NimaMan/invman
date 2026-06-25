#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# make_blog_figures.py
#
# OBJECTIVE
#   Generate publication-quality figures for the blog post
#   "One Recipe for Ten Inventory Problems: Evolution Strategies for Inventory
#   Control" (nimamanaf.com). All numbers are taken verbatim from the
#   manuscript learning_inventory_control_policies_es.tex and the post; nothing
#   is invented.
#
# FIGURES PRODUCED (saved to the blog's public/static/images/ as PNG @150 dpi
# plus SVG):
#   1. es_inventory_results_overview        -- a verdict-coded summary across the
#      ten problems. Each problem is one row; a colored marker encodes the honest
#      verdict (MATCH a proven optimum / BEAT a heuristic / GAP to an upper bound /
#      RESEARCH result on a faithful env), and the headline number is annotated.
#      Deliberately NOT a raw %-improvement bar chart, because the improvements
#      span +0.82% to +524% over three different kinds of comparator and a raw bar
#      chart would be misleading.
#   2. es_inventory_action_geometry         -- the method schematic: state ->
#      compact policy (normalize -> small backbone -> decoder in the heuristic's
#      coordinate system) -> valid action. Communicates "the action
#      parameterization is part of the policy".
#   3. es_inventory_action_space_trap       -- the multi-echelon action-space trap:
#      same tree / optimizer / horizon, only the decoder's reachable action set
#      changes, swinging from ~14.4% better than the best base-stock to ~239%
#      worse. Uses the real Setting-1 costs (911.4 base-stock, 779.8 direct,
#      3085.7 grid).
#
# ALGORITHM
#   - Set a single restrained, modern matplotlib style (Liberation Sans, light
#     grid, muted accessible palette) shared by all figures so the set looks
#     cohesive.
#   - Each figure is built by its own function and saved as both PNG and SVG.
#
# DATA PROVENANCE
#   Overview verdicts/numbers: post body + paper "What ties it together" table.
#   Action-space-trap costs: paper Table tab:me-results (Setting 1).
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import os

OUT = "/home/nima/code/web/personal/nimamanafcom/public/static/images"
os.makedirs(OUT, exist_ok=True)

# ---- shared style -------------------------------------------------------
PAPER = "#FBF6EC"   # site --paper-high: warm raised-plate stock behind the figures

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#52596A",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E3DCCE",
    "grid.linewidth": 0.9,
    "xtick.color": "#52596A",
    "ytick.color": "#52596A",
    "text.color": "#16202B",
    "axes.labelcolor": "#16202B",
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",
})

# Restrained, colorblind-aware palette, aligned to the nimamanaf.com site tokens
# (ink #16202b, navy #0b4f6c, verdigris #3e7e73, brass #b5611f) so the blog
# figures read as part of the same survey-plate visual system.
INK      = "#16202B"   # site --ink (primary text / spines)
SLATE    = "#52596A"   # site --graphite (secondary text, captions)
BEAT     = "#0B4F6C"   # site --navy     -> beat a heuristic
MATCH    = "#3E7E73"   # site --verdigris-> match a proven optimum
GAP      = "#B5611F"   # site --brass    -> gap to an upper bound (never "beaten")
RESEARCH = "#7A5195"   # purple          -> research result (faithful, not lit-anchored)
WORSE    = "#9A4F38"   # muted terracotta-> the action-space trap "worse" bar
GRIDC    = "#E3DCCE"   # warm paper-tone grid, matches the site stock


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=150)
    plt.close(fig)
    print("wrote", name)


# =========================================================================
# FIGURE 1 : verdict-coded results overview
# =========================================================================
def fig_overview():
    # (label, verdict-key, comparator text, headline annotation)
    # ordered top-to-bottom; grouped by verdict for visual clarity
    # All learned numbers are seed-robust means (mean ± std over >=5 optimizer
    # seeds) sourced from the paper abstract / tables and the seed-robust re-run
    # ledger (docs/benchmarks/SEED_ROBUST_RERUNS_2026_06_06.md).
    rows = [
        ("Dual sourcing",                 "match",    "capped dual-index (heuristic-near-optimal anchor)", "matches optimum  (≤0.11% band)"),
        ("Serial multi-echelon\n(Clark–Scarf)", "match", "proven Clark–Scarf optimum",          "matches optimum  (+0.011%)"),
        ("Lost sales",                    "beat",     "classical heuristics",                    "instance-best 22 / 24"),
        ("Fixed-cost lost sales",         "beat",     "(s,S), (s,nQ), (s,S,q)",                  "instance-best 47 / 48"),
        ("Divergent multi-echelon",       "beat",     "best in-env. base-stock",                 "+14.7% ± 1.6% / +12.0% ± 2.3%  (5 seeds)"),
        ("Perishable",                    "beat",     "best base-stock gate",                    "+1.17% / +0.84%  (5 seeds)"),
        ("General-network backorder",     "beat",     "published constant base-stock",           "+24.3% ± 1.8%  (5 seeds, below PPO)"),
        ("One-warehouse multi-retailer",  "beat",     "tuned base-stock gate",                   "+4.63% / +7.16% / +12.57%  (below PPO)"),
        ("Ameliorating inventory",        "gap",      "order-up-to gate;  LP upper bound",       "beats gate; LP is a bound"),
        ("Production / assembly\nnetwork","research", "env.'s own best heuristic",               "−2.20% to −8.77%  (faithful env)"),
    ]

    cmap = {"match": MATCH, "beat": BEAT, "gap": GAP, "research": RESEARCH}
    marker_map = {"match": "s", "beat": "o", "gap": "D", "research": "^"}

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.2, 7.0))

    ys = np.arange(n)[::-1]  # first row on top

    for y, (label, key, comp, headline) in zip(ys, rows):
        color = cmap[key]
        # subtle connector line from the problem name out to the marker
        ax.plot([0.0, 0.30], [y, y], color="#D8D8D8", lw=1.0, zorder=1)
        ax.scatter([0.30], [y], s=190, marker=marker_map[key], color=color,
                   edgecolor="white", linewidth=1.4, zorder=3)
        # problem name (left)
        ax.text(-0.03, y, label, ha="right", va="center", fontsize=12.5,
                color=INK, fontweight="bold")
        # comparator (small, grey, under the marker area)
        ax.text(0.36, y + 0.17, headline, ha="left", va="center", fontsize=12,
                color=color, fontweight="bold")
        ax.text(0.36, y - 0.20, f"vs. {comp}", ha="left", va="center",
                fontsize=10.0, color=SLATE, style="italic")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.8, n - 0.2)
    ax.axis("off")

    # title block
    fig.text(0.065, 0.965, "Ten inventory problems, one gradient-free recipe",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.915,
             "CMA-ES over compact, interpretable policies — read by honest verdict, not a single % scale; "
             "every learned number is a seed-robust mean over ≥5 optimizer seeds",
             fontsize=11.5, color=SLATE, va="top")

    # legend explaining the verdict types
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MATCH,
               markersize=12, label="MATCH a proven optimum (never “beat”)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BEAT,
               markersize=12, label="BEAT a heuristic comparator"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=GAP,
               markersize=11, label="GAP to an upper bound (reported, not beaten)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=RESEARCH,
               markersize=12, label="RESEARCH result on a faithful env"),
    ]
    leg = ax.legend(handles=legend_elems, loc="lower center",
                    bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
                    fontsize=10.8, handletextpad=0.5, columnspacing=1.6)

    fig.subplots_adjust(left=0.21, right=0.985, top=0.86, bottom=0.10)
    save(fig, "es_inventory_results_overview")


# =========================================================================
# FIGURE 2 : action-geometry method schematic
# =========================================================================
def fig_action_geometry():
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, tcolor=INK, fs=11.5, bold=True,
            rounding=0.10):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={rounding*min(w,h)}",
                           linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tcolor, fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=3)
        return (x, y, w, h)

    def arrow(x0, y0, x1, y1, color=SLATE):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                            mutation_scale=17, lw=1.8, color=color, zorder=1,
                            shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    yb = 24
    h = 13
    # box fills are faint tints of each element's accent over the warm paper stock,
    # so the schematic reads as one cohesive plate rather than white cards on cream
    box(2, yb, 17, h, "state\n$S_t$", "#E5EDF1", BEAT, INK, fs=12.5)
    # normalization
    box(26, yb, 17, h, "normalize\n$x=S_t/\\kappa$", "#F6F1E6", SLATE, INK, fs=11)
    # backbone
    box(50, yb, 20, h, "compact\nbackbone", "#F6F1E6", SLATE, INK, fs=11.5)
    # decoder (the emphasized element)
    box(77, yb, 22, h, "decoder in the\nheuristic's\ncoordinate system",
        "#EFE9F4", RESEARCH, INK, fs=10.5)
    # output action
    box(106, yb, 9, h, "valid\naction\n$a_t$", "#E6F0EC", MATCH, INK, fs=11)

    # backbone variants caption (placed inside box footprint via the box text instead)
    ax.text(60, yb + 2.6, "linear · tiny NN · soft tree", ha="center",
            va="center", fontsize=8.8, color=SLATE, style="italic")

    # arrows
    arrow(19, yb + h / 2, 26, yb + h / 2)
    arrow(43, yb + h / 2, 50, yb + h / 2)
    arrow(70, yb + h / 2, 77, yb + h / 2)
    arrow(99, yb + h / 2, 106, yb + h / 2)

    # dashed "policy" box around normalize+backbone+decoder
    pol = FancyBboxPatch((24, yb - 5.0), 77, h + 10.0,
                         boxstyle="round,pad=0,rounding_size=1.6",
                         linewidth=1.5, edgecolor=BEAT, facecolor="none",
                         linestyle=(0, (6, 4)), zorder=0)
    ax.add_patch(pol)
    ax.text(62.5, yb - 7.2,
            "policy $\\pi_\\theta$  —  all of this is learned by CMA-ES",
            ha="center", va="top", fontsize=11, color=BEAT, fontweight="bold")

    # input/output mini-labels
    ax.text(10.5, yb + h + 1.4, "input", ha="center", va="bottom",
            fontsize=9.5, color=MATCH)
    ax.text(110.5, yb + h + 1.4, "output", ha="center", va="bottom",
            fontsize=9.5, color=MATCH)

    # the load-bearing examples of "the heuristic's coordinate system"
    ax.text(58, yb - 13.0,
            "the decoder's coordinate system, by problem:   ordinal “one more unit” (lost sales)   ·   "
            "capped-dual-index (dual sourcing)   ·   echelon order-up-to (multi-echelon)",
            ha="center", va="top", fontsize=8.8, color=RESEARCH)

    fig.text(0.045, 0.965, "The action parameterization is part of the policy",
             fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.910,
             "A tiny policy reaches the good operating region when its decoder is shaped "
             "like the relevant heuristic — not by adding network width.",
             fontsize=11.5, color=SLATE, va="top")

    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.04)
    save(fig, "es_inventory_action_geometry")


# =========================================================================
# FIGURE 3 : the multi-echelon action-space trap
# =========================================================================
def fig_action_space_trap():
    # Setting-1 costs from paper Table tab:me-results (lower is better).
    # Seed-robust values: gate and direct-level tree are 5-seed means.
    base = 910.3    # best in-env. constant base-stock (gate seed-mean 910.3 ± 0.5)
    direct = 776.2  # direct-level soft tree (ours), 5-seed mean (± 14.3)
    grid = 3085.7   # grid-restricted tree (y^w <= 100)

    # percentage vs base-stock benchmark
    pct_direct = (direct - base) / base * 100.0   # -14.7
    pct_grid = (grid - base) / base * 100.0        # +239

    labels = ["Best constant\nbase-stock\n(benchmark)",
              "Direct-level tree\n(ours, 5-seed mean)",
              "Grid-restricted tree\n($y^w \\leq 100$)"]
    vals = [base, direct, grid]
    colors = [SLATE, MATCH, WORSE]

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    x = np.arange(3)
    bars = ax.bar(x, vals, width=0.62, color=colors, edgecolor=PAPER,
                  linewidth=1.4, zorder=3)

    # benchmark reference line (caption in the open gap between the direct and
    # grid bars, above the line, so it never sits on a bar or a value label)
    ax.axhline(base, color=SLATE, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.text(1.34, base + 28, "constant base-stock benchmark", ha="center",
            va="bottom", fontsize=9.5, color=SLATE, style="italic")

    # value labels + % annotations (seed-robust means).
    # The benchmark/direct bars are close in height, so the direct-tree caption
    # sits *inside* its bar to avoid colliding with the dashed benchmark line.
    ax.text(0, base + 95, f"{base:.0f}", ha="center", va="bottom",
            fontsize=12.5, color=INK, fontweight="bold")
    ax.text(1, direct - 70, f"{direct:.0f}\n({pct_direct:+.1f}%)\n5-seed mean",
            ha="center", va="top", fontsize=12, color=PAPER, fontweight="bold")
    ax.text(2, grid + 55, f"{grid:.0f}\n({pct_grid:+.0f}%)", ha="center",
            va="bottom", fontsize=12.5, color=WORSE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Long-run average cost  (lower is better)")
    ax.set_ylim(0, grid * 1.16)

    fig.text(0.045, 0.975, "The action-space trap (divergent multi-echelon)",
             fontsize=16.5, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.928,
             "Same soft tree, same optimizer, same horizon — only the decoder's reachable action set changes.",
             fontsize=11.2, color=SLATE, va="top")

    # arc-style annotation tying the two learned bars to the swing
    ax.annotate("", xy=(2, grid * 1.02), xytext=(1, direct + 360),
                arrowprops=dict(arrowstyle="-|>", color=WORSE, lw=1.6,
                                connectionstyle="arc3,rad=-0.25"), zorder=4)
    ax.text(1.5, grid * 1.07,
            "the grid physically\ncannot reach the\noperating region\n(needs $y^w\\approx300$–$525$)",
            ha="center", va="center", fontsize=9.8, color=WORSE)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.12)
    save(fig, "es_inventory_action_space_trap")


if __name__ == "__main__":
    fig_overview()
    fig_action_geometry()
    fig_action_space_trap()
    print("done")
