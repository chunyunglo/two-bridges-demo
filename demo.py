"""Run the whole demonstration on a synthetic world.

    python demo.py                        about two minutes on a laptop
    python demo.py --quick                fewer bootstrap samples and a shorter sample-size sweep
    python demo.py --seed 1 --out other   another random world, written to other/ instead

Writes results/*.csv, figures/demo.png and figures/method.png. Every number in the README comes from these tables.
"""
import argparse
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from twobridges import TwoBridges, WorldConfig, evaluate, make_world, spearman_cols

SEED = 20261112
PARTICIPANTS = 50                 # participants per cue in the main analysis
SWEEP = [5, 10, 20, 50, 100]
LADDER = [("rc_baseline", "baseline Rc"), ("rc_u", "all anchors, mean"), ("rc_uv", "+ response role"),
          ("rc_star", "+ ceiling correction"), ("f_star", "first-order route alone"), ("a_star", "A*: both routes")]
COLS = [c for c, _ in LADDER] + ["a_raw"]
HERE = Path(__file__).resolve().parent
GREY, BLUE, ORANGE, GREEN = "#8a8a8a", "#3b6ea8", "#d9731a", "#4d9a6a"


def run(quick: bool = False, seed: int = SEED, out: Path = HERE):
    t0 = time.time()
    n_boot = 200 if quick else 2000
    RESULTS, FIGURES = Path(out) / "results", Path(out) / "figures"
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    world = make_world(WorldConfig(participants=max(SWEEP)), seed=seed)
    dictionary = world.lexicon[["English", "Chinese"]]
    print(f"synthetic world: {len(world.en_words)} + {len(world.zh_words)} words, {len(dictionary)} dictionary pairs, "
          f"{len(world.ratings)} rated [{time.time() - t0:.0f}s]", flush=True)

    model = TwoBridges(world.associations("en", PARTICIPANTS), world.associations("zh", PARTICIPANTS), dictionary, seed=seed)
    scores = model.score().merge(world.lexicon, on=["English", "Chinese"])
    rated = world.ratings.merge(scores.drop(columns=["kind", "truth"]), on=["English", "Chinese"])
    probe = world.probe_pairs(seed=seed)
    probe = probe.merge(model.score(probe), on=["English", "Chinese"])
    print(f"scored {len(scores)} dictionary pairs and {len(probe)} other pairs [{time.time() - t0:.0f}s]", flush=True)

    # 1. Does each score predict the simulated ratings, and the truth behind them?
    sets = [("rated dictionary pairs, test half", rated[rated.split == "test"], "rating"),
            ("rated dictionary pairs, development half", rated[rated.split == "dev"], "rating"),
            ("all dictionary pairs", scores, "truth"),
            ("pairs that are not dictionary entries", probe, "rating"),
            ("pairs that are not dictionary entries", probe, "truth")]
    perf = pd.concat([evaluate(frame, COLS, target, n_boot=n_boot, seed=seed).assign(pairs=name)
                      for name, frame, target in sets], ignore_index=True)
    perf = perf[["pairs", "target", "score", "n", "rho", "lo", "hi", "diff", "diff_lo", "diff_hi"]]
    perf.round(4).to_csv(RESULTS / "performance.csv", index=False)

    # 2. How many pairs does each score cover? The baseline needs both words to have been cues.
    test = rated[rated.split == "test"]
    cover = pd.DataFrame([{"score": c, "dictionary_pairs": int(scores[c].notna().sum()), "rated_pairs": int(rated[c].notna().sum()),
                           "rated_test_pairs": int(test[c].notna().sum()),
                           "rho_on_own_test_pairs": spearman_cols(test.dropna(subset=[c])[[c]].to_numpy(float),
                                                                  test.dropna(subset=[c]).rating.to_numpy(float))[0]}
                          for c in COLS])
    cover.round(4).to_csv(RESULTS / "coverage.csv", index=False)

    # 3. How much association data does recovery need?
    rows = []
    for n in ([10, PARTICIPANTS] if quick else SWEEP):
        m = model if n == PARTICIPANTS else TwoBridges(world.associations("en", n), world.associations("zh", n), dictionary, seed=seed)
        s = m.score().merge(world.lexicon, on=["English", "Chinese"])
        e = evaluate(s, COLS, "truth", n_boot=n_boot, seed=seed).assign(participants=n)
        rows.append(e)
        print(f"  {n:>3} participants per cue: baseline {e.rho[0]:.3f}, A* {e.rho[COLS.index('a_star')]:.3f} "
              f"(n = {e.n[0]}) [{time.time() - t0:.0f}s]", flush=True)
    sweep = pd.concat(rows, ignore_index=True)
    sweep = sweep[["participants", "score", "n", "rho", "lo", "hi", "diff", "diff_lo", "diff_hi"]]
    sweep.round(4).to_csv(RESULTS / "sample_size.csv", index=False)

    scores.drop(columns=["rc_uv", "a_raw"]).round(4).to_csv(RESULTS / "pair_scores.csv", index=False)
    figure(rated[rated.split == "test"], perf, sweep, FIGURES)
    schematic(FIGURES)

    show = perf[perf.score.isin(["rc_baseline", "a_star"])].copy()
    show["rho [95% CI]"] = [f"{r:.3f} [{a:.3f}, {b:.3f}]" for r, a, b in zip(show.rho, show.lo, show.hi)]
    show["A* - baseline"] = [f"{d:+.3f} [{a:+.3f}, {b:+.3f}]" if s == "a_star" else "" for s, d, a, b in
                             zip(show.score, show["diff"], show.diff_lo, show.diff_hi)]
    print(show[["pairs", "target", "score", "n", "rho [95% CI]", "A* - baseline"]].to_string(index=False))
    print(cover.round(3).to_string(index=False))
    print(f"done [{time.time() - t0:.0f}s]")


STYLE = {
    "font.family": ["DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "savefig.dpi": 300,
}


def schematic(folder: Path):
    """The method as a diagram (no data): three inputs, two routes and a ceiling, one score."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    mpl.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, title, body="", edge="#444444", face="white", bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09", linewidth=0.9,
                                    edgecolor=edge, facecolor=face))
        if body:
            ax.text(x + 0.14, y + h - 0.13, title, ha="left", va="top", fontsize=7.5, fontweight="bold", color=edge)
            ax.text(x + 0.14, y + h - 0.42, body, ha="left", va="top", fontsize=6.8, linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7.2, linespacing=1.35,
                    fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                                     color="#555555", shrinkA=0, shrinkB=0))

    ax.add_patch(FancyBboxPatch((0.08, 0.12), 2.3, 3.36, boxstyle="round,pad=0,rounding_size=0.12", linewidth=0,
                                facecolor="#f1f3f5"))
    box(0.25, 2.48, 1.96, 0.82, "associations, language A\ncue → three responses", bold=False)
    box(0.25, 1.39, 1.96, 0.82, "bilingual dictionary", bold=False)
    box(0.25, 0.30, 1.96, 0.82, "associations, language B\ncue → three responses", bold=False)

    box(3.15, 2.48, 4.25, 1.0, "Second-order route: relational consistency Rc",
        "Do the two words relate to the translated anchor words\nin the same way? The two spaces are never aligned.", edge=BLUE)
    box(3.15, 1.30, 4.25, 1.0, "First-order route: fidelity F",
        "Do they bring the same things to mind, once one word's\nassociations are translated through the dictionary?", edge=GREEN)
    box(3.15, 0.12, 4.25, 1.0, "Within-language ceilings",
        "How well does a word agree with itself across two\nhalves of the participants? Divides the cue-role scores.", edge=GREY)
    ax.plot([2.38, 2.76], [1.80, 1.80], color="#555555", lw=0.8, solid_capstyle="butt")     # every route uses all three inputs
    ax.plot([2.76, 2.76], [0.62, 2.98], color="#555555", lw=0.8, solid_capstyle="projecting")
    for y in (2.98, 1.80, 0.62):
        arrow(2.76, y, 3.13, y)
        arrow(7.40, y, 8.13, 1.80 + (y - 1.80) * 0.28)
    ax.add_patch(FancyBboxPatch((8.15, 1.18), 1.75, 1.24, boxstyle="round,pad=0,rounding_size=0.09", linewidth=1.1,
                                edgecolor=ORANGE, facecolor="#fdf1e6"))
    ax.text(9.025, 2.02, "A*", ha="center", va="center", fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(9.025, 1.55, "mean of z-scores,\ncue and response roles", ha="center", va="center", fontsize=6.8, linespacing=1.35)
    fig.savefig(folder / "method.png", dpi=300)
    plt.close(fig)


def figure(test: pd.DataFrame, perf: pd.DataFrame, sweep: pd.DataFrame, folder: Path):
    mpl.rcParams.update(STYLE)
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.02], hspace=0.50, wspace=0.75, left=0.075, right=0.985, top=0.94, bottom=0.085)

    def tag(ax, letter, title, shift=0.0):
        ax.text(-shift, 1.045, title, transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left")
        ax.text(-shift - 0.02, 1.045, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

    # a, b: each score against the simulated ratings (test half)
    t = test.dropna(subset=["rc_baseline", "a_star"])
    rho = perf[(perf.pairs == "rated dictionary pairs, test half")].set_index("score").rho
    kinds = [("first", BLUE, "first translation"), ("second", GREEN, "second translation"), ("loose", ORANGE, "loose entry")]
    for k, (col, label, letter) in enumerate((("rc_baseline", "baseline Rc", "a"), ("a_star", "A* (z units)", "b"))):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        for kind, colour, name in kinds:
            g = t[t.kind == kind]
            ax.scatter(g[col], g.rating, s=7, color=colour, alpha=0.65, linewidths=0, label=name, rasterized=True)
        ax.set_xlabel(label)
        ax.set_ylabel("simulated rating (1 to 7)")
        ax.set_ylim(0.8, 7.2)
        ax.text(0.03, 0.97, f"Spearman ρ = {rho[col]:.2f}\n{len(t)} held-out pairs", transform=ax.transAxes, va="top", ha="left")
        if k == 1:
            ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2, markerscale=1.4)
        tag(ax, letter, "Baseline against the ratings" if k == 0 else "Both routes against the ratings")

    # c: the ladder from the baseline to A*
    ax = fig.add_subplot(gs[1, 0:2])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.115, box.y0, box.width - 0.115, box.height])
    p = perf[perf.pairs == "rated dictionary pairs, test half"].set_index("score")
    ypos = np.arange(len(LADDER))[::-1]
    colours = [GREY, BLUE, BLUE, BLUE, GREEN, ORANGE]
    for y, (col, _), colour in zip(ypos, LADDER, colours):
        ax.plot([p.lo[col], p.hi[col]], [y, y], color=colour, lw=1.2)
        ax.plot(p.rho[col], y, "o", color=colour, ms=4.5)
        ax.text(p.hi[col] + 0.004, y, f"{p.rho[col]:.2f}", va="center", ha="left", fontsize=6.5)
    ax.axvline(p.rho["rc_baseline"], color=GREY, lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(ypos, [name for _, name in LADDER])
    ax.set_ylim(-0.6, len(LADDER) - 0.4)
    ax.set_xlim(p.lo.min() - 0.01, p.hi.max() + 0.03)
    ax.set_xlabel("Spearman ρ with the ratings (95% interval)")
    tag(ax, "c", "What each step adds", shift=0.115 / ax.get_position().width)

    # d: how much data recovery needs
    ax = fig.add_subplot(gs[1, 2:4])
    for col, colour, name in (("rc_baseline", GREY, "baseline Rc"), ("a_star", ORANGE, "A*: both routes")):
        s = sweep[sweep.score == col]
        ax.fill_between(s.participants, s.lo, s.hi, color=colour, alpha=0.18, linewidth=0)
        ax.plot(s.participants, s.rho, "-o", color=colour, ms=3.5, label=name)
    ax.set_xscale("log")
    ax.set_xticks(sorted(sweep.participants.unique()), [str(v) for v in sorted(sweep.participants.unique())])
    ax.minorticks_off()
    ax.set_xlabel("participants per cue")
    ax.set_ylabel("Spearman ρ with the true equivalence")
    ax.legend(loc="lower right")
    tag(ax, "d", "Recovering the truth from less data")

    fig.savefig(folder / "demo.png", dpi=300)
    fig.savefig(folder / "demo.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="fewer bootstrap samples, shorter sweep")
    parser.add_argument("--seed", type=int, default=SEED, help="seed of the synthetic world")
    parser.add_argument("--out", type=Path, default=HERE, help="folder that receives results/ and figures/")
    args = parser.parse_args()
    run(args.quick, args.seed, args.out)
