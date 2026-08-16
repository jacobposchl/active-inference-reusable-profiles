#!/usr/bin/env python3
"""
Simple, self-contained figure renderer for fast iteration.

Reads a model-recovery run directory (real or synthetic) and draws both paper
figures. No pymdp, no src imports -- pure pandas/numpy/matplotlib -- so it renders
in seconds against `make_synthetic_run.py` output.

    python figures.py                      # uses results/model_recovery/run_synth
    python figures.py --results-dir <dir>  # any real run

Everything we iterate on lives in the STYLE block below.
"""
import os
import ast
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# ============================ STYLE (tweak here) ============================
MODEL = {"M1": "#5580b3", "M2": "#e0925f", "M3": "#5aa588"}   # muted blue / orange / teal
MODEL_LS = {"M1": "-", "M2": "--", "M3": ":"}                 # print/CVD fallback
CTX = {"volatile": "#cb6f60", "stable": "#5f8fbb"}           # soft red / blue (profiles)
INK = INK2 = INK3 = "#000000"                                 # all chrome/text is black
GRID = "#c9c9c9"                                              # grey horizontal gridlines

# Register bundled Roboto (if present in ./fonts) so the figures use it everywhere.
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
if os.path.isdir(_FONT_DIR):
    for _fp in glob.glob(os.path.join(_FONT_DIR, "*.ttf")):
        try:
            fm.fontManager.addfont(_fp)
        except Exception:
            pass
_HAVE_ROBOTO = any("Roboto" in f.name for f in fm.fontManager.ttflist)
# Roboto first, DejaVu as glyph-fallback (Roboto lacks e.g. the -> arrow).
FONT_FAMILY = (["Roboto", "DejaVu Sans"] if _HAVE_ROBOTO else ["DejaVu Sans"])


def apply_style():
    if not _HAVE_ROBOTO:
        print("[figures] Roboto not found; using default sans-serif. "
              "Drop Roboto-*.ttf into ./fonts to enable it.")
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 9,
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 14, "axes.labelweight": "normal", "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 10.5, "legend.frameon": True, "legend.loc": "upper left",
        "legend.facecolor": "#ececec", "legend.edgecolor": "#9a9a9a", "legend.framealpha": 1.0,
        "legend.handlelength": 1.6, "legend.handleheight": 1.4, "legend.handletextpad": 0.6,
        "axes.edgecolor": INK, "axes.linewidth": 1.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, out_path):
    """Write the figure to out_path and, alongside it, a vector .pdf for the paper."""
    fig.savefig(out_path)
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        fig.savefig(root + ".pdf")
    plt.close(fig)


def clean(ax, grid_axis="y", frame=True):
    # frame=True keeps the full boundary box (all four spines); the title is
    # drawn above the axes box, so it stays outside the box.
    if not frame:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def swatch_legend(ax, pairs, **kw):
    """Legend with chunky color-rectangle swatches (matching the bar panels)."""
    ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l) for l, c in pairs], **kw)


def panel_tag(ax, letter):
    ax.text(-0.065, 1.04, letter, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="bottom", ha="right", color=INK)


# ============================ DATA ============================
def _efe_separation(efe):
    """dG = G(pi_2nd-best) - G(pi_best), with "best" the lowest-EFE policy.

    Non-negative by construction; nan when a trial evaluated fewer than two
    policies. Assumes the logged vector is EFE (lower is better) rather than
    negative EFE -- flip the sort if the upstream convention changes.
    """
    a = np.sort(np.asarray(efe, dtype=float))
    return float(a[1] - a[0]) if a.size >= 2 else float("nan")


def load_runs(run_dir, model, generator="M3", fold=0):
    """List of per-run DataFrames for `model` fitted to `generator`'s data.

    Each CSV holds the same trial sequence replayed once per CV fold, so the file
    is K x the run length and every reversal appears K times. Those replays are
    duplicates of one run, not independent samples, so we keep a single fold and
    pool across the genuinely independent `run_*` files instead.
    """
    pat = os.path.join(run_dir, "trial_level", f"gen_{generator}", f"model_{model}", "run_*.csv")
    dfs = []
    for f in sorted(glob.glob(pat)):
        df = pd.read_csv(f)
        # Only the real pipeline replicates the trial sequence per fold; there `t`
        # repeats K times. The synthetic run writes 400 unique trials with `fold`
        # as a per-trial label (t % 5), so filtering it would gut the sequence.
        if fold is not None and "fold" in df.columns and df["t"].duplicated().any():
            # reset now: aligned()/around_reversal() index positionally off a clean RangeIndex
            df = df[df["fold"] == fold].reset_index(drop=True)
        w = df["belief_context"].apply(ast.literal_eval)
        df["w0"] = w.apply(lambda x: float(x[0]))   # volatile-profile weight = q(volatile)
        df["w1"] = w.apply(lambda x: float(x[1]))   # stable-profile weight   = q(stable)
        # EFE separation: how far the runner-up policy sits behind the best one,
        #   dG_t = G_t(pi_2nd-best) - G_t(pi_best),  "best" = lowest EFE.
        # Since q(pi) = sigma(-gamma * G), this is the gamma-independent driver of
        # policy-posterior entropy: a small dG leaves q(pi) flat however large
        # gamma is, so M1 can track uncertainty at fixed gamma through dG alone.
        # Accepts either a pre-summarised `efe_sep` or a per-trial `efe` vector.
        if "efe_sep" not in df.columns and "efe" in df.columns:
            e = df["efe"].apply(ast.literal_eval)
            df["efe_sep"] = e.apply(_efe_separation)
        dfs.append(df.reset_index(drop=True))
    return dfs


def aligned(dfs, col, direction, pre=10, post=40):
    """Mean and SD of `col` in a [-pre, +post] window around context reversals of
    a given direction ('v2s' volatile->stable, or 's2v').

    Returns (mean, sd, xs, n_segments); sd is across reversal segments.
    """
    segs = []
    for df in dfs:
        for r in df.index[df["is_reversal"] == 1]:
            if r - 1 < 0 or r + post > len(df) or r - pre < 0:
                continue
            prev, cur = df["true_context"].iloc[r - 1], df["true_context"].iloc[r]
            ok = (direction == "v2s" and prev == "volatile" and cur == "stable") or \
                 (direction == "s2v" and prev == "stable" and cur == "volatile")
            if ok:
                segs.append(df[col].iloc[r - pre:r + post].to_numpy())
    if not segs:
        return None, None, None, 0
    a = np.vstack(segs)
    return a.mean(0), a.std(0), np.arange(-pre, post), len(segs)


def around_reversal(dfs, col, pre=20, post=20):
    """Mean of `col` in a [-pre, +post] window around every context reversal."""
    segs = []
    for df in dfs:
        for r in df.index[df["is_reversal"] == 1]:
            if r - pre < 0 or r + post > len(df):
                continue
            segs.append(df[col].iloc[r - pre:r + post].to_numpy())
    if not segs:
        return None, np.arange(-pre, post)
    a = np.vstack(segs)
    return a.mean(0), np.arange(-pre, post)


def binned(dfs, xcol, ycol, nbins=8, xmax=None):
    """Mean of ycol within equal-width bins of xcol, pooled across runs.

    Returns (centres, mean, sem); sem is nan where a bin holds fewer than two
    non-nan trials.
    """
    d = pd.concat(dfs, ignore_index=True)
    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    hi = float(np.nanmax(x)) if xmax is None else xmax
    edges = np.linspace(0.0, hi, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ym = np.full(nbins, np.nan)
    ys = np.full(nbins, np.nan)
    for i in range(nbins):
        lo, up = edges[i], edges[i + 1]
        m = (x >= lo) & (x <= up) if i == nbins - 1 else (x >= lo) & (x < up)
        if m.any():
            yi = y[m]
            ym[i] = np.nanmean(yi)
            n = int(np.count_nonzero(~np.isnan(yi)))
            if n > 1:
                ys[i] = float(np.nanstd(yi, ddof=1) / np.sqrt(n))
    return ctr, ym, ys


def by_context(dfs, fn):
    """Apply fn(df_subset) for volatile/stable, pooled across runs -> (vol, stab)."""
    allrows = pd.concat(dfs, ignore_index=True)
    return fn(allrows[allrows.true_context == "volatile"]), fn(allrows[allrows.true_context == "stable"])


# ============================ FIGURE 2: AIC ============================
def fig_aic(run_dir, out_path):
    m = pd.read_csv(os.path.join(run_dir, "confusion", "aic_mean.csv"), index_col=0)
    se = pd.read_csv(os.path.join(run_dir, "confusion", "aic_se.csv"), index_col=0)
    gens = models = ["M1", "M2", "M3"]
    vals = m.loc[gens, models].values
    err = se.loc[gens, models].values

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(gens))
    w = 0.26
    for i, mdl in enumerate(models):
        ax.bar(x + (i - 1) * w, vals[:, i], w, label=mdl,
               color=MODEL[mdl], edgecolor="white", linewidth=0.7,
               yerr=err[:, i],
               error_kw=dict(elinewidth=0.8, capsize=2.5, capthick=0.8, ecolor=INK2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} data" for g in gens], fontsize=13)
    ax.set_ylabel(r"AIC$_{\mathrm{CV}}$", fontsize=15)
    ax.set_xlabel("Data generator", fontsize=15)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.2, handlelength=1.2)
    clean(ax, "y")
    ax.set_ylim(0, vals.max() * 1.12)
    fig.tight_layout()
    save_figure(fig, out_path)


# ============================ FIGURE 1: MECHANISTIC ============================
def fig_mechanistic(run_dir, out_path):
    runs = {m: load_runs(run_dir, m) for m in ("M1", "M2", "M3")}
    pe_cols = ("policy_entropy", "better_arm_entropy")
    missing_pe = [c for c in pe_cols if c not in runs["M1"][0].columns]
    has_pe = not missing_pe
    if missing_pe:
        # Runs written before the entropy columns existed would otherwise blank out
        # panels D, G and H with no indication that anything was dropped.
        warnings.warn(
            f"{run_dir}: trial-level CSVs are missing {', '.join(missing_pe)}; "
            "panels D, G and H (policy entropy) will render blank. Re-run the "
            "experiment to regenerate the run with these columns.",
            stacklevel=2,
        )

    has_efe = "efe_sep" in runs["M1"][0].columns
    if not has_efe:
        warnings.warn(
            f"{run_dir}: trial-level CSVs carry no `efe_sep` (or `efe`) column; "
            "panel H (policy entropy vs. EFE separation) will render blank. The EFE "
            "vector is computed in recovery_helpers._generate_trial_level_predictions "
            "but currently discarded -- log it and re-run to fill this panel.",
            stacklevel=2,
        )

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    (axA, axB, axC, axD), (axE, axF, axG, axH) = axes

    # --- A / B: profile recruitment around reversals (M3) ---
    for ax, direction, title, tag in [
        (axA, "v2s", "Profile recruitment: volatile → stable", "A"),
        (axB, "s2v", "Profile recruitment: stable → volatile", "B"),
    ]:
        w0, w0sd, xs, n = aligned(runs["M3"], "w0", direction)
        w1, w1sd, _, _ = aligned(runs["M3"], "w1", direction)
        if w0 is not None:
            for mu, sd, key, lab in [(w0, w0sd, "volatile", "$w_0$ volatile profile"),
                                     (w1, w1sd, "stable", "$w_1$ stable profile")]:
                ax.plot(xs, mu, color=CTX[key], lw=2, label=lab)
                ax.fill_between(xs, mu - sd, mu + sd, color=CTX[key], alpha=0.15, linewidth=0)
        ax.axvline(0, color=INK3, lw=1, ls="--")
        ax.set_title(f"{title}\n({n} reversals)")
        ax.set_xlabel("Trials relative to reversal")
        ax.set_ylabel("Profile weight")
        ax.set_ylim(0, 1)
        swatch_legend(ax, [("$w_0$ volatile profile", CTX["volatile"]),
                           ("$w_1$ stable profile", CTX["stable"])])
        clean(ax, "y")
        panel_tag(ax, tag)

    # --- C: effective precision around reversals, per model ---
    for m in ("M1", "M2", "M3"):
        g, xs = around_reversal(runs[m], "gamma")
        axC.plot(xs, g, color=MODEL[m], ls=MODEL_LS[m], lw=2, label=m)
    axC.axvline(0, color=INK3, lw=1, ls="--")
    axC.set_title("Effective precision around reversals")
    axC.set_xlabel("Trials relative to reversal")
    axC.set_ylabel(r"$\gamma_t^{\mathrm{eff}}$")
    axC.legend()
    clean(axC, "y")
    panel_tag(axC, "C")

    # --- D: policy-posterior entropy around reversals, per model ---
    if has_pe:
        for m in ("M1", "M2", "M3"):
            h, xs = around_reversal(runs[m], "policy_entropy")
            if h is not None:
                axD.plot(xs, h, color=MODEL[m], ls=MODEL_LS[m], lw=2, label=m)
        axD.axvline(0, color=INK3, lw=1, ls="--")
        axD.set_title("Policy entropy around reversals")
        axD.set_xlabel("Trials relative to reversal")
        axD.set_ylabel(r"H[$q(\pi)$]  (nats)")
        axD.legend()
        clean(axD, "y")
    else:
        axD.axis("off")
    panel_tag(axD, "D")

    # --- E: mean precision by true context, per model ---
    # by_context() splits on `true_context`, not the agent's posterior, so this
    # and panel F are conditioned on the generating regime.
    xm = np.arange(3)
    w = 0.36
    for j, ctx in enumerate(("volatile", "stable")):
        means = [by_context(runs[m], lambda d: d.gamma.mean())[j] for m in ("M1", "M2", "M3")]
        errs = [by_context(runs[m], lambda d: d.gamma.sem())[j] for m in ("M1", "M2", "M3")]
        axE.bar(xm + (j - 0.5) * w, means, w, yerr=errs, label=f"{ctx} context",
                color=CTX[ctx], edgecolor="white", linewidth=0.7,
                error_kw=dict(elinewidth=0.8, capsize=2.5, capthick=0.8, ecolor=INK2))
    axE.set_xticks(xm)
    axE.set_xticklabels(("M1", "M2", "M3"))
    axE.set_title("Precision by true context")
    axE.set_ylabel(r"Mean $\gamma$")
    axE.set_xlabel("Model")
    axE.legend()
    clean(axE, "y")
    panel_tag(axE, "E")

    # --- F: context-conditional hint-seeking (M3) ---
    hint = lambda d: (d.predicted_action == "act_hint").mean()
    hint_sem = lambda d: (d.predicted_action == "act_hint").astype(float).sem()
    vol, stab = by_context(runs["M3"], hint)
    vse, sse = by_context(runs["M3"], hint_sem)
    axF.bar([0, 1], [vol, stab], 1.0, yerr=[vse, sse],
            color=[CTX["volatile"], CTX["stable"]], edgecolor="white", linewidth=0.7,
            error_kw=dict(elinewidth=0.8, capsize=2.5, capthick=0.8, ecolor=INK2))
    axF.set_xticks([0, 1])
    axF.set_xticklabels(("volatile", "stable"), fontsize=12)
    axF.set_xlim(-0.6, 1.6)
    axF.set_title("Context-conditional hint-seeking (M3)")
    axF.set_ylabel("Hint-seeking rate")
    axF.set_xlabel("True context")
    axF.set_ylim(0, 1)
    clean(axF, "y")
    panel_tag(axF, "F")

    # --- G: policy entropy vs better-arm uncertainty, per model ---
    if has_pe:
        for m in ("M1", "M2", "M3"):
            cx, cy, _ = binned(runs[m], "better_arm_entropy", "policy_entropy",
                               nbins=8, xmax=float(np.log(2)))
            axG.plot(cx, cy, color=MODEL[m], ls=MODEL_LS[m], lw=2, marker="o", ms=4, label=m)
        axG.set_title("Policy entropy vs. arm uncertainty")
        axG.set_xlabel(r"H[$q$(better arm)]  (nats)")
        axG.set_ylabel(r"H[$q(\pi)$]  (nats)")
        axG.legend()
        clean(axG, "y")
    else:
        axG.axis("off")
    panel_tag(axG, "G")

    # --- H: policy entropy vs EFE separation, per model ---
    # dG is the gamma-independent driver of H[q(pi)]: when the top two policies are
    # nearly tied the posterior stays flat whatever gamma does, so M1's response to
    # uncertainty (panel G) has to arrive through this axis.
    if has_pe and has_efe:
        # Shared upper edge across models so the bins line up and the curves are
        # comparable; the 99th percentile keeps a thin tail from squashing the rest.
        pooled = pd.concat([d for m in ("M1", "M2", "M3") for d in runs[m]],
                           ignore_index=True)["efe_sep"].to_numpy(dtype=float)
        dg_max = float(np.nanpercentile(pooled, 99)) if np.isfinite(pooled).any() else 1.0
        for m in ("M1", "M2", "M3"):
            cx, cy, cs = binned(runs[m], "efe_sep", "policy_entropy",
                                nbins=8, xmax=dg_max)
            axH.plot(cx, cy, color=MODEL[m], ls=MODEL_LS[m], lw=2, marker="o", ms=4, label=m)
            axH.fill_between(cx, cy - cs, cy + cs, color=MODEL[m], alpha=0.15, linewidth=0)
        axH.set_title("Policy entropy vs. EFE separation")
        axH.set_xlabel(r"EFE separation  $\Delta G_t$")
        axH.set_ylabel(r"H[$q(\pi)$]  (nats)")
        axH.legend()
        clean(axH, "y")
    else:
        axH.axis("off")
    panel_tag(axH, "H")

    fig.tight_layout(w_pad=2.5, h_pad=3.0)
    save_figure(fig, out_path)


# ============================ MAIN ============================
def main():
    p = argparse.ArgumentParser(description="Render both paper figures from a run dir.")
    p.add_argument("--results-dir", default="results/model_recovery/run_synth")
    p.add_argument("--out", default=None, help="Output dir (default: figures/<run_label>).")
    args = p.parse_args()

    apply_style()
    run_label = os.path.basename(os.path.normpath(args.results_dir))
    out_dir = args.out or os.path.join("figures", run_label)
    os.makedirs(out_dir, exist_ok=True)

    aic_path = os.path.join(out_dir, "model_recovery_aic.png")
    mech_path = os.path.join(out_dir, "mechanistic_analysis.png")
    fig_aic(args.results_dir, aic_path)
    fig_mechanistic(args.results_dir, mech_path)
    print("wrote (each as .png and .pdf):\n"
          f"  {aic_path}\n  {mech_path}")


if __name__ == "__main__":
    main()
