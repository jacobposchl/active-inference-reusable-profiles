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

# M3 generating template (config M3_DEFAULTS: profile gammas [2, 4], Z = identity).
M3_GEN_GAMMA = (2.0, 4.0)
GEN_C = "#1f8a63"    # generator (template) gamma_t line
REC_C = "#9a9a9a"    # recovered (fitted) gamma_t line

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
        "legend.fontsize": 10.5, "legend.frameon": True,
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
def load_runs(run_dir, model, generator="M3"):
    """List of per-run DataFrames for `model` fitted to `generator`'s data."""
    pat = os.path.join(run_dir, "trial_level", f"gen_{generator}", f"model_{model}", "run_*.csv")
    dfs = []
    for f in sorted(glob.glob(pat)):
        df = pd.read_csv(f)
        w = df["belief_context"].apply(ast.literal_eval)
        df["w0"] = w.apply(lambda x: float(x[0]))   # volatile-profile weight = q(volatile)
        df["w1"] = w.apply(lambda x: float(x[1]))   # stable-profile weight   = q(stable)
        # M3 generator's effective precision (template gammas under identity Z):
        # gamma_t = q(volatile)*gamma0 + q(stable)*gamma1. Exact from the logged belief.
        df["gen_gamma"] = M3_GEN_GAMMA[0] * df["w0"] + M3_GEN_GAMMA[1] * df["w1"]
        dfs.append(df.reset_index(drop=True))
    return dfs


def aligned(dfs, col, direction, pre=10, post=40):
    """Mean of `col` in a [-pre, +post] window around context reversals of a
    given direction ('v2s' volatile->stable, or 's2v')."""
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
        return None, None, 0
    a = np.vstack(segs)
    return a.mean(0), np.arange(-pre, post), len(segs)


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
    """Mean of ycol within equal-width bins of xcol, pooled across runs."""
    d = pd.concat(dfs, ignore_index=True)
    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    hi = float(np.nanmax(x)) if xmax is None else xmax
    edges = np.linspace(0.0, hi, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ym = np.full(nbins, np.nan)
    for i in range(nbins):
        lo, up = edges[i], edges[i + 1]
        m = (x >= lo) & (x <= up) if i == nbins - 1 else (x >= lo) & (x < up)
        if m.any():
            ym[i] = np.nanmean(y[m])
    return ctr, ym


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
    fig.savefig(out_path)
    plt.close(fig)


# ============================ FIGURE 1: MECHANISTIC ============================
def fig_mechanistic(run_dir, out_path):
    runs = {m: load_runs(run_dir, m) for m in ("M1", "M2", "M3")}
    has_pe = all(c in runs["M1"][0].columns for c in ("policy_entropy", "better_arm_entropy"))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    (axA, axB, axC, axD), (axE, axF, axG, axH) = axes

    # --- A / B: profile recruitment around reversals (M3) ---
    for ax, direction, title, tag in [
        (axA, "v2s", "Profile recruitment: volatile → stable", "A"),
        (axB, "s2v", "Profile recruitment: stable → volatile", "B"),
    ]:
        w0, xs, n = aligned(runs["M3"], "w0", direction)
        w1, _, _ = aligned(runs["M3"], "w1", direction)
        if w0 is not None:
            ax.plot(xs, w0, color=CTX["volatile"], lw=2, label="$w_0$ volatile profile")
            ax.plot(xs, w1, color=CTX["stable"], lw=2, label="$w_1$ stable profile")
        ax.axvline(0, color=INK3, lw=1, ls="--")
        ax.set_title(f"{title}\n({n} reversals)")
        ax.set_xlabel("Trials relative to reversal")
        ax.set_ylabel("Profile weight")
        ax.set_ylim(0, 1)
        swatch_legend(ax, [("$w_0$ volatile profile", CTX["volatile"]),
                           ("$w_1$ stable profile", CTX["stable"])], loc="center right")
        clean(ax, "y")
        panel_tag(ax, tag)

    # --- C: effective precision around reversals, per model ---
    for m in ("M1", "M2", "M3"):
        g, xs = around_reversal(runs[m], "gamma")
        axC.plot(xs, g, color=MODEL[m], ls=MODEL_LS[m], lw=2, label=m)
    axC.axvline(0, color=INK3, lw=1, ls="--")
    axC.set_title("Effective precision around reversals")
    axC.set_xlabel("Trials relative to reversal")
    axC.set_ylabel(r"Effective $\gamma_t$")
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

    # --- E: mean precision by inferred context, per model ---
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
    axE.set_title("Precision by inferred context")
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
    axF.set_xlabel("Inferred context")
    axF.set_ylim(0, 1)
    clean(axF, "y")
    panel_tag(axF, "F")

    # --- G: profile weight over the run (M3), data-driven (no re-sim) ---
    W0 = np.vstack([df["w0"].to_numpy() for df in runs["M3"]])
    W1 = np.vstack([df["w1"].to_numpy() for df in runs["M3"]])
    t = np.arange(W0.shape[1])
    for W, key, lab in [(W0, "volatile", "$w_0$ volatile profile"),
                        (W1, "stable", "$w_1$ stable profile")]:
        mu, sd = W.mean(0), W.std(0)
        axG.plot(t, mu, color=CTX[key], lw=1.8, label=lab)
        axG.fill_between(t, mu - sd, mu + sd, color=CTX[key], alpha=0.15, linewidth=0)
    for r in runs["M3"][0].index[runs["M3"][0]["is_reversal"] == 1]:
        axG.axvline(r, color=INK3, lw=0.6, ls=":", alpha=0.7)
    axG.set_title("Profile weights over trials (M3)")
    axG.set_xlabel("Trial")
    axG.set_ylabel("Profile weight")
    axG.set_ylim(0, 1)
    swatch_legend(axG, [("$w_0$ volatile profile", CTX["volatile"]),
                        ("$w_1$ stable profile", CTX["stable"])], loc="center right")
    clean(axG, "y")
    panel_tag(axG, "G")

    # --- H: policy entropy vs better-arm uncertainty, per model ---
    if has_pe:
        for m in ("M1", "M2", "M3"):
            cx, cy = binned(runs[m], "better_arm_entropy", "policy_entropy",
                            nbins=8, xmax=float(np.log(2)))
            axH.plot(cx, cy, color=MODEL[m], ls=MODEL_LS[m], lw=2, marker="o", ms=4, label=m)
        axH.set_title("Policy entropy vs. arm uncertainty")
        axH.set_xlabel(r"H[$q$(better arm)]  (nats)")
        axH.set_ylabel(r"H[$q(\pi)$]  (nats)")
        axH.legend()
        clean(axH, "y")
    else:
        axH.axis("off")
    panel_tag(axH, "H")

    fig.tight_layout(w_pad=2.5, h_pad=3.0)
    fig.savefig(out_path)
    plt.close(fig)


# ============================ FIGURE 3: CONTEXT & PRECISION ============================
def fig_context_precision(run_dir, out_path):
    """State-context belief and effective precision around regime change.

    The precision row overlays the M3 GENERATING template (gamma=[2,4], which tracks
    context) with the RECOVERED fit (gamma=[5,5], which is flat) -- clearly labeled so
    the template is never read as the fitted model.
    """
    runs = load_runs(run_dir, "M3")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    (axA, axB), (axC, axD) = axes
    cols = [("v2s", "volatile → stable", axA, axC, "A", "C"),
            ("s2v", "stable → volatile", axB, axD, "B", "D")]

    for direction, dname, ax_ctx, ax_gam, tag_ctx, tag_gam in cols:
        # --- top: state-context belief q(context) ---
        qv, xs, n = aligned(runs, "w0", direction)
        qsb, _, _ = aligned(runs, "w1", direction)
        if qv is not None:
            ax_ctx.plot(xs, qv, color=CTX["volatile"], lw=2.2, label="q(volatile)")
            ax_ctx.plot(xs, qsb, color=CTX["stable"], lw=2.2, label="q(stable)")
        ax_ctx.axvline(0, color=INK3, lw=1, ls="--")
        ax_ctx.set_title(f"State context: {dname}\n({n} reversals)")
        ax_ctx.set_xlabel("Trials relative to reversal")
        ax_ctx.set_ylabel("q(context)")
        ax_ctx.set_ylim(0, 1)
        swatch_legend(ax_ctx, [("q(volatile)", CTX["volatile"]),
                               ("q(stable)", CTX["stable"])], loc="center right")
        clean(ax_ctx, "y")
        panel_tag(ax_ctx, tag_ctx)

        # --- bottom: effective precision, generator (template) vs recovered (fit) ---
        gg, _, _ = aligned(runs, "gen_gamma", direction)
        rg, _, _ = aligned(runs, "gamma", direction)
        if gg is not None:
            ax_gam.plot(xs, gg, color=GEN_C, lw=2.6, label="generator (γ = [2, 4])")
            ax_gam.plot(xs, rg, color=REC_C, lw=2.2, ls="--", label="recovered fit (γ = [5, 5])")
        ax_gam.axvline(0, color=INK3, lw=1, ls="--")
        ax_gam.set_title(f"Effective precision: {dname}")
        ax_gam.set_xlabel("Trials relative to reversal")
        ax_gam.set_ylabel(r"Effective $\gamma_t$")
        ax_gam.set_ylim(1.5, 5.5)
        ax_gam.legend(loc="center right")
        clean(ax_gam, "y")
        panel_tag(ax_gam, tag_gam)

    fig.suptitle("State context & effective precision around regime change\n"
                 "M3 generative template (γ = [2, 4])  vs  recovered fit (γ = [5, 5])",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(w_pad=2.5, h_pad=3.0, rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


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
    ctx_path = os.path.join(out_dir, "context_precision.png")
    fig_aic(args.results_dir, aic_path)
    fig_mechanistic(args.results_dir, mech_path)
    fig_context_precision(args.results_dir, ctx_path)
    print(f"wrote:\n  {aic_path}\n  {mech_path}\n  {ctx_path}")


if __name__ == "__main__":
    main()
