from __future__ import annotations

import argparse
import textwrap
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, Any
import re


# Publication-style Matplotlib defaults: tweak font sizes, family, and
# layout options to produce figures suitable for papers and presentations.
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
    'axes.grid': False,
})

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DEFAULT_FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'


def summarize_array(name: str, arr: np.ndarray) -> str:
    """Return a one-paragraph summary for a numpy array."""
    try:
        arr = np.asarray(arr)
    except Exception:
        return f"{name}: (non-numeric or unsupported type)"

    s = [f"- {name}: dtype={arr.dtype}, shape={arr.shape}"]
    if arr.size == 0:
        s.append("  - empty array")
        return "\n".join(s)

    if np.issubdtype(arr.dtype, np.number):
        s.append(f"  - min={np.nanmin(arr):.6g}, mean={np.nanmean(arr):.6g}, max={np.nanmax(arr):.6g}, std={np.nanstd(arr):.6g}")
        # If it's 2D and smallish, show a small excerpt
        if arr.ndim == 2 and max(arr.shape) <= 10:
            s.append("  - sample values:")
            s.extend(["    " + str(row) for row in arr.tolist()])
        else:
            # show corners for large arrays
            if arr.ndim == 2:
                r0 = arr[0, :min(5, arr.shape[1])]
                r1 = arr[-1, :min(5, arr.shape[1])]
                s.append(f"  - top-row-sample={np.array2string(r0, precision=5, max_line_width=120)}")
                s.append(f"  - bottom-row-sample={np.array2string(r1, precision=5, max_line_width=120)}")
    else:
        s.append(f"  - non-numeric type; len={arr.size}")

    return "\n".join(s)


def analyze_npz(path: Path, fig_dir: Path, save_md: bool = True) -> Dict[str, Any]:
    """Load an NPZ file, generate summaries, save annotated heatmaps and a markdown report.

    Returns a dictionary with metadata and statistics.
    """
    report_lines = []
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    base = path.stem

    # Derive a human-friendly display name from the filename stem. For
    # known experiment stems like 'k2_vs_k1_grid_results_A' we map to
    # 'Case A'. Otherwise, fall back to a prettified title (underscores
    # -> spaces, title-cased).
    m = re.match(r"k2_vs_k1_grid_results_([A-Za-z0-9]+)$", base)
    if m:
        suffix = m.group(1)
        display_name = f"Case {suffix.upper()}" if len(suffix) == 1 else f"Case {suffix}"
    else:
        display_name = base.replace('_', ' ').replace('-', ' ').title()

    report_lines.append(f"# Heatmap analysis — {display_name}\n")
    report_lines.append(f"File: `{path}`\n")

    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    report_lines.append("## Contents\n")
    report_lines.append(f"Found keys: {keys}\n")

    meta = {"file": str(path), "keys": keys, "arrays": {}}

    # Map potential axis labels
    prob_rewards = data.get('prob_rewards', None)
    prob_hints = data.get('prob_hints', None)
    if prob_rewards is not None:
        report_lines.append(f"- `prob_rewards`: {np.asarray(prob_rewards).tolist()}\n")
    if prob_hints is not None:
        report_lines.append(f"- `prob_hints`: {np.asarray(prob_hints).tolist()}\n")

    # Walk arrays and summarize
    for k in keys:
        arr = data[k]
        report = summarize_array(k, arr)
        report_lines.append(report + "\n")
        meta['arrays'][k] = {
            'dtype': str(arr.dtype),
            'shape': tuple(arr.shape),
            'min': float(np.nanmin(arr)) if np.issubdtype(arr.dtype, np.number) and arr.size else None,
            'mean': float(np.nanmean(arr)) if np.issubdtype(arr.dtype, np.number) and arr.size else None,
            'max': float(np.nanmax(arr)) if np.issubdtype(arr.dtype, np.number) and arr.size else None,
            'std': float(np.nanstd(arr)) if np.issubdtype(arr.dtype, np.number) and arr.size else None,
        }

    # Create figure(s) for delta arrays if present
    os.makedirs(fig_dir, exist_ok=True)

    # Common deltas
    deltas = [('ll_delta', 'Δ Log Likelihood (K=2 - K=1)', 'coolwarm'),
              ('acc_delta', 'Δ Accuracy (K=2 - K=1)', 'coolwarm'),
              ('bic_delta', 'Δ BIC (K=2 - K=1)', 'coolwarm')]

    for key, title, cmap in deltas:
        if key in data.files:
            arr = np.asarray(data[key], dtype=float)
            if arr.ndim == 2:
                # Compute symmetric vmin/vmax around zero for diverging colormaps
                finite_vals = arr[np.isfinite(arr)]
                if finite_vals.size == 0:
                    vmin, vmax = -1.0, 1.0
                else:
                    mx = float(np.nanmax(np.abs(finite_vals)))
                    # If mx is very small, expand a bit for good contrast
                    if mx == 0:
                        mx = 1e-6
                    vmin, vmax = -mx, mx

                fig, ax = plt.subplots(figsize=(6, 5))

                im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')

                # Title and axis labels
                # Use the friendly display name in figure titles (keeps
                # filenames unchanged but makes plots publication-ready).
                ax.set_title(f"{title} — {display_name}")

                # If prob_hints/prob_rewards available, use them as tick labels
                if prob_hints is not None:
                    ax.set_xticks(range(len(prob_hints)))
                    ax.set_xticklabels([f"{float(x):.2f}" for x in prob_hints], rotation=45, ha='right')
                else:
                    ax.set_xticks(range(arr.shape[1]))

                if prob_rewards is not None:
                    ax.set_yticks(range(len(prob_rewards)))
                    ax.set_yticklabels([f"{float(x):.2f}" for x in prob_rewards])
                else:
                    ax.set_yticks(range(arr.shape[0]))

                # Use human-friendly axis labels (avoid code-like underscores)
                ax.set_xlabel('Probability of hint')
                ax.set_ylabel('Probability of reward')

                # Colorbar with label and ticks formatted
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.set_ylabel(title, rotation=270, labelpad=12)

                # Annotate cell values when the grid is small; choose text
                # color for readability depending on background luminance.
                if arr.shape[0] <= 12 and arr.shape[1] <= 12:
                    # Normalize values to [0,1] for cmap lookup
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    cmap_obj = plt.get_cmap(cmap)
                    for (i, j), val in np.ndenumerate(arr):
                        if not np.isfinite(val):
                            txt = 'n/a'
                        else:
                            txt = f"{val:.2f}"
                        rgba = cmap_obj(norm(val)) if np.isfinite(val) else (1.0, 1.0, 1.0, 1.0)
                        # Perceptual luminance approximation to pick black/white text
                        r, g, b, _ = rgba
                        luminance = 0.299 * r + 0.587 * g + 0.114 * b
                        txt_color = 'white' if luminance < 0.5 else 'black'
                        ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=txt_color)

                # Save high-resolution figures suitable for publication
                fig_path_png = fig_dir / f"{base}_{key}.png"
                fig_path_pdf = fig_dir / f"{base}_{key}.pdf"
                plt.tight_layout()
                plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
                plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
                plt.close(fig)
                report_lines.append(f"Saved annotated heatmap to `{fig_path_png}` and `{fig_path_pdf}`\n")
                meta['arrays'][key]['figure_png'] = str(fig_path_png)
                meta['arrays'][key]['figure_pdf'] = str(fig_path_pdf)

    # Write markdown report
    md_path = path.with_suffix('.md')
    if save_md:
        with open(md_path, 'w', encoding='utf8') as fh:
            fh.write('\n'.join(report_lines))
        print(f"Wrote markdown summary: {md_path}")

    return meta


def main(fnames: list[str] | None = None, out_dir: str | None = None):
    parser = argparse.ArgumentParser(description='Thorough heatmap NPZ analysis')
    parser.add_argument('-f', '--files', nargs='+', help='NPZ files or suffix letters (A/B/C)', default=None)
    parser.add_argument('-o', '--out-dir', help='Directory to write figures', default=None)
    args = parser.parse_args()

    # Determine files
    if args.files:
        files = []
        for item in args.files:
            if item.upper() in ('A', 'B', 'C'):
                files.append(DEFAULT_DATA_DIR / f'k2_vs_k1_grid_results_{item.upper()}.npz')
            else:
                files.append(Path(item))
    else:
        # default: A, B, C in data dir
        files = [DEFAULT_DATA_DIR / f'k2_vs_k1_grid_results_{s}.npz' for s in ('A', 'B', 'C')]

    fig_dir = Path(args.out_dir) if args.out_dir else DEFAULT_FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)

    metas = {}
    for f in files:
        try:
            print(f"Analyzing {f}...")
            meta = analyze_npz(f, fig_dir, save_md=True)
            metas[f.name] = meta
        except Exception as e:
            print(f"Failed to analyze {f}: {e}")

    # After processing individual files, compute averaged heatmaps across
    # the provided NPZ files for the common delta arrays (if present in the
    # set). This produces three extra figures titled "Averaged".
    # Collect arrays per-key from the files we attempted to analyze.
    collected = { 'll_delta': [], 'acc_delta': [], 'bic_delta': [] }
    collected_shapes = {}
    # Try to read prob_hints/prob_rewards from first available file
    _global_prob_hints = None
    _global_prob_rewards = None
    for f in files:
        try:
            if not f.exists():
                continue
            npz = np.load(f, allow_pickle=True)
            if _global_prob_hints is None and 'prob_hints' in npz.files:
                _global_prob_hints = np.asarray(npz['prob_hints'])
            if _global_prob_rewards is None and 'prob_rewards' in npz.files:
                _global_prob_rewards = np.asarray(npz['prob_rewards'])
            for key in list(collected.keys()):
                if key in npz.files:
                    arr = np.asarray(npz[key], dtype=float)
                    collected[key].append(arr)
                    collected_shapes[key] = arr.shape
        except Exception:
            # skip files we can't open here
                continue

    # For each key, if we have at least one array, compute the mean across
    # files and render an averaged figure.
    for key, arrs in collected.items():
        if not arrs:
            continue
        # Stack along a new axis and average
        stack = np.stack(arrs, axis=0)
        avg = np.nanmean(stack, axis=0)
        # Plot using the same visual conventions as individual analyses
        # (diverging colormap centered at zero)
        finite_vals = avg[np.isfinite(avg)]
        if finite_vals.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            mx = float(np.nanmax(np.abs(finite_vals)))
            if mx == 0:
                mx = 1e-6
            vmin, vmax = -mx, mx

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(avg, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest', aspect='auto')
        # Title should explicitly say "Averaged" as requested
        title_map = {
            'll_delta': 'Δ Log Likelihood (K=2 − K=1)',
            'acc_delta': 'Δ Accuracy (K=2 − K=1)',
            'bic_delta': 'Δ BIC (K=2 − K=1)'
        }
        ax.set_title(f"{title_map.get(key, key)} — Averaged")
        if _global_prob_hints is not None:
            ax.set_xticks(range(len(_global_prob_hints)))
            ax.set_xticklabels([f"{float(x):.2f}" for x in _global_prob_hints], rotation=45, ha='right')
        else:
            ax.set_xticks(range(avg.shape[1]))
        if _global_prob_rewards is not None:
            ax.set_yticks(range(len(_global_prob_rewards)))
            ax.set_yticklabels([f"{float(x):.2f}" for x in _global_prob_rewards])
        else:
            ax.set_yticks(range(avg.shape[0]))
        ax.set_xlabel('Probability of hint')
        ax.set_ylabel('Probability of reward')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(title_map.get(key, key), rotation=270, labelpad=12)
        # Annotate small grids
        if avg.shape[0] <= 12 and avg.shape[1] <= 12:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap('RdBu_r')
            for (i, j), val in np.ndenumerate(avg):
                if not np.isfinite(val):
                    txt = 'n/a'
                else:
                    txt = f"{val:.2f}"
                rgba = cmap_obj(norm(val)) if np.isfinite(val) else (1.0, 1.0, 1.0, 1.0)
                r, g, b, _ = rgba
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = 'white' if luminance < 0.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=txt_color)

        fig_path_png = fig_dir / f"averaged_{key}.png"
        fig_path_pdf = fig_dir / f"averaged_{key}.pdf"
        plt.tight_layout()
        plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved averaged heatmap: {fig_path_png} and {fig_path_pdf}")
        # Record in metas under a synthetic key
        metas[f"averaged_{key}"] = {'figure_png': str(fig_path_png), 'figure_pdf': str(fig_path_pdf)}

    # Save a small JSON index summarizing all analyzed files
    index_path = DEFAULT_DATA_DIR / 'heatmap_analysis_index.json'
    with open(index_path, 'w', encoding='utf8') as fh:
        json.dump(metas, fh, indent=2)
    print(f"Wrote analysis index: {index_path}")


if __name__ == '__main__':
    main()
