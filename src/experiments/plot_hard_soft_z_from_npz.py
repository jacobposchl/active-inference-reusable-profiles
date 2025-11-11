"""Quick plotter for hard_soft_z results

Reads the numeric results saved by `hard_soft_z_barplot.py` (NPZ) and
creates a clean grouped bar chart with a left axis for total reward and
right axis for accuracy (0-100%). This is intended as a quick script
to re-generate the figure from `results/data/hard_soft_z_results.npz`.

Usage (Windows cmd.exe):
  python src\experiments\plot_hard_soft_z_from_npz.py

Options:
  --input   Path to NPZ file (default: results/data/hard_soft_z_results.npz)
  --outdir  Directory to save figures (default: results/figures)
  --dpi     Save DPI (default: 300)
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def load_results(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = [str(x) for x in data['labels']]
    total_reward_means = np.array(data['total_reward_means']).astype(float)
    total_reward_stds = np.array(data['total_reward_stds']).astype(float)
    accuracy_means = np.array(data['accuracy_means']).astype(float)
    accuracy_stds = np.array(data['accuracy_stds']).astype(float)
    choice_accuracy_means = np.array(data['choice_accuracy_means']).astype(float)
    choice_accuracy_stds = np.array(data['choice_accuracy_stds']).astype(float)

    return {
        'labels': labels,
        'total_reward_means': total_reward_means,
        'total_reward_stds': total_reward_stds,
        'accuracy_means': accuracy_means,
        'accuracy_stds': accuracy_stds,
        'choice_accuracy_means': choice_accuracy_means,
        'choice_accuracy_stds': choice_accuracy_stds
    }


def plot(results, outdir: Path, dpi=300):
    labels = results['labels']
    means = results['total_reward_means']
    stds = results['total_reward_stds']
    acc_means = results['accuracy_means']
    acc_stds = results['accuracy_stds']
    cho_means = results['choice_accuracy_means']
    cho_stds = results['choice_accuracy_stds']

    # Set plotting theme for publication-ready figures
    sns.set_theme(style='whitegrid', context='paper', font='sans-serif', rc={
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.35,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
    })

    x = np.arange(len(labels))
    width = 0.25
    pos1 = x - width
    pos2 = x
    pos3 = x + width

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Use a consistent, colorblind-friendly palette
    colors = sns.color_palette('deep', 3)

    # Compute left axis limits - MUST start at 0 to align with ax2's zero point
    try:
        reward_min = float(np.min(means - stds))
        reward_max = float(np.max(means + stds))
    except Exception:
        reward_min, reward_max = float(np.min(means)), float(np.max(means))

    rng = reward_max - reward_min if (reward_max - reward_min) > 0 else 1.0
    margin = 0.15 * rng
    # CRITICAL FIX: Start at 0, not negative values
    # This ensures ax1 and ax2 have aligned zero baselines for bar plots
    lower = 0.0
    upper = reward_max + margin
    ax1.set_ylim(lower, upper)

    # Plot on ax1 (reward)
    bars1 = ax1.bar(pos1, means, width, yerr=stds, capsize=4,
                    label='Total reward', color=colors[0], edgecolor='k', linewidth=0.6, alpha=0.95)

    # Create secondary axis
    ax2 = ax1.twinx()

    # Plot on ax2 (accuracy)
    bars2 = ax2.bar(pos2, acc_means, width, yerr=acc_stds, capsize=4,
                    label='Accuracy (all trials)', color=colors[1], edgecolor='k', linewidth=0.6, alpha=0.95)
    bars3 = ax2.bar(pos3, cho_means, width, yerr=cho_stds, capsize=4,
                    label='Accuracy (choice-only)', color=colors[2], edgecolor='k', linewidth=0.6, alpha=0.95)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Total reward', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax1.set_title('Ablation: Z-matrix mixing — Profile set C (P(reward)=0.65, P(hint)=0.65)', fontsize=11)

    # Combine legends and place outside the plot area (clean for publication)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(h1 + h2, l1 + l2, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    legend.get_frame().set_edgecolor('none')

    # Format right axis as percent and set to 0..100%

    ax2.set_ylim(0.0, 1.0)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    # Styling: lighter grid lines, remove top/right spines
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # Ticks and font sizes
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='y', which='major', labelsize=9)

    # Helper: add numeric labels
    def autolabel(ax, bars, fmt="{:.2f}", pct=False):
        """Attach a text label above each bar in *bars*, formatted.

        If pct=True, format as percent (no decimals). Otherwise use fmt.
        """
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        offset = 0.02 * y_range

        for bar in bars:
            h = bar.get_height()
            y_pos = h + offset

            if pct:
                txt = f"{h:.0%}"
            else:
                txt = fmt.format(h)

            ax.text(bar.get_x() + bar.get_width() / 2., y_pos, txt,
                    ha='center', va='bottom', fontsize=8)

    autolabel(ax1, bars1, "{:.1f}")
    autolabel(ax2, bars2, pct=True)
    autolabel(ax2, bars3, pct=True)

    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / 'hard_soft_z_barplot_clean.png'
    pdf = outdir / 'hard_soft_z_barplot_clean.pdf'
    fig.savefig(png, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {png}\n       {pdf}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='results/data/hard_soft_z_results.npz',
                        help='Path to results NPZ file')
    parser.add_argument('--outdir', '-o', type=str, default='results/figures', help='Output directory for figures')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    args = parser.parse_args()

    npz_path = Path(args.input)
    if not npz_path.exists():
        raise FileNotFoundError(f"Results file not found: {npz_path}")

    results = load_results(npz_path)
    plot(results, Path(args.outdir), dpi=args.dpi)


if __name__ == '__main__':
    main()