import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_per_run_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert fitted ll fields
            entry = {
                'generator': r['generator'],
                'run_idx': int(r['run_idx']),
                'seed': int(r['seed'])
            }
            for k in r:
                if k.startswith('fitted_ll_'):
                    entry[k] = float(r[k])
            entry['winner'] = r.get('winner', '')
            rows.append(entry)
    return rows


def read_confusion_csv(path):
    conf = {}
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        models = header[1:]
        for row in reader:
            gen = row[0]
            counts = list(map(int, row[1:]))
            conf[gen] = dict(zip(models, counts))
    return conf, models


def summarize_and_plot(per_run_path, conf_path, out_dir='results/figures'):
    os.makedirs(out_dir, exist_ok=True)
    rows = read_per_run_csv(per_run_path)
    conf, models = read_confusion_csv(conf_path)

    # organize deltas per generator
    gens = sorted(set(r['generator'] for r in rows))
    deltas_M3_M1 = {g: [] for g in gens}
    deltas_M3_M2 = {g: [] for g in gens}
    for r in rows:
        g = r['generator']
        ll_M1 = r.get('fitted_ll_M1')
        ll_M2 = r.get('fitted_ll_M2')
        ll_M3 = r.get('fitted_ll_M3')
        if ll_M1 is not None and ll_M3 is not None:
            deltas_M3_M1[g].append(ll_M3 - ll_M1)
        if ll_M2 is not None and ll_M3 is not None:
            deltas_M3_M2[g].append(ll_M3 - ll_M2)

    # Print numeric summaries
    summary_lines = []
    for g in gens:
        a = np.array(deltas_M3_M1[g]) if deltas_M3_M1[g] else np.array([])
        b = np.array(deltas_M3_M2[g]) if deltas_M3_M2[g] else np.array([])
        def sstats(x):
            if x.size == 0:
                return 'n=0'
            return f'n={x.size} mean={x.mean():.3f} sd={x.std():.3f} med={np.median(x):.3f} min={x.min():.3f} max={x.max():.3f}'
        summary_lines.append(f"Generator={g} | Δ(M3-M1): {sstats(a)} | Δ(M3-M2): {sstats(b)}")

    print('\n'.join(summary_lines))

    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = gens
    data1 = [deltas_M3_M1[g] if deltas_M3_M1[g] else [np.nan] for g in gens]
    data2 = [deltas_M3_M2[g] if deltas_M3_M2[g] else [np.nan] for g in gens]

    axes[0].boxplot(data1, labels=labels, patch_artist=True)
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_title('ΔLL = M3 - M1 (fitted) by generator')
    axes[0].set_ylabel('ΔLL (nats)')

    axes[1].boxplot(data2, labels=labels, patch_artist=True)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_title('ΔLL = M3 - M2 (fitted) by generator')

    plt.tight_layout()
    outpath = os.path.join(out_dir, 'model_recovery_deltas_boxplots_fitted_seed1.png')
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    # Confusion heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    gens_order = sorted(conf.keys())
    mat = np.array([[conf[g].get(m, 0) for m in models] for g in gens_order])
    im = ax.imshow(mat, cmap='Blues')
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(np.arange(len(gens_order)))
    ax.set_yticklabels(gens_order)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha='center', va='center', color='black')
    ax.set_title('Model recovery (fitted) confusion')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    heatpath = os.path.join(out_dir, 'model_recovery_fitted_confusion_seed1.png')
    fig.savefig(heatpath, dpi=200)
    plt.close(fig)

    print(f"\nSaved boxplots: {outpath}")
    print(f"Saved confusion heatmap: {heatpath}")
    return summary_lines, outpath, heatpath


if __name__ == '__main__':
    per_run_path = os.path.join('results', 'csv', 'model_recovery_fitted_per_run_seed1.csv')
    conf_path = os.path.join('results', 'csv', 'model_recovery_fitted_confusion_seed1.csv')
    summarize_and_plot(per_run_path, conf_path)
