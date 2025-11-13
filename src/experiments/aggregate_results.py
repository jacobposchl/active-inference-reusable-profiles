"""
Aggregate per-run CSVs into summary tables and Markdown report.

Usage:
    python src\experiments\aggregate_results.py

Produces:
- results/summary/model_comparison_summary.csv  (wide table with per-interval per-model means and stds)
- results/summary/model_comparison_summary.md   (human-readable Markdown with tables)

The script reads CSV files from results/csv/*.csv produced by model_comparison.py

"""
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
from scipy import stats

CSV_GLOB = os.path.join('results', 'csv', 'model_comparison_*.csv')
OUT_DIR = os.path.join('results', 'summary')
os.makedirs(OUT_DIR, exist_ok=True)


def mean_std_ci(x, ci=95):
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    mean = x.mean()
    std = x.std(ddof=1)
    sem = stats.sem(x, nan_policy='omit') if n > 1 else np.nan
    if n > 1:
        t = stats.t.ppf(1 - (1 - ci/100)/2, df=n-1)
        lo = mean - t * sem
        hi = mean + t * sem
    else:
        lo, hi = mean, mean
    return mean, std, (lo, hi)


def summarize_df(df):
    """Return a summary DataFrame grouped by interval and model."""
    metrics = ['mean_accuracy', 'total_reward', 'adaptation_time', 'mean_gamma', 'log_likelihood', 'aic', 'bic']
    rows = []
    grouped = df.groupby(['interval', 'model'])
    for (interval, model), g in grouped:
        n = len(g)
        row = {'interval': interval, 'model': model, 'n_runs': n}
        for m in metrics:
            mean, std, (lo, hi) = mean_std_ci(g[m].dropna().values, ci=95)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
            row[f'{m}_95ci_lo'] = lo
            row[f'{m}_95ci_hi'] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def make_markdown_table(summary_df, outpath_md):
    """Create a human readable Markdown file with tables per interval."""
    with open(outpath_md, 'w', encoding='utf-8') as f:
        f.write('# Model comparison summary\n\n')
        intervals = sorted(summary_df['interval'].unique(), key=lambda x: (str(x)))
        for interval in intervals:
            f.write(f'## Interval: {interval}\n\n')
            sub = summary_df[summary_df['interval'] == interval].sort_values('model')
            # create a compact table with selected columns
            cols = ['model', 'n_runs',
                    'mean_accuracy_mean', 'mean_accuracy_std',
                    'total_reward_mean', 'total_reward_std',
                    'log_likelihood_mean', 'log_likelihood_std',
                    'aic_mean', 'aic_std',
                    'bic_mean', 'bic_std']
            sub = sub[cols]
            sub = sub.rename(columns={
                'model': 'Model', 'n_runs': 'Runs',
                'mean_accuracy_mean': 'Accuracy (mean)', 'mean_accuracy_std': 'Accuracy (std)',
                'total_reward_mean': 'Total Reward (mean)', 'total_reward_std': 'Total Reward (std)',
                'log_likelihood_mean': 'LogLik (mean)', 'log_likelihood_std': 'LogLik (std)',
                'aic_mean': 'AIC (mean)', 'aic_std': 'AIC (std)',
                'bic_mean': 'BIC (mean)', 'bic_std': 'BIC (std)'
            })
            f.write(sub.to_markdown(index=False))
            f.write('\n\n')

    print(f'Markdown summary written to {outpath_md}')


def main():
    files = glob.glob(CSV_GLOB)
    if len(files) == 0:
        print('No CSV files found in results/csv. Please run model_comparison.py first.')
        return
    dfs = []
    for p in files:
        try:
            d = pd.read_csv(p)
            # Ensure interval is str for grouping consistency
            d['interval'] = d['interval'].astype(str)
            dfs.append(d)
        except Exception as e:
            print(f'Failed to read {p}: {e}')
    df = pd.concat(dfs, ignore_index=True)

    summary = summarize_df(df)
    # Save wide CSV
    out_csv = os.path.join(OUT_DIR, 'model_comparison_summary.csv')
    summary.to_csv(out_csv, index=False)
    print(f'Summary CSV written to {out_csv}')

    # Also write a readable Markdown
    out_md = os.path.join(OUT_DIR, 'model_comparison_summary.md')
    make_markdown_table(summary, out_md)

    # Additionally print a short comparison table (best model per interval by AIC)
    bests = summary.loc[summary.groupby('interval')['aic_mean'].idxmin()]
    print('\nBest models by interval (AIC mean):')
    print(bests[['interval', 'model', 'aic_mean', 'bic_mean']].to_string(index=False))

    # Paired statistical tests: compare M3 to M1 and M2 per interval for selected metrics
    metrics_to_test = ['mean_accuracy', 'log_likelihood', 'total_reward']
    stats_rows = []
    intervals = sorted(df['interval'].unique())
    for interval in intervals:
        sub = df[df['interval'] == interval]
        models = sub['model'].unique()
        # require M3 present
        if 'M3' not in models:
            continue
        for comp in ['M1', 'M2']:
            if comp not in models:
                continue
            # align by run index
            m3 = sub[sub['model'] == 'M3'].sort_values('run_idx')
            mm = sub[sub['model'] == comp].sort_values('run_idx')
            # ensure same number of runs
            n = min(len(m3), len(mm))
            for metric in metrics_to_test:
                x = m3[metric].values[:n]
                y = mm[metric].values[:n]
                # paired t-test
                try:
                    t_res = stats.ttest_rel(x, y, nan_policy='omit')
                    t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
                except Exception:
                    t_stat, t_p = np.nan, np.nan
                # wilcoxon
                try:
                    w_res = stats.wilcoxon(x, y)
                    w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    w_stat, w_p = np.nan, np.nan
                # Cohen's d for paired samples (mean diff / sd of diffs)
                diffs = x - y
                try:
                    d = np.nanmean(diffs) / np.nanstd(diffs, ddof=1)
                except Exception:
                    d = np.nan
                # bootstrap 95% CI for mean difference
                try:
                    n_boot = 5000
                    boots = []
                    rng = np.random.default_rng(42)
                    for _ in range(n_boot):
                        idx = rng.integers(0, n, n)
                        boots.append(np.nanmean(diffs[idx]))
                    lo = np.percentile(boots, 2.5)
                    hi = np.percentile(boots, 97.5)
                except Exception:
                    lo, hi = np.nan, np.nan

                stats_rows.append({
                    'interval': interval,
                    'compared_to': comp,
                    'metric': metric,
                    'n': n,
                    't_stat': t_stat,
                    't_p': t_p,
                    'wilcoxon_stat': w_stat,
                    'wilcoxon_p': w_p,
                    'cohen_d': d,
                    'mean_diff': np.nanmean(diffs) if n>0 else np.nan,
                    'mean_diff_ci_lo': lo,
                    'mean_diff_ci_hi': hi
                })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(OUT_DIR, 'model_comparison_stats.csv')
    stats_df.to_csv(stats_csv, index=False)
    print(f'Paired stats CSV written to {stats_csv}')

    # Append stats to the Markdown summary
    with open(out_md, 'a', encoding='utf-8') as f:
        f.write('\n\n## Paired statistical tests (M3 vs M1/M2)\n\n')
        if stats_df.empty:
            f.write('No paired statistics could be computed.\n')
        else:
            for interval in sorted(stats_df['interval'].unique()):
                f.write(f'### Interval: {interval}\n\n')
                sub = stats_df[stats_df['interval'] == interval]
                f.write(sub.to_markdown(index=False))
                f.write('\n\n')

if __name__ == '__main__':
    main()
