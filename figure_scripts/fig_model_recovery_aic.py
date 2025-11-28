"""
Figure: Model Recovery Results (AIC Comparison)

Creates a grouped bar chart showing AIC scores for each model fitted to each generator's data.
Demonstrates that M3 is necessary to explain M3-generated data while simpler models win on simpler data.

Usage:
    python figure_scripts/fig_model_recovery_aic.py [--results-dir RESULTS_DIR] [--output OUTPUT]

Input:
    - results/model_recovery/run_YYYYMMDD_HHMMSS/confusion/aic_mean.csv
    - results/model_recovery/run_YYYYMMDD_HHMMSS/confusion/aic_se.csv

Output:
    - figures/model_recovery_aic.png (or specified output path)
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_confusion_matrices(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AIC mean and SE confusion matrices from results directory."""
    aic_mean = pd.read_csv(os.path.join(results_dir, 'confusion', 'aic_mean.csv'), index_col=0)
    aic_se = pd.read_csv(os.path.join(results_dir, 'confusion', 'aic_se.csv'), index_col=0)
    return aic_mean, aic_se


def create_model_recovery_figure(aic_mean: pd.DataFrame, aic_se: pd.DataFrame, output_path: str):
    """
    Create grouped bar chart showing AIC comparison across generators.
    
    Layout:
    - X-axis: Three groups (M1 Data, M2 Data, M3 Data)
    - Y-axis: AIC (lower is better)
    - Three bars per group: M1 fitted, M2 fitted, M3 fitted
    - Error bars: ±SE across runs
    - Visual marking: Winner highlighted with gold edge
    """
    
    # Filter to only M1, M2, M3 generators (skip RL baselines)
    generators = ['M1', 'M2', 'M3']
    models = ['M1', 'M2', 'M3']
    
    # Extract data
    aic_values = aic_mean.loc[generators, models].values
    aic_errors = aic_se.loc[generators, models].values
    
    # Setup figure with clean styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors for each model - distinctive, colorblind-friendly palette
    colors = {
        'M1': '#4C72B0',  # Steel blue
        'M2': '#55A868',  # Sage green  
        'M3': '#C44E52',  # Muted red
    }
    
    # Bar positioning
    x = np.arange(len(generators))
    width = 0.25
    offsets = [-width, 0, width]
    
    # Find winners for each generator (lowest AIC)
    winners = aic_values.argmin(axis=1)
    
    # Plot bars for each model
    bars_list = []
    for i, model in enumerate(models):
        values = aic_values[:, i]
        errors = aic_errors[:, i]
        
        bars = ax.bar(
            x + offsets[i], 
            values, 
            width,
            label=f'{model} fitted',
            color=colors[model],
            edgecolor='black',
            linewidth=0.8,
            yerr=errors,
            capsize=4,
            error_kw={'linewidth': 1.5, 'capthick': 1.5}
        )
        bars_list.append(bars)
    
    # Highlight winners with gold edge and star marker (only on the bars, not legend)
    for gen_idx, winner_idx in enumerate(winners):
        winner_bar = bars_list[winner_idx][gen_idx]
        winner_bar.set_edgecolor('#FFD700')  # Gold
        winner_bar.set_linewidth(3)
        
        # Add star above winning bar
        bar_height = aic_values[gen_idx, winner_idx]
        bar_error = aic_errors[gen_idx, winner_idx]
        ax.annotate(
            '★', 
            xy=(x[gen_idx] + offsets[winner_idx], bar_height + bar_error + 8),
            ha='center', va='bottom',
            fontsize=18,
            color='#FFD700'
        )
    
    # Reset edge colors for legend handles (so legend doesn't show gold edges)
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=colors[m], edgecolor='black', linewidth=0.8) 
                      for m in models]
    legend_labels = [f'{m} fitted' for m in models]
    
    # Formatting
    ax.set_xlabel('Data Generator', fontsize=14, fontweight='bold')
    ax.set_ylabel('AIC', fontsize=14, fontweight='bold')
    ax.set_title('Model Recovery: AIC Comparison\n', fontsize=16, fontweight='bold')
    
    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g} Data' for g in generators], fontsize=12)
    
    # Y-axis formatting
    ax.tick_params(axis='y', labelsize=11)
    
    # Legend with clean handles (no gold highlighting)
    legend = ax.legend(
        legend_handles,
        legend_labels,
        title='Fitted Model',
        title_fontsize=12,
        fontsize=11,
        loc='upper left',
        framealpha=0.95
    )
    
    # Set y-axis limits with some padding
    y_max = aic_values.max() + aic_errors.max() + 30
    ax.set_ylim(0, y_max)
    
    # Add horizontal reference line at M3's winning AIC to emphasize the gap
    m3_on_m3 = aic_values[2, 2]  # M3 generator, M3 fitted
    ax.axhline(y=m3_on_m3, color=colors['M3'], linestyle='--', alpha=0.3, linewidth=1)
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()
    
    return fig


def print_summary_stats(aic_mean: pd.DataFrame, aic_se: pd.DataFrame):
    """Print summary statistics for the figure."""
    generators = ['M1', 'M2', 'M3']
    models = ['M1', 'M2', 'M3']
    
    print("\n" + "="*60)
    print("MODEL RECOVERY AIC SUMMARY")
    print("="*60)
    
    for gen in generators:
        print(f"\n{gen} Data:")
        values = aic_mean.loc[gen, models]
        errors = aic_se.loc[gen, models]
        winner = values.idxmin()
        
        for model in models:
            marker = " ★" if model == winner else ""
            print(f"  {model} fitted: {values[model]:.1f} ± {errors[model]:.1f}{marker}")
        
        # Calculate margin of victory
        sorted_vals = values.sort_values()
        margin = sorted_vals.iloc[1] - sorted_vals.iloc[0]
        print(f"  → Winner: {winner} (margin: {margin:.1f} AIC points)")
    
    # M3 vs others on M3 data
    m3_on_m3 = aic_mean.loc['M3', 'M3']
    m1_on_m3 = aic_mean.loc['M3', 'M1']
    m2_on_m3 = aic_mean.loc['M3', 'M2']
    
    print("\n" + "-"*60)
    print("KEY FINDING: M3 necessary for M3 data")
    print(f"  M3 vs M1 on M3 data: {m1_on_m3 - m3_on_m3:.1f} AIC points better")
    print(f"  M3 vs M2 on M3 data: {m2_on_m3 - m3_on_m3:.1f} AIC points better")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate Model Recovery AIC Figure')
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results/model_recovery/run_20251125_142037',
        help='Path to model recovery results directory'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='figures/model_recovery_aic.png',
        help='Output path for figure'
    )
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.results_dir}")
    aic_mean, aic_se = load_confusion_matrices(args.results_dir)
    
    # Print summary
    print_summary_stats(aic_mean, aic_se)
    
    # Create figure
    create_model_recovery_figure(aic_mean, aic_se, args.output)


if __name__ == '__main__':
    main()

