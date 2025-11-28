"""
Figure: Mechanistic Validation of Profile-Based Value Model (M3)

Creates a multi-panel figure demonstrating HOW M3 works and proving its computational
mechanism is qualitatively different from M1/M2.

Panels:
1. Profile Recruitment Dynamics - w₀ and w₁ aligned to context reversals
2. Emergent Precision Dynamics - γₜ comparison across models  
3. Action Strategy Modulation - Hint-seeking rate over time
4. Micro-Reversal Behavior - Zoomed view of volatile context period

Usage:
    python figure_scripts/fig_mechanistic_analysis.py [--results-dir RESULTS_DIR] [--output OUTPUT]

Input:
    - results/model_recovery/run_YYYYMMDD_HHMMSS/trial_level/gen_M3/model_*/run_*.csv

Output:
    - figures/mechanistic_analysis.png
"""

import argparse
import os
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d
from pathlib import Path


def load_trial_data(results_dir: str, generator: str = 'M3') -> dict:
    """
    Load trial-level data for all models fitted to a specific generator's data.
    
    Returns dict: {model_name: list of DataFrames (one per run)}
    """
    data = {}
    
    # Required columns
    required_cols = ['t', 'true_context', 'belief_context', 'gamma', 'predicted_action', 'is_reversal']
    
    for model in ['M1', 'M2', 'M3']:
        model_dir = os.path.join(results_dir, 'trial_level', f'gen_{generator}', f'model_{model}')
        
        if not os.path.exists(model_dir):
            print(f"Warning: {model_dir} not found")
            continue
            
        run_files = sorted(glob.glob(os.path.join(model_dir, 'run_*.csv')))
        
        if not run_files:
            print(f"Warning: No run files found in {model_dir}")
            continue
        
        dfs = []
        for f in run_files:
            try:
                df = pd.read_csv(f)
                
                # Check required columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing columns in {f}: {missing_cols}")
                    continue
                
                # Parse belief_context from string to array (robust parsing)
                def parse_belief(x, idx):
                    """Parse belief_context and extract index idx."""
                    if pd.isna(x):
                        return 0.0
                    if isinstance(x, str):
                        try:
                            parsed = ast.literal_eval(x)
                            return float(parsed[idx]) if isinstance(parsed, (list, tuple, np.ndarray)) and len(parsed) > idx else 0.0
                        except (ValueError, SyntaxError, IndexError):
                            return 0.0
                    elif isinstance(x, (list, tuple, np.ndarray)):
                        return float(x[idx]) if len(x) > idx else 0.0
                    else:
                        return 0.0
                
                df['belief_volatile'] = df['belief_context'].apply(lambda x: parse_belief(x, 0))
                df['belief_stable'] = df['belief_context'].apply(lambda x: parse_belief(x, 1))
                
                # Verify beliefs sum to ~1.0 (with tolerance)
                belief_sum = df['belief_volatile'] + df['belief_stable']
                if not np.allclose(belief_sum, 1.0, atol=0.1):
                    print(f"Warning: Beliefs in {f} don't sum to 1.0 (range: {belief_sum.min():.3f} - {belief_sum.max():.3f})")
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
        
        if dfs:
            data[model] = dfs
    
    return data


def compute_reversal_aligned_data_single_run(df: pd.DataFrame, window: int = 10) -> dict:
    """
    Align data to context reversals for a SINGLE run.
    
    Returns dict with keys: 'volatile_to_stable', 'stable_to_volatile'
    Each contains list of individual reversal segments (not averaged).
    """
    results = {
        'volatile_to_stable': {'profile_0': [], 'profile_1': [], 'gamma': [], 'hint_rate': []},
        'stable_to_volatile': {'profile_0': [], 'profile_1': [], 'gamma': [], 'hint_rate': []}
    }
    
    # Find reversal points
    reversal_indices = df[df['is_reversal'] == 1].index.tolist()
    
    for rev_idx in reversal_indices:
        # Skip if too close to boundaries
        if rev_idx < window or rev_idx + window >= len(df):
            continue
        
        # Get context before and after reversal
        context_before = df.loc[rev_idx - 1, 'true_context'] if rev_idx > 0 else None
        context_after = df.loc[rev_idx, 'true_context']
        
        # Determine reversal type
        if context_before == 'volatile' and context_after == 'stable':
            key = 'volatile_to_stable'
        elif context_before == 'stable' and context_after == 'volatile':
            key = 'stable_to_volatile'
        else:
            continue
        
        # Extract window around reversal
        start = rev_idx - window
        end = rev_idx + window
        
        segment = df.loc[start:end-1].copy()
        
        if len(segment) == 2 * window:
            results[key]['profile_0'].append(segment['belief_volatile'].values)
            results[key]['profile_1'].append(segment['belief_stable'].values)
            results[key]['gamma'].append(segment['gamma'].values)
            results[key]['hint_rate'].append(segment['hint_flag'].values)
    
    return results


def compute_context_conditional_stats_single_run(df: pd.DataFrame) -> dict:
    """
    Compute statistics conditional on true context for a SINGLE run.
    
    Note: hint_rate is computed from predicted_action (model's choice), 
    NOT hint_flag (which is the generator's choice).
    """
    vol_mask = df['true_context'] == 'volatile'
    stab_mask = df['true_context'] == 'stable'
    
    volatile_gamma = df.loc[vol_mask, 'gamma'].values
    stable_gamma = df.loc[stab_mask, 'gamma'].values
    
    # Compute hint rate from predicted_action (model's predicted action)
    # hint_flag is from the generator, not the fitted model!
    def is_hint_action(x):
        """Check if action is a hint action."""
        if pd.isna(x):
            return 0
        x_str = str(x).lower()
        return 1 if 'hint' in x_str or x_str == 'act_hint' else 0
    
    volatile_pred_hint = df.loc[vol_mask, 'predicted_action'].apply(is_hint_action).values
    stable_pred_hint = df.loc[stab_mask, 'predicted_action'].apply(is_hint_action).values
    
    return {
        'volatile': {
            'gamma_mean': np.mean(volatile_gamma) if len(volatile_gamma) > 0 else 0,
            'gamma_se': np.std(volatile_gamma) / np.sqrt(len(volatile_gamma)) if len(volatile_gamma) > 0 else 0,
            'hint_rate': np.mean(volatile_pred_hint) if len(volatile_pred_hint) > 0 else 0,
            'hint_se': np.std(volatile_pred_hint) / np.sqrt(len(volatile_pred_hint)) if len(volatile_pred_hint) > 0 else 0
        },
        'stable': {
            'gamma_mean': np.mean(stable_gamma) if len(stable_gamma) > 0 else 0,
            'gamma_se': np.std(stable_gamma) / np.sqrt(len(stable_gamma)) if len(stable_gamma) > 0 else 0,
            'hint_rate': np.mean(stable_pred_hint) if len(stable_pred_hint) > 0 else 0,
            'hint_se': np.std(stable_pred_hint) / np.sqrt(len(stable_pred_hint)) if len(stable_pred_hint) > 0 else 0
        }
    }


def create_mechanistic_figure(data: dict, output_path: str, run_idx: int = 0):
    """
    Create multi-panel mechanistic analysis figure using a SINGLE run.
    
    Parameters:
    -----------
    data : dict
        Dictionary with model names as keys and list of DataFrames as values
    output_path : str
        Where to save the figure
    run_idx : int
        Which run to use (default: 0)
    """
    # Setup figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = {
        'M1': '#4C72B0',
        'M2': '#55A868', 
        'M3': '#C44E52',
        'volatile': '#E74C3C',
        'stable': '#3498DB',
        'profile_0': '#E74C3C',  # Volatile profile
        'profile_1': '#3498DB',  # Stable profile
    }
    
    # Use single run for all analyses
    window = 10  # Reduced from 40 to 10
    x_aligned = np.arange(-window, window)
    
    # Get single run data for M3
    m3_df = data['M3'][run_idx]
    m3_aligned = compute_reversal_aligned_data_single_run(m3_df, window=window)
    
    # =========================================================================
    # Panel A: Profile Recruitment Dynamics (Volatile → Stable)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    v2s_data = m3_aligned['volatile_to_stable']
    if len(v2s_data['profile_0']) > 0:
        # Plot individual reversal events as semi-transparent lines
        for i, (p0, p1) in enumerate(zip(v2s_data['profile_0'], v2s_data['profile_1'])):
            alpha = 0.4 if len(v2s_data['profile_0']) > 1 else 1.0
            lw = 1.5 if len(v2s_data['profile_0']) > 1 else 2
            label_p0 = 'w₀ (Volatile profile)' if i == 0 else None
            label_p1 = 'w₁ (Stable profile)' if i == 0 else None
            ax_a.plot(x_aligned, p0, color=colors['profile_0'], linewidth=lw, 
                      alpha=alpha, label=label_p0)
            ax_a.plot(x_aligned, p1, color=colors['profile_1'], linewidth=lw,
                      alpha=alpha, label=label_p1)
        
        ax_a.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Reversal')
        n = len(v2s_data['profile_0'])
        ax_a.set_title(f'A. Profile Recruitment: Volatile → Stable\n(n={n} reversal{"s" if n > 1 else ""}, run {run_idx})', 
                       fontsize=12, fontweight='bold')
        
        # Compute y-axis range from data
        all_vals = np.concatenate(v2s_data['profile_0'] + v2s_data['profile_1'])
        y_min, y_max = all_vals.min(), all_vals.max()
        y_margin = (y_max - y_min) * 0.1
        ax_a.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax_a.text(0.5, 0.5, 'No volatile→stable\nreversals found', ha='center', va='center',
                  transform=ax_a.transAxes)
        ax_a.set_title('A. Profile Recruitment: Volatile → Stable', fontsize=12, fontweight='bold')
    
    ax_a.set_xlabel('Trials relative to reversal', fontsize=11)
    ax_a.set_ylabel('Profile weight', fontsize=11)
    ax_a.legend(loc='best', fontsize=9)
    ax_a.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel B: Profile Recruitment Dynamics (Stable → Volatile)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    s2v_data = m3_aligned['stable_to_volatile']
    if len(s2v_data['profile_0']) > 0:
        # Plot individual reversal events as semi-transparent lines
        for i, (p0, p1) in enumerate(zip(s2v_data['profile_0'], s2v_data['profile_1'])):
            alpha = 0.4 if len(s2v_data['profile_0']) > 1 else 1.0
            lw = 1.5 if len(s2v_data['profile_0']) > 1 else 2
            label_p0 = 'w₀ (Volatile profile)' if i == 0 else None
            label_p1 = 'w₁ (Stable profile)' if i == 0 else None
            ax_b.plot(x_aligned, p0, color=colors['profile_0'], linewidth=lw,
                      alpha=alpha, label=label_p0)
            ax_b.plot(x_aligned, p1, color=colors['profile_1'], linewidth=lw,
                      alpha=alpha, label=label_p1)
        
        ax_b.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Reversal')
        n = len(s2v_data['profile_0'])
        ax_b.set_title(f'B. Profile Recruitment: Stable → Volatile\n(n={n} reversal{"s" if n > 1 else ""}, run {run_idx})',
                       fontsize=12, fontweight='bold')
        
        # Compute y-axis range from data
        all_vals = np.concatenate(s2v_data['profile_0'] + s2v_data['profile_1'])
        y_min, y_max = all_vals.min(), all_vals.max()
        y_margin = (y_max - y_min) * 0.1
        ax_b.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax_b.text(0.5, 0.5, 'No stable→volatile\nreversals found', ha='center', va='center',
                  transform=ax_b.transAxes)
        ax_b.set_title('B. Profile Recruitment: Stable → Volatile', fontsize=12, fontweight='bold')
    
    ax_b.set_xlabel('Trials relative to reversal', fontsize=11)
    ax_b.set_ylabel('Profile weight', fontsize=11)
    ax_b.legend(loc='best', fontsize=9)
    ax_b.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel C: Emergent Precision Dynamics (γₜ comparison)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Compute reversal-aligned gamma for all models using SINGLE run
    for model, model_data in data.items():
        model_df = model_data[run_idx]
        aligned = compute_reversal_aligned_data_single_run(model_df, window=window)
        
        # Collect gamma traces from both reversal types
        gamma_traces = []
        for rev_type in ['volatile_to_stable', 'stable_to_volatile']:
            gamma_traces.extend(aligned[rev_type]['gamma'])
        
        if gamma_traces:
            # Plot individual traces with low alpha, mean with high alpha
            for i, trace in enumerate(gamma_traces):
                ax_c.plot(x_aligned, trace, color=colors[model], linewidth=1, alpha=0.3)
            
            # Plot mean
            mean_gamma = np.mean(gamma_traces, axis=0)
            ax_c.plot(x_aligned, mean_gamma, color=colors[model], linewidth=2.5,
                      label=f'{model}')
    
    ax_c.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax_c.set_xlabel('Trials relative to reversal', fontsize=11)
    ax_c.set_ylabel('Effective γₜ', fontsize=11)
    ax_c.set_title(f'C. Precision Dynamics Around Reversals\n(run {run_idx})', fontsize=12, fontweight='bold')
    ax_c.legend(loc='best', fontsize=9)
    ax_c.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel D: Context-Conditional Gamma Comparison
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    
    x_pos = np.arange(3)
    width = 0.35
    
    volatile_gammas = []
    stable_gammas = []
    volatile_se = []
    stable_se = []
    
    for model in ['M1', 'M2', 'M3']:
        model_df = data[model][run_idx]
        stats = compute_context_conditional_stats_single_run(model_df)
        volatile_gammas.append(stats['volatile']['gamma_mean'])
        stable_gammas.append(stats['stable']['gamma_mean'])
        volatile_se.append(stats['volatile']['gamma_se'])
        stable_se.append(stats['stable']['gamma_se'])
    
    bars1 = ax_d.bar(x_pos - width/2, volatile_gammas, width, yerr=volatile_se,
                     label='Volatile context', color=colors['volatile'], capsize=4)
    bars2 = ax_d.bar(x_pos + width/2, stable_gammas, width, yerr=stable_se,
                     label='Stable context', color=colors['stable'], capsize=4)
    
    ax_d.set_xlabel('Model', fontsize=11)
    ax_d.set_ylabel('Mean γ', fontsize=11)
    ax_d.set_title(f'D. Context-Conditional Precision (run {run_idx})', fontsize=12, fontweight='bold')
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(['M1', 'M2', 'M3'])
    ax_d.legend(loc='upper left', fontsize=9)
    ax_d.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for M3's context-dependency
    if stable_gammas[2] > volatile_gammas[2]:
        ax_d.annotate('Context-dependent\nbaselines', xy=(2, stable_gammas[2]),
                      xytext=(2.3, stable_gammas[2] * 0.8),
                      fontsize=9, ha='left',
                      arrowprops=dict(arrowstyle='->', color='gray'))
    
    # =========================================================================
    # Panel E: Context-Conditional Hint-Seeking Rate
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    
    volatile_hints = []
    stable_hints = []
    volatile_hint_se = []
    stable_hint_se = []
    
    for model in ['M1', 'M2', 'M3']:
        model_df = data[model][run_idx]
        stats = compute_context_conditional_stats_single_run(model_df)
        volatile_hints.append(stats['volatile']['hint_rate'])
        stable_hints.append(stats['stable']['hint_rate'])
        volatile_hint_se.append(stats['volatile']['hint_se'])
        stable_hint_se.append(stats['stable']['hint_se'])
    
    bars1 = ax_e.bar(x_pos - width/2, volatile_hints, width, yerr=volatile_hint_se,
                     label='Volatile context', color=colors['volatile'], capsize=4)
    bars2 = ax_e.bar(x_pos + width/2, stable_hints, width, yerr=stable_hint_se,
                     label='Stable context', color=colors['stable'], capsize=4)
    
    ax_e.set_xlabel('Model', fontsize=11)
    ax_e.set_ylabel('Hint-seeking rate', fontsize=11)
    ax_e.set_title(f'E. Context-Conditional Hint-Seeking (run {run_idx})', fontsize=12, fontweight='bold')
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(['M1', 'M2', 'M3'])
    ax_e.legend(loc='upper right', fontsize=9)
    ax_e.grid(True, alpha=0.3, axis='y')
    ax_e.set_ylim(0, 1)
    
    # =========================================================================
    # Panel F: Micro-Reversal Behavior (Zoomed Volatile Period)
    # =========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    
    # Use same run
    df = data['M3'][run_idx]
    
    # Find a volatile period with at least 50 trials
    volatile_mask = df['true_context'] == 'volatile'
    volatile_indices = df[volatile_mask].index.tolist()
    
    if len(volatile_indices) >= 50:
        # Take trials 10-60 of volatile period to show micro-reversals
        start_idx = volatile_indices[10] if len(volatile_indices) > 10 else volatile_indices[0]
        end_idx = min(start_idx + 50, volatile_indices[-1])
        
        segment = df.loc[start_idx:end_idx].copy()
        x_seg = np.arange(len(segment))
        
        # Plot profile weights
        p0_vals = segment['belief_volatile'].values
        p1_vals = segment['belief_stable'].values
        
        ax_f.plot(x_seg, p0_vals, color=colors['profile_0'], 
                  linewidth=2, label='w₀ (Volatile profile)')
        ax_f.plot(x_seg, p1_vals, color=colors['profile_1'],
                  linewidth=2, label='w₁ (Stable profile)')
        
        # Mark micro-reversals (arm switches every 10 trials)
        for i, (idx, row) in enumerate(segment.iterrows()):
            if i > 0 and i % 10 == 0:
                ax_f.axvline(i, color='gray', linestyle=':', alpha=0.5)
        
        ax_f.set_xlabel('Trial (within volatile context)', fontsize=11)
        ax_f.set_ylabel('Profile weight', fontsize=11)
        ax_f.set_title(f'F. Profile Stability During Micro-Reversals\n(run {run_idx})',
                       fontsize=12, fontweight='bold')
        ax_f.legend(loc='best', fontsize=9)
        
        # Zoom y-axis to show variation
        all_vals = np.concatenate([p0_vals, p1_vals])
        y_min, y_max = all_vals.min(), all_vals.max()
        y_margin = (y_max - y_min) * 0.15
        ax_f.set_ylim(y_min - y_margin, y_max + y_margin)
        ax_f.grid(True, alpha=0.3)
        
        # Add annotation
        ax_f.annotate('Profile tracks context,\nnot arm switches', 
                      xy=(25, (y_min + y_max)/2), fontsize=9, ha='center',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    else:
        ax_f.text(0.5, 0.5, 'Insufficient volatile\ntrials for analysis', 
                  ha='center', va='center', transform=ax_f.transAxes)
        ax_f.set_title('F. Profile Stability During Micro-Reversals',
                       fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Final formatting
    # =========================================================================
    fig.suptitle('Mechanistic Validation: Profile-Based Value Model (M3)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.close()


def print_mechanistic_summary(data: dict, run_idx: int = 0):
    """Print key mechanistic findings for a single run."""
    print("\n" + "="*70)
    print(f"MECHANISTIC ANALYSIS SUMMARY (Run {run_idx})")
    print("="*70)
    
    for model in ['M1', 'M2', 'M3']:
        model_df = data[model][run_idx]
        stats = compute_context_conditional_stats_single_run(model_df)
        print(f"\n{model}:")
        print(f"  Volatile context: γ={stats['volatile']['gamma_mean']:.3f}, "
              f"hint_rate={stats['volatile']['hint_rate']:.3f}")
        print(f"  Stable context:   γ={stats['stable']['gamma_mean']:.3f}, "
              f"hint_rate={stats['stable']['hint_rate']:.3f}")
        
        # Context difference
        gamma_diff = stats['stable']['gamma_mean'] - stats['volatile']['gamma_mean']
        hint_diff = stats['volatile']['hint_rate'] - stats['stable']['hint_rate']
        print(f"  Δγ (stable-volatile): {gamma_diff:+.3f}")
        print(f"  Δhint (volatile-stable): {hint_diff:+.3f}")
    
    print("\n" + "-"*70)
    print("KEY MECHANISTIC SIGNATURES:")
    
    m3_stats = compute_context_conditional_stats_single_run(data['M3'][run_idx])
    m1_stats = compute_context_conditional_stats_single_run(data['M1'][run_idx])
    
    # Check for M3's unique signature
    m3_gamma_diff = m3_stats['stable']['gamma_mean'] - m3_stats['volatile']['gamma_mean']
    m1_gamma_diff = m1_stats['stable']['gamma_mean'] - m1_stats['volatile']['gamma_mean']
    
    print(f"\n1. Context-dependent precision baselines:")
    print(f"   M3 shows {m3_gamma_diff:.2f} higher γ in stable vs volatile")
    print(f"   M1 shows {m1_gamma_diff:.2f} (should be ~0 for static model)")
    
    m3_hint_diff = m3_stats['volatile']['hint_rate'] - m3_stats['stable']['hint_rate']
    print(f"\n2. Context-appropriate information seeking:")
    print(f"   M3 hint rate {m3_hint_diff:+.2%} higher in volatile context")
    
    print("="*70 + "\n")


def create_single_panel(data: dict, panel: str, output_path: str, run_idx: int = 0):
    """
    Create a single panel figure for debugging.
    
    Parameters:
    -----------
    panel : str
        One of 'A', 'B', 'C', 'D', 'E', 'F'
    """
    # Color scheme
    colors = {
        'M1': '#4C72B0',
        'M2': '#55A868', 
        'M3': '#C44E52',
        'volatile': '#E74C3C',
        'stable': '#3498DB',
        'profile_0': '#E74C3C',
        'profile_1': '#3498DB',
    }
    
    window = 10
    x_aligned = np.arange(-window, window)
    
    # Get single run data for M3
    m3_df = data['M3'][run_idx]
    m3_aligned = compute_reversal_aligned_data_single_run(m3_df, window=window)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if panel.upper() == 'A':
        # Panel A: Profile Recruitment Dynamics (Volatile → Stable)
        v2s_data = m3_aligned['volatile_to_stable']
        
        print(f"\n=== PANEL A DEBUG (Volatile → Stable) ===")
        print(f"Number of reversals found: {len(v2s_data['profile_0'])}")
        
        if len(v2s_data['profile_0']) > 0:
            for i, (p0, p1) in enumerate(zip(v2s_data['profile_0'], v2s_data['profile_1'])):
                print(f"  Reversal {i}: p0 range [{p0.min():.3f}, {p0.max():.3f}], p1 range [{p1.min():.3f}, {p1.max():.3f}]")
                alpha = 0.6 if len(v2s_data['profile_0']) > 1 else 1.0
                label_p0 = 'w₀ (Volatile profile)' if i == 0 else None
                label_p1 = 'w₁ (Stable profile)' if i == 0 else None
                ax.plot(x_aligned, p0, color=colors['profile_0'], linewidth=2, alpha=alpha, label=label_p0)
                ax.plot(x_aligned, p1, color=colors['profile_1'], linewidth=2, alpha=alpha, label=label_p1)
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Reversal')
            n = len(v2s_data['profile_0'])
            ax.set_title(f'A. Profile Recruitment: Volatile → Stable\n(n={n} reversal{"s" if n > 1 else ""}, run {run_idx})')
            
            # Auto y-axis from data
            all_vals = np.concatenate(v2s_data['profile_0'] + v2s_data['profile_1'])
            y_min, y_max = all_vals.min(), all_vals.max()
            margin = max(0.05, (y_max - y_min) * 0.1)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.text(0.5, 0.5, 'No volatile→stable reversals found', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Trials relative to reversal')
        ax.set_ylabel('Profile weight (belief about context)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    elif panel.upper() == 'B':
        # Panel B: Profile Recruitment Dynamics (Stable → Volatile)
        s2v_data = m3_aligned['stable_to_volatile']
        
        print(f"\n=== PANEL B DEBUG (Stable → Volatile) ===")
        print(f"Number of reversals found: {len(s2v_data['profile_0'])}")
        
        if len(s2v_data['profile_0']) > 0:
            for i, (p0, p1) in enumerate(zip(s2v_data['profile_0'], s2v_data['profile_1'])):
                print(f"  Reversal {i}: p0 range [{p0.min():.3f}, {p0.max():.3f}], p1 range [{p1.min():.3f}, {p1.max():.3f}]")
                alpha = 0.6 if len(s2v_data['profile_0']) > 1 else 1.0
                label_p0 = 'w₀ (Volatile profile)' if i == 0 else None
                label_p1 = 'w₁ (Stable profile)' if i == 0 else None
                ax.plot(x_aligned, p0, color=colors['profile_0'], linewidth=2, alpha=alpha, label=label_p0)
                ax.plot(x_aligned, p1, color=colors['profile_1'], linewidth=2, alpha=alpha, label=label_p1)
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Reversal')
            n = len(s2v_data['profile_0'])
            ax.set_title(f'B. Profile Recruitment: Stable → Volatile\n(n={n} reversal{"s" if n > 1 else ""}, run {run_idx})')
            
            all_vals = np.concatenate(s2v_data['profile_0'] + s2v_data['profile_1'])
            y_min, y_max = all_vals.min(), all_vals.max()
            margin = max(0.05, (y_max - y_min) * 0.1)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.text(0.5, 0.5, 'No stable→volatile reversals found', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Trials relative to reversal')
        ax.set_ylabel('Profile weight (belief about context)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    elif panel.upper() == 'C':
        # Panel C: Emergent Precision Dynamics
        print(f"\n=== PANEL C DEBUG (Precision Dynamics) ===")
        
        for model in ['M1', 'M2', 'M3']:
            model_df = data[model][run_idx]
            aligned = compute_reversal_aligned_data_single_run(model_df, window=window)
            
            gamma_traces = []
            for rev_type in ['volatile_to_stable', 'stable_to_volatile']:
                gamma_traces.extend(aligned[rev_type]['gamma'])
            
            if gamma_traces:
                mean_gamma = np.mean(gamma_traces, axis=0)
                print(f"  {model}: gamma range [{mean_gamma.min():.3f}, {mean_gamma.max():.3f}], mean={mean_gamma.mean():.3f}")
                
                for trace in gamma_traces:
                    ax.plot(x_aligned, trace, color=colors[model], linewidth=1, alpha=0.3)
                ax.plot(x_aligned, mean_gamma, color=colors[model], linewidth=2.5, label=f'{model}')
            else:
                print(f"  {model}: No gamma traces found")
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Trials relative to reversal')
        ax.set_ylabel('Effective γₜ')
        ax.set_title(f'C. Precision Dynamics Around Reversals (run {run_idx})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    elif panel.upper() == 'D':
        # Panel D: Context-Conditional Gamma
        print(f"\n=== PANEL D DEBUG (Context-Conditional Gamma) ===")
        
        x_pos = np.arange(3)
        width = 0.35
        volatile_gammas, stable_gammas = [], []
        volatile_se, stable_se = [], []
        
        for model in ['M1', 'M2', 'M3']:
            model_df = data[model][run_idx]
            stats = compute_context_conditional_stats_single_run(model_df)
            print(f"  {model}: volatile γ={stats['volatile']['gamma_mean']:.3f}, stable γ={stats['stable']['gamma_mean']:.3f}")
            volatile_gammas.append(stats['volatile']['gamma_mean'])
            stable_gammas.append(stats['stable']['gamma_mean'])
            volatile_se.append(stats['volatile']['gamma_se'])
            stable_se.append(stats['stable']['gamma_se'])
        
        ax.bar(x_pos - width/2, volatile_gammas, width, yerr=volatile_se, label='Volatile context', color=colors['volatile'], capsize=4)
        ax.bar(x_pos + width/2, stable_gammas, width, yerr=stable_se, label='Stable context', color=colors['stable'], capsize=4)
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean γ')
        ax.set_title(f'D. Context-Conditional Precision (run {run_idx})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['M1', 'M2', 'M3'])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
    elif panel.upper() == 'E':
        # Panel E: Context-Conditional Hint-Seeking
        print(f"\n=== PANEL E DEBUG (Context-Conditional Hint-Seeking) ===")
        
        x_pos = np.arange(3)
        width = 0.35
        volatile_hints, stable_hints = [], []
        volatile_hint_se, stable_hint_se = [], []
        
        for model in ['M1', 'M2', 'M3']:
            model_df = data[model][run_idx]
            stats = compute_context_conditional_stats_single_run(model_df)
            print(f"  {model}: volatile hint={stats['volatile']['hint_rate']:.3f}, stable hint={stats['stable']['hint_rate']:.3f}")
            volatile_hints.append(stats['volatile']['hint_rate'])
            stable_hints.append(stats['stable']['hint_rate'])
            volatile_hint_se.append(stats['volatile']['hint_se'])
            stable_hint_se.append(stats['stable']['hint_se'])
        
        ax.bar(x_pos - width/2, volatile_hints, width, yerr=volatile_hint_se, label='Volatile context', color=colors['volatile'], capsize=4)
        ax.bar(x_pos + width/2, stable_hints, width, yerr=stable_hint_se, label='Stable context', color=colors['stable'], capsize=4)
        ax.set_xlabel('Model')
        ax.set_ylabel('Hint-seeking rate')
        ax.set_title(f'E. Context-Conditional Hint-Seeking (run {run_idx})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['M1', 'M2', 'M3'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
    elif panel.upper() == 'F':
        # Panel F: Micro-Reversal Behavior
        print(f"\n=== PANEL F DEBUG (Micro-Reversal Behavior) ===")
        
        df = data['M3'][run_idx]
        volatile_mask = df['true_context'] == 'volatile'
        volatile_indices = df[volatile_mask].index.tolist()
        
        print(f"  Total volatile trials: {len(volatile_indices)}")
        
        if len(volatile_indices) >= 50:
            start_idx = volatile_indices[10] if len(volatile_indices) > 10 else volatile_indices[0]
            end_idx = min(start_idx + 50, volatile_indices[-1])
            
            segment = df.loc[start_idx:end_idx].copy()
            x_seg = np.arange(len(segment))
            
            p0_vals = segment['belief_volatile'].values
            p1_vals = segment['belief_stable'].values
            
            print(f"  Segment length: {len(segment)}")
            print(f"  p0 (volatile belief) range: [{p0_vals.min():.3f}, {p0_vals.max():.3f}]")
            print(f"  p1 (stable belief) range: [{p1_vals.min():.3f}, {p1_vals.max():.3f}]")
            
            ax.plot(x_seg, p0_vals, color=colors['profile_0'], linewidth=2, label='w₀ (Volatile profile)')
            ax.plot(x_seg, p1_vals, color=colors['profile_1'], linewidth=2, label='w₁ (Stable profile)')
            
            for i in range(len(segment)):
                if i > 0 and i % 10 == 0:
                    ax.axvline(i, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Trial (within volatile context)')
            ax.set_ylabel('Profile weight')
            ax.set_title(f'F. Profile Stability During Micro-Reversals (run {run_idx})')
            ax.legend(loc='best')
            
            all_vals = np.concatenate([p0_vals, p1_vals])
            y_min, y_max = all_vals.min(), all_vals.max()
            margin = max(0.05, (y_max - y_min) * 0.15)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient volatile trials', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Determine output filename
    base, ext = os.path.splitext(output_path)
    panel_output = f"{base}_panel_{panel.upper()}{ext}"
    
    output_dir = os.path.dirname(panel_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(panel_output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPanel {panel.upper()} saved to: {panel_output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Mechanistic Analysis Figure')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/model_recovery/run_20251125_142037',
        help='Path to model recovery results directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/mechanistic_analysis.png',
        help='Output path for figure'
    )
    parser.add_argument(
        '--generator',
        type=str,
        default='M3',
        help='Which generator data to analyze (default: M3)'
    )
    parser.add_argument(
        '--panel',
        type=str,
        default='all',
        choices=['A', 'B', 'C', 'D', 'E', 'F', 'all', 'a', 'b', 'c', 'd', 'e', 'f', 'All', 'ALL'],
        help='Which panel to generate (A-F or "all")'
    )
    args = parser.parse_args()
    
    print(f"Loading trial-level data from: {args.results_dir}")
    data = load_trial_data(args.results_dir, generator=args.generator)
    
    if not data:
        print("ERROR: No data loaded. Check results directory.")
        return
    
    print(f"Loaded data for models: {list(data.keys())}")
    for model, dfs in data.items():
        print(f"  {model}: {len(dfs)} runs, {len(dfs[0]) if dfs else 0} trials each")
    
    # Print summary statistics
    print_mechanistic_summary(data, run_idx=0)
    
    # Create figure(s)
    if args.panel.lower() == 'all':
        create_mechanistic_figure(data, args.output, run_idx=0)
    else:
        create_single_panel(data, args.panel, args.output, run_idx=0)


if __name__ == '__main__':
    main()

