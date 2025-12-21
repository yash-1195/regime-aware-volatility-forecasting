"""
Plotting helpers.

"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def savefig(fig, outpath: str | Path, dpi: int = 200) -> None:
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def set_mpl_defaults() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
    })


def plot_regimes_over_time(
    data: pd.DataFrame,
    value_col: str,
    regime_col: str,
    title: str = "Volatility Regimes Over Time",
    ylabel: str = "Squared Returns",
    figsize: Tuple[int, int] = (14, 6),
    regime_colors: Optional[Dict] = None,
    show_thresholds: bool = False,
    thresholds: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    if regime_colors is None:
        regime_colors = {
            'Low':'#2ecc71',
            'Medium': '#f39c12',
            'High': '#e74c3c'
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(data.index, data[value_col], 
            color='black', linewidth=0.5, alpha=0.7, label=ylabel)
    
    for regime in data[regime_col].dropna().unique():
        regime_mask = data[regime_col] == regime
        regime_periods = data[regime_mask].index
        
        if len(regime_periods) > 0:
            breaks = np.where(np.diff(data.index.get_indexer(regime_periods)) > 1)[0]
            segments = np.split(regime_periods, breaks + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    ax.axvspan(
                        segment[0], segment[-1],
                        alpha=0.2,
                        color=regime_colors.get(regime, 'gray'),
                        label=regime if segment is segments[0] else ""
                    )
    
    if show_thresholds and thresholds is not None:
        ax.axhline(
            thresholds[0], 
            color='blue', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.7,
            label='Lower Threshold'
        )
        ax.axhline(
            thresholds[1], 
            color='red', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.7,
            label='Upper Threshold'
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper left', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_volatility_proxy_with_regimes(
    data: pd.DataFrame,
    proxy_col: str,
    regime_col: str,
    thresholds: Tuple[float, float],
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    regime_colors = {
        'Low': '#2ecc71',
        'Medium': '#f39c12',
        'High': '#e74c3c'
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for regime in ['Low', 'Medium', 'High']:
        mask = data[regime_col] == regime
        ax.scatter(
            data[mask].index,
            data[mask][proxy_col],
            c=regime_colors[regime],
            label=regime,
            alpha=0.6,
            s=10
        )
    
    ax.axhline(
        thresholds[0],
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Lower Threshold ({thresholds[0]:.6f})'
    )
    ax.axhline(
        thresholds[1],
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Upper Threshold ({thresholds[1]:.6f})'
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Trailing Volatility Proxy', fontsize=12)
    ax.set_title('Volatility Proxy and Regime Thresholds', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_distributions(
    squared_returns: pd.Series,
    regimes: pd.Series,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    df = pd.DataFrame({
        'squared_returns': squared_returns,
        'regime': regimes
    }).dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    regime_order = ['Low', 'Medium', 'High']
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    # Boxplot
    data_by_regime = [df[df['regime'] == r]['squared_returns'].values 
                      for r in regime_order if r in df['regime'].unique()]
    labels_present = [r for r in regime_order if r in df['regime'].unique()]
    
    bp = axes[0].boxplot(data_by_regime, labels=labels_present, patch_artist=True)
    for patch, regime in zip(bp['boxes'], labels_present):
        patch.set_facecolor(colors[regime])
        patch.set_alpha(0.7)
    
    axes[0].set_title('Boxplot of Squared Returns by Regime', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Regime', fontsize=11)
    axes[0].set_ylabel('Squared Returns', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Violin plot (approximated with filled areas)
    for i, regime in enumerate(labels_present):
        data = df[df['regime'] == regime]['squared_returns'].values
        parts = axes[1].violinplot([data], positions=[i], widths=0.7, 
                                   showmeans=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[regime])
            pc.set_alpha(0.7)
    
    axes[1].set_xticks(range(len(labels_present)))
    axes[1].set_xticklabels(labels_present)
    axes[1].set_title('Distribution of Squared Returns by Regime', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Regime', fontsize=11)
    axes[1].set_ylabel('Squared Returns', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_regime_persistence(
    persistence_stats: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    regime_colors = [colors.get(r, 'gray') for r in persistence_stats.index]
    
    axes[0].bar(
        persistence_stats.index,
        persistence_stats['avg_duration'],
        color=regime_colors,
        alpha=0.7,
        edgecolor='black'
    )
    axes[0].set_title('Average Regime Duration', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Regime', fontsize=11)
    axes[0].set_ylabel('Average Duration (days)', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(
        persistence_stats.index,
        persistence_stats['num_episodes'],
        color=regime_colors,
        alpha=0.7,
        edgecolor='black'
    )
    axes[1].set_title('Number of Regime Episodes', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Regime', fontsize=11)
    axes[1].set_ylabel('Number of Episodes', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_transition_matrix(
    transition_probs: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(transition_probs.values, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(transition_probs.columns)))
    ax.set_yticks(np.arange(len(transition_probs.index)))
    ax.set_xticklabels(transition_probs.columns)
    ax.set_yticklabels(transition_probs.index)
    
    for i in range(len(transition_probs.index)):
        for j in range(len(transition_probs.columns)):
            text = ax.text(j, i, f'{transition_probs.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Transition Probability', rotation=270, labelpad=20)
    
    ax.set_title('Regime Transition Probability Matrix', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('To Regime', fontsize=11)
    ax.set_ylabel('From Regime', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_regime_coverage(
    coverage_df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    regime_colors = [colors.get(r, 'gray') for r in coverage_df.index]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.pie(
        coverage_df['count'],
        labels=coverage_df.index,
        colors=regime_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    ax.set_title('Regime Coverage', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig
