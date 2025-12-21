import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


def regime_distribution_statistics(
    squared_returns: pd.Series,
    regimes: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame({
        'squared_returns': squared_returns,
        'regime': regimes
    }).dropna()
    
    stats_dict = df.groupby('regime')['squared_returns'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('25%', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('75%', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('skewness', lambda x: stats.skew(x)),
        ('kurtosis', lambda x: stats.kurtosis(x))
    ])
    
    return stats_dict


def calculate_regime_persistence(regimes: pd.Series) -> pd.DataFrame:
    regimes_clean = regimes.dropna()
    
    regime_changes = regimes_clean != regimes_clean.shift(1)
    regime_runs = regime_changes.cumsum()
    
    # Create DataFrame for proper aggregation
    run_data = pd.DataFrame({
        'regime': regimes_clean.values,
        'run_id': regime_runs.values
    }, index=regimes_clean.index)
    
    run_lengths = run_data.groupby('run_id').agg({
        'regime': 'first',
        'run_id': 'count'
    })
    run_lengths.columns = ['regime', 'duration']
    
    persistence_stats = run_lengths.groupby('regime')['duration'].agg([
        ('avg_duration', 'mean'),
        ('median_duration', 'median'),
        ('max_duration', 'max'),
        ('min_duration', 'min'),
        ('num_episodes', 'count')
    ])
    
    return persistence_stats


def create_regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    regimes_clean = regimes.dropna()
    
    regime_from = regimes_clean.shift(1)
    regime_to = regimes_clean
    
    transitions = pd.DataFrame({
        'from': regime_from.iloc[1:],
        'to': regime_to.iloc[1:]
    })
    
    transition_matrix = pd.crosstab(
        transitions['from'],
        transitions['to'],
        margins=True
    )
    
    return transition_matrix


def calculate_transition_probabilities(
    transition_matrix: pd.DataFrame
) -> pd.DataFrame:
    if 'All' in transition_matrix.index:
        transition_matrix = transition_matrix.drop('All', axis=0)
    if 'All' in transition_matrix.columns:
        transition_matrix = transition_matrix.drop('All', axis=1)
    
    transition_probs = transition_matrix.div(
        transition_matrix.sum(axis=1),
        axis=0
    )
    
    return transition_probs


def calculate_regime_coverage(regimes: pd.Series) -> pd.DataFrame:
    coverage = regimes.value_counts().to_frame('count')
    coverage['percentage'] = (coverage['count'] / coverage['count'].sum() * 100)
    coverage = coverage.sort_index()
    
    return coverage


def validate_regime_separation(
    squared_returns: pd.Series,
    regimes: pd.Series,
    alpha: float = 0.05
) -> dict:
    """
    Test whether squared return distributions differ across regimes
    using the Kruskal-Wallis H-test.
    """

    # Collect data for each regime
    grouped_data = []
    labels = []

    for regime in regimes.dropna().unique():
        values = squared_returns[regimes == regime].dropna()
        if len(values) > 0:
            grouped_data.append(values)
            labels.append(regime)

    # Require at least two non-empty groups
    if len(grouped_data) < 2:
        return {
            "test": "Kruskal-Wallis H-test",
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "interpretation": "Insufficient data to test regime separation"
        }

    # Perform Kruskal-Wallis test
    stat, p_value = stats.kruskal(*grouped_data)

    significant = p_value < alpha

    interpretation = (
        "Regimes are significantly different (p < {:.3f})".format(alpha)
        if significant
        else "No statistically significant difference between regimes"
    )

    return {
        "test": "Kruskal-Wallis H-test",
        "statistic": stat,
        "p_value": p_value,
        "significant": significant,
        "interpretation": interpretation
    }


def generate_regime_validation_report(
    squared_returns: pd.Series,
    regimes: pd.Series,
    volatility_proxy: pd.Series,
    thresholds: Tuple[float, float]
) -> Dict[str, any]:
    report = {
        'thresholds': {
            'lower': thresholds[0],
            'upper': thresholds[1]
        },
        'coverage': calculate_regime_coverage(regimes),
        'distribution_stats': regime_distribution_statistics(
            squared_returns, regimes
        ),
        'persistence': calculate_regime_persistence(regimes),
        'transition_matrix': create_regime_transition_matrix(regimes),
        'transition_probabilities': calculate_transition_probabilities(
            create_regime_transition_matrix(regimes)
        ),
        'separation_test': validate_regime_separation(
            squared_returns, regimes
        )
    }
    
    return report