#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obesity Trends Analysis: COVID-19 Impact Study
================================================
Statistical analysis of obesity rate changes across US states and demographics
comparing pre-COVID (2011-2018) with COVID-era (2021-2023) periods.

Data Source: BRFSS (Behavioral Risk Factor Surveillance System)
Authors: Ahmed & Caruso
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# Time intervals for analysis
INTERVALS = {
    'early': (2011, 2013),
    'middle': (2016, 2018),
    'late': (2021, 2023)
}

# Bootstrap parameters
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Demographic category orderings
INCOME_ORDER = [
    "Less than $15,000",
    "$15,000 - $24,999",
    "$25,000 - $34,999",
    "$35,000 - $49,999",
    "$50,000 - $74,999",
    "$75,000 or greater"
]

RACE_ORDER = [
    'Asian',
    'Hispanic',
    'Non-Hispanic Black',
    'American Indian/Alaska Native',
    'Hawaiian/Pacific Islander',
    '2 or more races',
    'Other'
]

AGE_ORDER = ['18 - 24', '25 - 34', '35 - 44', '45 - 54', '55 - 64', '65 or older']

EDUCATION_ORDER = [
    'Less than high school',
    'High school graduate',
    'Some college or technical sch',
    'College graduate'
]

# Columns to retain
COLUMNS_TO_KEEP = [
    'YearStart', 'YearEnd', 'LocationDesc', 'Question', 'Data_Value',
    'Age(years)', 'Education', 'Sex', 'Income', 'Race/Ethnicity',
    'Low_Confidence_Limit', 'High_Confidence_Limit ', 'Sample_Size'
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate the obesity dataset."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def split_by_period(df: pd.DataFrame, intervals: Dict[str, Tuple[int, int]]) -> Dict[str, pd.DataFrame]:
    """Split dataframe into time periods."""
    periods = {}
    
    for period_name, (start_year, end_year) in intervals.items():
        mask = (
            ((df['YearStart'] >= start_year) & (df['YearStart'] <= end_year)) |
            ((df['YearEnd'] >= start_year) & (df['YearEnd'] <= end_year))
        )
        periods[period_name] = df[mask].copy()
        print(f"{period_name.capitalize()} period ({start_year}-{end_year}): {periods[period_name].shape[0]} rows")
    
    return periods


def filter_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Keep only specified columns that exist in the dataframe."""
    columns_present = [col for col in columns if col in df.columns]
    print(f"Retained {len(columns_present)}/{len(columns)} columns")
    return df[columns_present].copy()


def filter_obesity_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for rows containing obesity-related questions."""
    return df[df['Question'].str.contains('Obesity', case=False, na=False)]


def compute_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean of values."""
    return np.sum(weights * values) / np.sum(weights)


def bootstrap_difference(
    df_early: pd.DataFrame,
    df_late: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for difference in means.
    
    Returns:
        (ci_lower, ci_upper, p_value)
    """
    boot_diffs = []
    
    for _ in range(n_boot):
        # Resample both periods
        sample_early = df_early.sample(n=len(df_early), replace=True)
        sample_late = df_late.sample(n=len(df_late), replace=True)
        
        # Compute weighted means
        mean_early = compute_weighted_mean(
            sample_early['Data_Value'] / 100,
            sample_early['Sample_Size']
        )
        mean_late = compute_weighted_mean(
            sample_late['Data_Value'] / 100,
            sample_late['Sample_Size']
        )
        
        boot_diffs.append(mean_late - mean_early)
    
    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    
    return ci_lower, ci_upper, p_value


def bootstrap_did(
    df_early: pd.DataFrame,
    df_middle: pd.DataFrame,
    df_late: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP
) -> Tuple[float, float, float, float]:
    """
    Bootstrap Difference-in-Differences analysis.
    
    Returns:
        (observed_did, ci_lower, ci_upper, p_value)
    """
    # Observed DiD
    mean_early = compute_weighted_mean(df_early['Data_Value'] / 100, df_early['Sample_Size'])
    mean_middle = compute_weighted_mean(df_middle['Data_Value'] / 100, df_middle['Sample_Size'])
    mean_late = compute_weighted_mean(df_late['Data_Value'] / 100, df_late['Sample_Size'])
    
    D1 = mean_middle - mean_early
    D2 = mean_late - mean_middle
    observed_did = D1 - D2
    
    # Bootstrap
    boot_did = []
    for _ in range(n_boot):
        b_early = df_early.sample(n=len(df_early), replace=True)
        b_middle = df_middle.sample(n=len(df_middle), replace=True)
        b_late = df_late.sample(n=len(df_late), replace=True)
        
        m_early = compute_weighted_mean(b_early['Data_Value'] / 100, b_early['Sample_Size'])
        m_middle = compute_weighted_mean(b_middle['Data_Value'] / 100, b_middle['Sample_Size'])
        m_late = compute_weighted_mean(b_late['Data_Value'] / 100, b_late['Sample_Size'])
        
        D1_b = m_middle - m_early
        D2_b = m_late - m_middle
        boot_did.append(D1_b - D2_b)
    
    boot_did = np.array(boot_did)
    ci_lower = np.percentile(boot_did, 2.5)
    ci_upper = np.percentile(boot_did, 97.5)
    p_value = np.mean(np.abs(boot_did) >= abs(observed_did))
    
    return observed_did, ci_lower, ci_upper, p_value


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_ecdf_comparison(
    data_early: np.ndarray,
    data_late: np.ndarray,
    state_name: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot ECDF comparison between early and late periods."""
    # Compute ECDFs
    x_early = np.sort(data_early)
    y_early = np.arange(1, len(x_early) + 1) / len(x_early)
    
    x_late = np.sort(data_late)
    y_late = np.arange(1, len(x_late) + 1) / len(x_late)
    
    # Compute statistics
    median_early = np.median(data_early)
    median_late = np.median(data_late)
    mean_early = np.mean(data_early)
    mean_late = np.mean(data_late)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # ECDFs
    ax.plot(x_early, y_early, '-', color='blue', linewidth=2, label='2011-2013 ECDF', zorder=5)
    ax.plot(x_late, y_late, '--', color='red', linewidth=2, label='2021-2023 ECDF', zorder=5)
    
    # Medians
    ax.axvline(median_early, color='blue', linestyle='--', linewidth=1.5,
               label=f"Median (Early): {median_early:.1f}%", zorder=10)
    ax.axvline(median_late, color='red', linestyle='--', linewidth=1.5,
               label=f"Median (Late): {median_late:.1f}%", zorder=10)
    
    # Normal CDFs for reference
    x_vals = np.linspace(min(x_early.min(), x_late.min()),
                        max(x_early.max(), x_late.max()), 200)
    ax.plot(x_vals, norm.cdf(x_vals, loc=mean_early, scale=np.std(data_early)),
            ':', color='blue', linewidth=2, alpha=0.5, label='Normal CDF (Early)', zorder=1)
    ax.plot(x_vals, norm.cdf(x_vals, loc=mean_late, scale=np.std(data_late)),
            ':', color='red', linewidth=2, alpha=0.5, label='Normal CDF (Late)', zorder=1)
    
    ax.set_xlabel("Obesity Rate (%)", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.set_title(f"ECDF of Obesity Rate for {state_name}: Early vs Late Periods", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_demographic_comparison(
    plot_df: pd.DataFrame,
    demographic_name: str,
    category_col: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot bar chart comparing changes across demographic groups."""
    fig, ax = plt.subplots(figsize=figsize)
    
    bar_width = 0.35
    indices = np.arange(len(plot_df))
    
    ax.bar(indices - bar_width/2, plot_df['early-middle'], width=bar_width,
           label='2011–2013 → 2016–2018 (pre-COVID)', color='skyblue', edgecolor='black')
    ax.bar(indices + bar_width/2, plot_df['middle-late'], width=bar_width,
           label='2016–2018 → 2021–2023 (COVID-era)', color='orchid', edgecolor='black')
    
    # Set x-axis labels
    category_col = category_col or demographic_name
    ax.set_xticks(indices)
    ax.set_xticklabels(plot_df[category_col], rotation=45, ha='right')
    
    ax.set_ylabel('Difference in Mean Obesity Proportion', fontsize=12)
    ax.set_title(f'Obesity Rate Changes by {demographic_name} Across Time Intervals', fontsize=14)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_state_differences(
    df_early: pd.DataFrame,
    df_middle: pd.DataFrame,
    df_late: pd.DataFrame
) -> pd.DataFrame:
    """Analyze mean obesity differences across states."""
    # Compute means by state
    early_means = df_early.groupby('LocationDesc')['Data_Value'].mean().rename('mean_early')
    middle_means = df_middle.groupby('LocationDesc')['Data_Value'].mean().rename('mean_middle')
    late_means = df_late.groupby('LocationDesc')['Data_Value'].mean().rename('mean_late')
    
    # Combine
    means_df = pd.concat([early_means, middle_means, late_means], axis=1)
    
    # Compute differences
    means_df['mean_diff'] = means_df['mean_late'] - means_df['mean_early']
    means_df['upper_mean_dif'] = means_df['mean_late'] - means_df['mean_middle']
    means_df['lower_mean_dif'] = means_df['mean_middle'] - means_df['mean_early']
    
    # Keep only states with all periods
    means_df = means_df.dropna(subset=['mean_early', 'mean_late', 'mean_middle'])
    
    return means_df


def analyze_demographic(
    df_obesity: pd.DataFrame,
    demographic_col: str,
    intervals: Dict[str, Tuple[int, int]],
    category_order: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze obesity changes for a demographic variable.
    
    Returns:
        (differences_df, did_df)
    """
    # Ensure numeric columns
    df_obesity['Data_Value'] = pd.to_numeric(df_obesity['Data_Value'], errors='coerce')
    df_obesity['Sample_Size'] = pd.to_numeric(df_obesity['Sample_Size'], errors='coerce')
    df_obesity = df_obesity.dropna(subset=['Data_Value', 'Sample_Size'])
    
    categories = df_obesity[demographic_col].dropna().unique()
    diff_results = []
    did_results = []
    
    for cat in categories:
        if pd.isna(cat) or cat == 'Data not reported':
            continue
        
        df_cat = df_obesity[df_obesity[demographic_col] == cat]
        
        # Compute weighted means for each interval
        means = {}
        for key, (start, end) in intervals.items():
            df_interval = df_cat[(df_cat['YearStart'] >= start) & (df_cat['YearEnd'] <= end)]
            if len(df_interval) == 0:
                continue
            means[key] = compute_weighted_mean(
                df_interval['Data_Value'] / 100,
                df_interval['Sample_Size']
            )
        
        # Compute simple differences
        if 'early' in means and 'middle' in means:
            diff_results.append({
                demographic_col: cat,
                'Comparison': 'early-middle',
                'Difference': means['middle'] - means['early']
            })
        
        if 'middle' in means and 'late' in means:
            diff_results.append({
                demographic_col: cat,
                'Comparison': 'middle-late',
                'Difference': means['late'] - means['middle']
            })
        
        # Difference-in-Differences
        if all(k in means for k in ['early', 'middle', 'late']):
            df_early = df_cat[(df_cat['YearStart'] >= 2011) & (df_cat['YearEnd'] <= 2013)]
            df_middle = df_cat[(df_cat['YearStart'] >= 2016) & (df_cat['YearEnd'] <= 2018)]
            df_late = df_cat[(df_cat['YearStart'] >= 2021) & (df_cat['YearEnd'] <= 2023)]
            
            if len(df_early) > 0 and len(df_middle) > 0 and len(df_late) > 0:
                did_obs, ci_lower, ci_upper, p_value = bootstrap_did(df_early, df_middle, df_late)
                
                did_results.append({
                    demographic_col: cat,
                    'DiD': did_obs,
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper,
                    'p_value': p_value
                })
    
    diff_df = pd.DataFrame(diff_results)
    did_df = pd.DataFrame(did_results)
    
    return diff_df, did_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(data_path: str):
    """Main analysis pipeline."""
    
    np.random.seed(RANDOM_SEED)
    
    print("=" * 80)
    print("OBESITY TRENDS ANALYSIS: COVID-19 IMPACT STUDY")
    print("=" * 80)
    
    # 1. Load data
    print("\n[1/7] Loading data...")
    df = load_data(data_path)
    
    # 2. Split by time periods
    print("\n[2/7] Splitting data by time periods...")
    periods = split_by_period(df, INTERVALS)
    
    # 3. Filter columns
    print("\n[3/7] Filtering columns...")
    for period_name in periods:
        periods[period_name] = filter_columns(periods[period_name], COLUMNS_TO_KEEP)
    
    # 4. Filter for obesity questions
    print("\n[4/7] Filtering for obesity-related questions...")
    obesity_periods = {}
    for period_name, period_df in periods.items():
        obesity_periods[period_name] = filter_obesity_questions(period_df)
        print(f"  {period_name.capitalize()}: {obesity_periods[period_name].shape[0]} obesity rows")
    
    df_obesity = filter_obesity_questions(df)
    
    # 5. State-level analysis
    print("\n[5/7] Analyzing state-level differences...")
    means_df = analyze_state_differences(
        obesity_periods['early'],
        obesity_periods['middle'],
        obesity_periods['late']
    )
    
    print(f"  Number of states analyzed: {means_df.shape[0]}")
    print(f"  Mean difference (late - early): {means_df['mean_diff'].mean():.3f}%")
    print(f"  Std deviation: {means_df['mean_diff'].std():.3f}%")
    
    # Paired t-test
    lower_dif = means_df['lower_mean_dif'].dropna().values
    upper_dif = means_df['upper_mean_dif'].dropna().values
    t_stat, p_val = stats.ttest_rel(upper_dif, lower_dif)
    
    print(f"\n  Paired t-test (COVID vs pre-COVID):")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_val:.4f}")
    
    # 6. Demographic analyses
    print("\n[6/7] Analyzing demographic groups...")
    
    demographics = {
        'Income': INCOME_ORDER,
        'Race/Ethnicity': RACE_ORDER,
        'Sex': None,
        'Age(years)': AGE_ORDER,
        'Education': EDUCATION_ORDER
    }
    
    for demo_col, order in demographics.items():
        print(f"\n  Analyzing {demo_col}...")
        diff_df, did_df = analyze_demographic(df_obesity, demo_col, INTERVALS, order)
        
        if not did_df.empty:
            print(f"    DiD results:")
            print(did_df.to_string(index=False))
    
    # 7. Summary
    print("\n[7/7] Analysis complete!")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total states analyzed: {means_df.shape[0]}")
    print(f"Average obesity increase (2011-2023): {means_df['mean_diff'].mean():.2f}%")
    print(f"States with largest increase:")
    print(means_df.nlargest(3, 'mean_diff')[['mean_early', 'mean_late', 'mean_diff']])
    print("=" * 80)


if __name__ == "__main__":
    # Set your data file path here
    DATA_PATH = input("Enter the path to your obesity CSV file: ").strip()
    
    # Alternative: hardcode for convenience
    # DATA_PATH = "path/to/your/obesity_data.csv"
    
    try:
        main(DATA_PATH)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please provide a valid file path.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
