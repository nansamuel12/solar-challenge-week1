"""
Cross-Country Comparison Script
Synthesizes cleaned datasets to identify relative solar potential and key differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .data_loader import combine_countries, load_all_countries, clean_solar_data
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def calculate_solar_potential_metrics(df: pd.DataFrame, country: str) -> dict:
    """
    Calculate comprehensive solar potential metrics for a country.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned country dataset
    country : str
        Country name
    
    Returns:
    --------
    dict
        Dictionary of solar potential metrics
    """
    metrics = {
        'country': country,
        'total_records': len(df),
        'date_range_days': (df['Timestamp'].max() - df['Timestamp'].min()).days,
    }
    
    # Daily metrics
    daily_stats = df.groupby(['Year', 'Month', 'Day']).agg({
        'GHI': ['sum', 'mean', 'max'],
        'DNI': 'mean',
        'DHI': 'mean',
        'Tamb': 'mean',
        'RH': 'mean',
        'WS': 'mean',
        'Precipitation': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['Year', 'Month', 'Day', 'GHI_sum', 'GHI_mean', 'GHI_max', 
                          'DNI_mean', 'DHI_mean', 'Tamb_mean', 'RH_mean', 'WS_mean', 'Precipitation_sum']
    
    # Calculate daily solar energy (kWh/m²/day)
    daily_stats['Daily_Solar_Energy'] = daily_stats['GHI_sum'] / 1000 / 60  # Approximate conversion
    
    # Key metrics
    metrics['avg_daily_solar_energy'] = float(daily_stats['Daily_Solar_Energy'].mean())
    metrics['max_daily_solar_energy'] = float(daily_stats['Daily_Solar_Energy'].max())
    metrics['annual_solar_energy'] = float(daily_stats['Daily_Solar_Energy'].sum())
    metrics['avg_daily_ghi'] = float(daily_stats['GHI_mean'].mean())
    metrics['max_daily_ghi'] = float(daily_stats['GHI_max'].max())
    metrics['avg_dni'] = float(df['DNI'].mean())
    metrics['avg_dhi'] = float(df['DHI'].mean())
    
    # Weather metrics
    metrics['avg_temperature'] = float(df['Tamb'].mean())
    metrics['avg_humidity'] = float(df['RH'].mean())
    metrics['avg_wind_speed'] = float(df['WS'].mean())
    metrics['total_precipitation'] = float(df['Precipitation'].sum())
    metrics['rainy_days'] = int((daily_stats['Precipitation_sum'] > 0).sum())
    
    # Solar hours (GHI > 10 W/m²)
    metrics['solar_hours'] = int((df['GHI'] > 10).sum())
    metrics['avg_daily_solar_hours'] = metrics['solar_hours'] / len(daily_stats) if len(daily_stats) > 0 else 0
    
    # Clear sky ratio (high DNI relative to GHI)
    clear_sky_mask = (df['DNI'] > 0) & (df['GHI'] > 10)
    if clear_sky_mask.sum() > 0:
        dni_ghi_ratio = (df.loc[clear_sky_mask, 'DNI'] / (df.loc[clear_sky_mask, 'GHI'] + 1e-6)).mean()
        metrics['clear_sky_ratio'] = float(dni_ghi_ratio)
        metrics['clear_sky_days'] = int((daily_stats['DNI_mean'] / (daily_stats['GHI_mean'] + 1e-6) > 0.6).sum())
    else:
        metrics['clear_sky_ratio'] = 0.0
        metrics['clear_sky_days'] = 0
    
    # Seasonal analysis
    df['Season'] = df['Month'].map({
        12: 'Dry', 1: 'Dry', 2: 'Dry',
        3: 'Dry', 4: 'Wet', 5: 'Wet',
        6: 'Wet', 7: 'Wet', 8: 'Wet',
        9: 'Wet', 10: 'Dry', 11: 'Dry'
    })
    
    seasonal_stats = df.groupby('Season')['GHI'].mean()
    metrics['dry_season_ghi'] = float(seasonal_stats.get('Dry', 0))
    metrics['wet_season_ghi'] = float(seasonal_stats.get('Wet', 0))
    
    # Solar potential score (composite metric)
    # Normalize and weight different factors
    ghi_score = min(metrics['avg_daily_ghi'] / 500, 1.0) * 0.4  # Max ~500 W/m² avg is excellent
    solar_hours_score = min(metrics['avg_daily_solar_hours'] / 12, 1.0) * 0.3  # 12 hours is excellent
    clear_sky_score = metrics['clear_sky_ratio'] * 0.2  # Higher is better
    temp_score = (1 - abs(metrics['avg_temperature'] - 25) / 25) * 0.1  # 25°C is optimal
    
    metrics['solar_potential_score'] = float(ghi_score + solar_hours_score + clear_sky_score + temp_score) * 100
    
    return metrics


def create_irradiance_boxplots(datasets: dict, output_path: Path):
    """
    Create side-by-side boxplots for GHI, DNI, and DHI across countries.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of country datasets
    output_path : Path
        Path to save plots
    """
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for boxplots
    irradiance_data = []
    countries = list(datasets.keys())
    
    for country in countries:
        df = datasets[country]
        for metric in ['GHI', 'DNI', 'DHI']:
            if metric in df.columns:
                values = df[metric].dropna()
                for val in values:
                    irradiance_data.append({
                        'Country': country,
                        'Metric': metric,
                        'Value': val
                    })
    
    irradiance_df = pd.DataFrame(irradiance_data)
    
    # Create boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Side-by-Side Boxplots: Solar Irradiance Components by Country', 
                 fontsize=16, fontweight='bold')
    
    metrics = ['GHI', 'DNI', 'DHI']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        metric_data = irradiance_df[irradiance_df['Metric'] == metric]
        sns.boxplot(data=metric_data, x='Country', y='Value', ax=axes[i], 
                   palette=colors[:len(countries)])
        axes[i].set_title(f'{metric} Distribution')
        axes[i].set_ylabel(f'{metric} (W/m²)')
        axes[i].grid(True, alpha=0.3)
        
        # Add median values as text
        for j, country in enumerate(countries):
            country_data = metric_data[metric_data['Country'] == country]['Value']
            median_val = country_data.median()
            axes[i].text(j, median_val, f'{median_val:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    boxplot_file = plots_dir / "irradiance_boxplots.png"
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved irradiance boxplots to {boxplot_file}")
    
    return irradiance_df


def create_summary_statistics_table(datasets: dict) -> pd.DataFrame:
    """
    Generate summary table comparing mean, median, and standard deviation across countries.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of country datasets
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics table
    """
    summary_data = []
    countries = list(datasets.keys())
    metrics = ['GHI', 'DNI', 'DHI']
    
    for country in countries:
        df = datasets[country]
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                summary_data.append({
                    'Country': country,
                    'Metric': metric,
                    'Mean': values.mean(),
                    'Median': values.median(),
                    'Std Dev': values.std(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Count': len(values)
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS TABLE")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.2f'))
    print("="*80)
    
    return summary_df


def perform_statistical_tests(datasets: dict, alpha: float = 0.05):
    """
    Perform one-way ANOVA or Kruskal-Wallis test on GHI to validate findings.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of country datasets
    alpha : float
        Significance level for hypothesis testing
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: GHI COMPARISON ACROSS COUNTRIES")
    print("="*80)
    
    # Extract GHI data for each country
    ghi_data = []
    country_names = []
    
    for country, df in datasets.items():
        if 'GHI' in df.columns:
            ghi_values = df['GHI'].dropna()
            ghi_data.append(ghi_values)
            country_names.append(country)
            print(f"{country}: {len(ghi_values)} GHI measurements, "
                  f"Mean = {ghi_values.mean():.2f} W/m², "
                  f"Std = {ghi_values.std():.2f} W/m²")
    
    if len(ghi_data) < 2:
        print("Insufficient data for statistical comparison")
        return
    
    # Test for normality to decide between ANOVA and Kruskal-Wallis
    print(f"\nTesting normality (alpha = {alpha}):")
    use_anova = True
    
    for i, (country, data) in enumerate(zip(country_names, ghi_data)):
        # Shapiro-Wilk test for normality (sample size limit)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data[:5000])  # Limit sample size for Shapiro
            print(f"{country}: Shapiro-Wilk p-value = {p_value:.6f}")
            if p_value < alpha:
                use_anova = False
        else:
            # For large samples, use Anderson-Darling
            stat, critical_values, significance_level = stats.anderson(data, 'norm')
            print(f"{country}: Anderson-Darling statistic = {stat:.4f}")
            print(f"  Critical values at {significance_level[-1]}%: {critical_values[-1]:.4f}")
            if stat > critical_values[-1]:
                use_anova = False
    
    # Perform appropriate test
    print(f"\nChosen test: {'One-way ANOVA' if use_anova else 'Kruskal-Wallis'}")
    
    if use_anova:
        # One-way ANOVA
        stat, p_value = stats.f_oneway(*ghi_data)
        test_name = "One-way ANOVA"
        
        # Calculate effect size (eta-squared)
        total_mean = np.mean(np.concatenate(ghi_data))
        ss_between = sum(len(data) * (np.mean(data) - total_mean)**2 for data in ghi_data)
        ss_total = sum((val - total_mean)**2 for data in ghi_data for val in data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
    else:
        # Kruskal-Wallis test
        stat, p_value = stats.kruskal(*ghi_data)
        test_name = "Kruskal-Wallis"
        eta_squared = None  # Effect size not typically calculated for Kruskal-Wallis
    
    # Report results
    print(f"\n{test_name} Results:")
    print(f"Test statistic: {stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    
    if eta_squared is not None:
        print(f"Effect size (eta-squared): {eta_squared:.4f}")
    
    # Interpretation
    print(f"\nInterpretation (alpha = {alpha}):")
    if p_value < alpha:
        print("[PASS] REJECT null hypothesis - Significant differences exist between countries")
        
        # Post-hoc tests if significant
        if len(country_names) > 2:
            print("\nPost-hoc pairwise comparisons:")
            for i in range(len(country_names)):
                for j in range(i+1, len(country_names)):
                    if use_anova:
                        t_stat, p_pair = stats.ttest_ind(ghi_data[i], ghi_data[j])
                        test_type = "t-test"
                    else:
                        t_stat, p_pair = stats.mannwhitneyu(ghi_data[i], ghi_data[j], alternative='two-sided')
                        test_type = "Mann-Whitney U"
                    
                    significance = "[SIGNIFICANT]" if p_pair < alpha else "[NOT significant]"
                    print(f"  {country_names[i]} vs {country_names[j]}: {test_type} p = {p_pair:.6f} ({significance})")
    else:
        print("[FAIL] FAIL to reject null hypothesis - No significant differences between countries")
    
    print("="*80)
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'significant': p_value < alpha
    }


def generate_strategic_recommendations(comparison_df: pd.DataFrame, statistical_results: dict):
    """
    Generate final strategic recommendations for MoonLight Energy Solutions.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Country comparison metrics
    statistical_results : dict
        Results from statistical analysis
    """
    print("\n" + "="*80)
    print("STRATEGIC RECOMMENDATIONS FOR MOONLIGHT ENERGY SOLUTIONS")
    print("="*80)
    
    # Identify top performing country
    top_country = comparison_df.index[0]
    top_score = comparison_df.loc[top_country, 'solar_potential_score']
    
    print(f"\n[TOP RANKED] TOP RANKED COUNTRY: {top_country}")
    print(f"   Solar Potential Score: {top_score:.2f}")
    print(f"   Average Daily GHI: {comparison_df.loc[top_country, 'avg_daily_ghi']:.2f} W/m²")
    print(f"   Annual Solar Energy: {comparison_df.loc[top_country, 'annual_solar_energy']:.0f} kWh/m²/year")
    
    # Statistical validation
    print(f"\n[STATS] STATISTICAL VALIDATION:")
    if statistical_results['significant']:
        print(f"   [PASS] {statistical_results['test_name']} shows significant differences (p = {statistical_results['p_value']:.6f})")
        if statistical_results['eta_squared']:
            effect_size = statistical_results['eta_squared']
            if effect_size > 0.14:
                print(f"   [LARGE] Large effect size (eta-squared = {effect_size:.3f}) - Strong practical significance")
            elif effect_size > 0.06:
                print(f"   [MEDIUM] Medium effect size (eta-squared = {effect_size:.3f}) - Moderate practical significance")
            else:
                print(f"   [SMALL] Small effect size (eta-squared = {effect_size:.3f}) - Limited practical significance")
    else:
        print(f"   [WARNING] No significant differences found (p = {statistical_results['p_value']:.6f})")
    
    # Investment recommendations
    print(f"\n[INVEST] INVESTMENT RECOMMENDATIONS:")
    print(f"   1. [PRIORITY] PRIORITY MARKET: {top_country}")
    print(f"      - Highest solar potential score")
    print(f"      - Optimal conditions for near-term deployment")
    print(f"      - Best ROI potential for initial investment")
    
    if len(comparison_df) > 1:
        second_country = comparison_df.index[1]
        print(f"   2. [EXPANSION] EXPANSION MARKET: {second_country}")
        print(f"      - Second-highest potential")
        print(f"      - Portfolio diversification opportunity")
        print(f"      - Risk mitigation through geographic spread")
    
    print(f"\n[DEPLOY] DEPLOYMENT STRATEGY:")
    print(f"   • Phase 1 (0-12 months): Grid-tied pilot plants in {top_country}")
    print(f"   • Phase 2 (12-24 months): Scale up in {top_country}, begin planning in secondary markets")
    print(f"   • Phase 3 (24-36 months): Full portfolio deployment across all viable markets")
    
    print(f"\n[OPTIMIZE] OPTIMIZATION INSIGHTS:")
    print(f"   • Target commissioning during peak GHI months")
    print(f"   • Prioritize sites with high clear-sky ratios")
    print(f"   • Implement continuous monitoring for performance validation")
    print(f"   • Consider hybrid systems where weather variability is high")
    
    print(f"\n[TIMING] MARKET ENTRY TIMING:")
    print(f"   • Immediate: Begin feasibility studies in {top_country}")
    print(f"   • Short-term: Secure permits and partnerships")
    print(f"   • Medium-term: Deploy pilot facilities")
    print(f"   • Long-term: Scale to commercial operations")
    
    print("="*80)
    print("[COMPLETE] Strategic analysis complete - Ready for executive review!")
    print("="*80)


def compare_countries(data_dir: str = "data", output_dir: str = "data"):
    """
    Compare solar potential across countries with comprehensive analysis.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing cleaned CSV files or raw files
    output_dir : str
        Directory to save comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load cleaned datasets from data/ directory
    print("Loading cleaned country datasets...")
    cleaned_datasets = {}
    
    # Map country names to cleaned file names
    country_files = {
        'Benin': 'benin_cleaned.csv',
        'Sierra Leone': 'sierra_leone_cleaned.csv', 
        'Togo': 'togo_cleaned.csv'
    }
    
    for country, filename in country_files.items():
        file_path = Path(data_dir) / filename
        if file_path.exists():
            print(f"Loading {country} from {file_path}...")
            df = pd.read_csv(file_path)
            # Convert Timestamp back to datetime if needed
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            cleaned_datasets[country] = df
            print(f"  Loaded {len(df)} records for {country}")
        else:
            print(f"Warning: {file_path} not found")
    
    if not cleaned_datasets:
        print("No cleaned datasets loaded. Please check file paths.")
        return None, None, None
    
    # Step 1: Create side-by-side boxplots for GHI, DNI, DHI
    print("\n" + "="*60)
    print("STEP 1: CREATING IRRADIANCE BOXPLOTS")
    print("="*60)
    irradiance_df = create_irradiance_boxplots(cleaned_datasets, output_path)
    
    # Step 2: Generate summary statistics table
    print("\n" + "="*60)
    print("STEP 2: GENERATING SUMMARY STATISTICS TABLE")
    print("="*60)
    summary_df = create_summary_statistics_table(cleaned_datasets)
    
    # Save summary statistics
    summary_file = output_path / "summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary statistics to {summary_file}")
    
    # Step 3: Calculate solar potential metrics
    print("\n" + "="*60)
    print("STEP 3: CALCULATING SOLAR POTENTIAL METRICS")
    print("="*60)
    print("Calculating solar potential metrics...")
    country_metrics = {}
    for country, df in cleaned_datasets.items():
        print(f"  Analyzing {country}...")
        country_metrics[country] = calculate_solar_potential_metrics(df, country)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(country_metrics).T
    comparison_df = comparison_df.sort_values('solar_potential_score', ascending=False)
    
    # Save comparison results
    comparison_file = output_path / "country_comparison.csv"
    comparison_df.to_csv(comparison_file)
    print(f"Saved comparison results to {comparison_file}")
    
    # Save metrics as JSON
    metrics_file = output_path / "country_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(country_metrics, f, indent=2, default=str)
    print(f"Saved metrics to {metrics_file}")
    
    # Step 4: Perform statistical analysis (ANOVA/Kruskal-Wallis)
    print("\n" + "="*60)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("="*60)
    statistical_results = perform_statistical_tests(cleaned_datasets)
    
    # Save statistical results
    stats_file = output_path / "statistical_analysis.json"
    with open(stats_file, 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    print(f"Saved statistical analysis to {stats_file}")
    
    # Step 5: Generate strategic recommendations
    print("\n" + "="*60)
    print("STEP 5: STRATEGIC RECOMMENDATIONS")
    print("="*60)
    generate_strategic_recommendations(comparison_df, statistical_results)
    
    # Step 6: Generate additional visualizations
    print("\n" + "="*60)
    print("STEP 6: GENERATING ADDITIONAL VISUALIZATIONS")
    print("="*60)
    print("Generating comparison visualizations...")
    generate_comparison_plots(cleaned_datasets, country_metrics, output_path)
    
    # Print final comparison summary
    print("\n" + "="*80)
    print("FINAL COUNTRY COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Country':<20} {'Solar Score':<15} {'Avg Daily GHI':<20} {'Annual Energy':<20}")
    print("-"*80)
    for country in comparison_df.index:
        score = comparison_df.loc[country, 'solar_potential_score']
        ghi = comparison_df.loc[country, 'avg_daily_ghi']
        energy = comparison_df.loc[country, 'annual_solar_energy']
        print(f"{country:<20} {score:<15.2f} {ghi:<20.2f} {energy:<20.2f}")
    print("="*80)
    
    return cleaned_datasets, country_metrics, comparison_df


def generate_comparison_plots(datasets: dict, metrics: dict, output_path: Path):
    """
    Generate comparison plots across countries.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of country datasets
    metrics : dict
        Dictionary of country metrics
    output_path : Path
        Path to save plots
    """
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for plotting
    countries = list(metrics.keys())
    
    # Plot 1: Solar Potential Score Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Country Solar Potential Comparison', fontsize=16, fontweight='bold')
    
    scores = [metrics[c]['solar_potential_score'] for c in countries]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    axes[0, 0].barh(countries, scores, color=colors[:len(countries)])
    axes[0, 0].set_xlabel('Solar Potential Score')
    axes[0, 0].set_title('Overall Solar Potential Score')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    for i, (country, score) in enumerate(zip(countries, scores)):
        axes[0, 0].text(score + 1, i, f'{score:.2f}', va='center')
    
    # Plot 2: Average Daily GHI Comparison
    ghi_values = [metrics[c]['avg_daily_ghi'] for c in countries]
    axes[0, 1].bar(countries, ghi_values, color=colors[:len(countries)], alpha=0.7)
    axes[0, 1].set_ylabel('Average Daily GHI (W/m²)')
    axes[0, 1].set_title('Average Daily Global Horizontal Irradiance')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, (country, ghi) in enumerate(zip(countries, ghi_values)):
        axes[0, 1].text(i, ghi + 5, f'{ghi:.2f}', ha='center', va='bottom')
    
    # Plot 3: Annual Solar Energy Comparison
    energy_values = [metrics[c]['annual_solar_energy'] for c in countries]
    axes[1, 0].bar(countries, energy_values, color=colors[:len(countries)], alpha=0.7)
    axes[1, 0].set_ylabel('Annual Solar Energy (kWh/m²/year)')
    axes[1, 0].set_title('Annual Solar Energy Potential')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, (country, energy) in enumerate(zip(countries, energy_values)):
        axes[1, 0].text(i, energy + 50, f'{energy:.0f}', ha='center', va='bottom')
    
    # Plot 4: Monthly GHI Comparison
    monthly_data = []
    for country in countries:
        df = datasets[country]
        monthly_avg = df.groupby('Month')['GHI'].mean()
        monthly_data.append(monthly_avg.values)
    
    x = np.arange(1, 13)
    width = 0.25
    for i, (country, data) in enumerate(zip(countries, monthly_data)):
        axes[1, 1].bar(x + i*width, data, width, label=country, alpha=0.7)
    
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average GHI (W/m²)')
    axes[1, 1].set_title('Monthly GHI Comparison')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_plot = plots_dir / "country_comparison.png"
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {comparison_plot}")
    
    # Plot 5: Detailed metrics radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics for radar chart
    categories = ['Daily GHI', 'Solar Hours', 'Clear Sky', 'Temperature', 'Annual Energy']
    
    # Normalize each metric to 0-1 scale
    normalized_data = {}
    for country in countries:
        m = metrics[country]
        normalized_data[country] = [
            m['avg_daily_ghi'] / 500,  # Max ~500
            m['avg_daily_solar_hours'] / 12,  # Max ~12 hours
            m['clear_sky_ratio'],  # Already 0-1
            (50 - abs(m['avg_temperature'] - 25)) / 50,  # Optimal at 25°C
            m['annual_solar_energy'] / 2000  # Normalize
        ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, country in enumerate(countries):
        values = normalized_data[country] + [normalized_data[country][0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Solar Potential Metrics Comparison (Normalized)', pad=20, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    radar_plot = plots_dir / "solar_metrics_radar.png"
    plt.savefig(radar_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved radar chart to {radar_plot}")


if __name__ == "__main__":
    print("Starting cross-country comparison analysis...")
    datasets, metrics, comparison_df = compare_countries()
    print("\nCross-country comparison complete!")

