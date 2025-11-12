"""
Cross-Country Comparison Script
Synthesizes cleaned datasets to identify relative solar potential and key differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import combine_countries, load_all_countries, clean_solar_data
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


def compare_countries(data_dir: str = "data", output_dir: str = "data"):
    """
    Compare solar potential across countries.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing cleaned CSV files or raw files
    output_dir : str
        Directory to save comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load and clean all country datasets
    print("Loading country datasets...")
    datasets = load_all_countries(data_dir)
    
    if not datasets:
        print("No datasets loaded. Please check file paths.")
        return
    
    # Clean datasets
    cleaned_datasets = {}
    for country, df in datasets.items():
        print(f"Cleaning {country}...")
        df_clean, _ = clean_solar_data(df)
        cleaned_datasets[country] = df_clean
    
    # Calculate metrics for each country
    print("\nCalculating solar potential metrics...")
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
    print(f"\nSaved comparison results to {comparison_file}")
    
    # Save metrics as JSON
    metrics_file = output_path / "country_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(country_metrics, f, indent=2, default=str)
    print(f"Saved metrics to {metrics_file}")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COUNTRY COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Country':<20} {'Solar Score':<15} {'Avg Daily GHI':<20} {'Annual Energy':<20}")
    print("-"*80)
    for country in comparison_df.index:
        score = comparison_df.loc[country, 'solar_potential_score']
        ghi = comparison_df.loc[country, 'avg_daily_ghi']
        energy = comparison_df.loc[country, 'annual_solar_energy']
        print(f"{country:<20} {score:<15.2f} {ghi:<20.2f} {energy:<20.2f}")
    
    # Generate comparison visualizations
    print("\nGenerating comparison visualizations...")
    generate_comparison_plots(cleaned_datasets, country_metrics, output_path)
    
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

