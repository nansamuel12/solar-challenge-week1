"""
Data Profiling and Cleaning Script
Profiles, cleans, and explores each country's solar dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from .data_loader import load_all_countries, clean_solar_data, profile_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def profile_and_clean_countries(data_dir: str = "src", output_dir: str = "data"):
    """
    Profile and clean data for all countries.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing raw CSV files
    output_dir : str
        Directory to save cleaned data and profiles
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load all country datasets
    print("Loading country datasets...")
    datasets = load_all_countries(data_dir)
    
    if not datasets:
        print("No datasets loaded. Please check file paths.")
        return
    
    # Profile and clean each country
    profiles = {}
    cleaned_datasets = {}
    original_datasets = {}
    
    for country, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"Profiling {country}")
        print(f"{'='*60}")
        
        # Store original before cleaning
        original_datasets[country] = df.copy()
        
        # Initial profile
        print(f"Initial records: {len(df)}")
        print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Clean data
        print(f"\nCleaning data for {country}...")
        df_clean, df_original = clean_solar_data(df)
        cleaned_datasets[country] = df_clean
        
        print(f"Records before cleaning: {len(df_original)}")
        print(f"Records after cleaning: {len(df_clean)}")
        print(f"Records removed: {len(df_original) - len(df_clean)}")
        print(f"Missing values before cleaning: {df_original.isnull().sum().sum()}")
        print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        
        # Generate profile with original data for comparison
        profile = profile_data(df_clean, country, df_original)
        profiles[country] = profile
        
        # Print columns with >5% nulls
        if profile['columns_high_null']:
            print(f"\nColumns with >5% nulls:")
            for col, pct in profile['columns_high_null'].items():
                print(f"  {col}: {pct:.2f}%")
        else:
            print(f"\nNo columns with >5% nulls")
        
        # Save cleaned data
        output_file = output_path / f"{country.lower().replace(' ', '_')}_cleaned.csv"
        df_clean.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")
        
        # Print key statistics
        print(f"\nKey Statistics for {country}:")
        print(f"  Average GHI: {profile['statistics']['GHI']['mean']:.2f} W/m²")
        print(f"  Max GHI: {profile['statistics']['GHI']['max']:.2f} W/m²")
        print(f"  Average Temperature: {profile['statistics']['Tamb']['mean']:.2f} °C")
        print(f"  Average Humidity: {profile['statistics']['RH']['mean']:.2f} %")
        
        if 'solar_metrics' in profile:
            print(f"  Average Daily GHI: {profile['solar_metrics']['avg_daily_ghi']:.2f} W/m²")
            print(f"  Total Solar Hours: {profile['solar_metrics']['total_solar_hours']}")
    
    # Save profiles
    profiles_file = output_path / "country_profiles.json"
    with open(profiles_file, 'w') as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"\nSaved profiles to {profiles_file}")
    
    return cleaned_datasets, profiles, original_datasets


def generate_exploratory_plots(datasets: dict, original_datasets: dict, output_dir: str = "data"):
    """
    Generate exploratory plots for each country including pre/post-clean comparison and Wind Rose.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of cleaned country datasets
    original_datasets : dict
        Dictionary of original (before cleaning) country datasets
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(exist_ok=True, parents=True)
    
    for country, df in datasets.items():
        print(f"Generating plots for {country}...")
        df_original = original_datasets.get(country, df)
        
        # Sample data for plotting (every 100th point for performance)
        df_sample = df.iloc[::100].copy() if len(df) > 100 else df
        df_original_sample = df_original.iloc[::100].copy() if len(df_original) > 100 else df_original
        
        # Time series analysis - Pre/Post cleaning comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Solar Data Analysis - {country}', fontsize=16, fontweight='bold')
        
        # Plot 1: GHI over time - Pre vs Post cleaning
        axes[0, 0].plot(df_original_sample['Timestamp'], df_original_sample['GHI'], 
                       alpha=0.4, linewidth=0.5, label='Before Cleaning', color='red')
        axes[0, 0].plot(df_sample['Timestamp'], df_sample['GHI'], 
                       alpha=0.6, linewidth=0.5, label='After Cleaning', color='blue')
        axes[0, 0].set_title('GHI Over Time - Cleaning Impact')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('GHI (W/m²)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Daily average GHI by month
        daily_ghi = df.groupby(['Year', 'Month', 'Day'])['GHI'].mean().reset_index()
        monthly_avg = daily_ghi.groupby('Month')['GHI'].mean()
        axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='orange', alpha=0.7)
        axes[0, 1].set_title('Average Daily GHI by Month')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average GHI (W/m²)')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Hourly average GHI
        hourly_avg = df.groupby('Hour')['GHI'].mean()
        axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Average GHI by Hour of Day')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average GHI (W/m²)')
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: GHI distribution - Pre vs Post
        axes[1, 1].hist(df_original['GHI'], bins=50, alpha=0.5, color='red', 
                       edgecolor='black', label='Before Cleaning')
        axes[1, 1].hist(df['GHI'], bins=50, alpha=0.5, color='blue', 
                       edgecolor='black', label='After Cleaning')
        axes[1, 1].set_title('GHI Distribution - Cleaning Impact')
        axes[1, 1].set_xlabel('GHI (W/m²)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(df['GHI'].mean(), color='green', linestyle='--', 
                          label=f'Mean (Cleaned): {df["GHI"].mean():.2f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_path / f"{country.lower().replace(' ', '_')}_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved analysis plot to {plot_file}")
        
        # Wind Rose Plot
        if 'WD' in df.columns and 'WS' in df.columns:
            generate_wind_rose(df, country, output_path)
        
        # Pre/Post cleaning sensor comparison plots
        generate_sensor_comparison_plots(df_original, df, country, output_path)


def generate_wind_rose(df: pd.DataFrame, country: str, output_path: Path):
    """
    Generate a Wind Rose plot showing wind direction and speed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe
    country : str
        Country name
    output_path : Path
        Path to save the plot
    """
    try:
        # Filter out invalid wind data
        wind_data = df[(df['WD'].notna()) & (df['WS'].notna()) & 
                       (df['WD'] >= 0) & (df['WD'] <= 360) & 
                       (df['WS'] >= 0)].copy()
        
        if len(wind_data) == 0:
            print(f"  No valid wind data for {country}")
            return
        
        # Create wind rose using polar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Convert wind direction to radians
        wind_direction = np.deg2rad(wind_data['WD'])
        wind_speed = wind_data['WS']
        
        # Create bins for wind speed
        speed_bins = [0, 2, 5, 10, 15, 20, 30, 100]
        speed_labels = ['0-2', '2-5', '5-10', '10-15', '15-20', '20-30', '30+']
        wind_data['Speed_Bin'] = pd.cut(wind_speed, bins=speed_bins, labels=speed_labels)
        
        # Create direction bins (16 directions)
        direction_bins = np.linspace(0, 360, 17)
        direction_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        wind_data['Direction_Bin'] = pd.cut(wind_data['WD'], bins=direction_bins, 
                                           labels=direction_labels, include_lowest=True)
        
        # Count frequencies
        direction_rad = np.deg2rad(np.linspace(0, 360, 16, endpoint=False))
        
        # Aggregate by direction and speed
        for i, speed_label in enumerate(speed_labels):
            speed_data = wind_data[wind_data['Speed_Bin'] == speed_label]
            if len(speed_data) > 0:
                direction_counts = speed_data['Direction_Bin'].value_counts().sort_index()
                
                # Map to radians
                counts = []
                for dir_label in direction_labels:
                    if dir_label in direction_counts.index:
                        counts.append(direction_counts[dir_label])
                    else:
                        counts.append(0)
                
                # Normalize by total
                total = sum(counts)
                if total > 0:
                    counts_norm = [c / total * 100 for c in counts]
                    
                    # Plot bars
                    bars = ax.bar(direction_rad, counts_norm, width=2*np.pi/16, 
                                 label=f'{speed_label} m/s', alpha=0.7)
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.deg2rad(np.linspace(0, 360, 8, endpoint=False)), 
                         ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_rlabel_position(90)
        ax.set_ylabel('Frequency (%)', labelpad=20)
        ax.set_title(f'Wind Rose - {country}', pad=20, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        wind_rose_file = output_path / f"{country.lower().replace(' ', '_')}_wind_rose.png"
        plt.savefig(wind_rose_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved wind rose to {wind_rose_file}")
        
    except Exception as e:
        print(f"  Error generating wind rose for {country}: {str(e)}")


def generate_sensor_comparison_plots(df_original: pd.DataFrame, df_clean: pd.DataFrame, 
                                     country: str, output_path: Path):
    """
    Generate pre/post cleaning comparison plots for key sensors.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before cleaning
    df_clean : pd.DataFrame
        Cleaned dataframe
    country : str
        Country name
    output_path : Path
        Path to save the plots
    """
    sensor_cols = ['GHI', 'DNI', 'DHI', 'Tamb', 'WS']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Sensor Data Comparison - Pre vs Post Cleaning - {country}', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, col in enumerate(sensor_cols):
        if col in df_original.columns and col in df_clean.columns:
            ax = axes[idx]
            
            # Sample data for performance
            sample_size = min(1000, len(df_original), len(df_clean))
            orig_sample = df_original[col].sample(n=sample_size, random_state=42) if len(df_original) > sample_size else df_original[col]
            clean_sample = df_clean[col].sample(n=sample_size, random_state=42) if len(df_clean) > sample_size else df_clean[col]
            
            ax.hist(orig_sample.dropna(), bins=50, alpha=0.5, color='red', 
                   label='Before Cleaning', edgecolor='black')
            ax.hist(clean_sample.dropna(), bins=50, alpha=0.5, color='blue', 
                   label='After Cleaning', edgecolor='black')
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    # Remove empty subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    comparison_file = output_path / f"{country.lower().replace(' ', '_')}_sensor_comparison.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved sensor comparison plot to {comparison_file}")


if __name__ == "__main__":
    print("Starting data profiling and cleaning process...")
    cleaned_datasets, profiles, original_datasets = profile_and_clean_countries()
    
    if cleaned_datasets:
        print("\nGenerating exploratory plots...")
        generate_exploratory_plots(cleaned_datasets, original_datasets)
        print("\nData profiling and cleaning complete!")

