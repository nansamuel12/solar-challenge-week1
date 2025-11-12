"""
Data Loading Utility Module
Provides functions to load and preprocess solar data from CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def load_country_data(file_path: str, country_name: str) -> pd.DataFrame:
    """
    Load and preprocess solar data for a specific country.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    country_name : str
        Name of the country (for identification)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed dataframe with country identifier
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Add country identifier
        df['Country'] = country_name
        
        # Convert Timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Extract temporal features
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfYear'] = df['Timestamp'].dt.dayofyear
        df['WeekOfYear'] = df['Timestamp'].dt.isocalendar().week
        
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame()


def load_all_countries(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load data for all countries.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping country names to their dataframes
    """
    data_dir = Path(data_dir)
    countries = {
        'Benin': 'benin-malanville.csv',
        'Sierra Leone': 'sierraleone-bumbuna.csv',
        'Togo': 'togo-dapaong_qc.csv'
    }
    
    datasets = {}
    for country, filename in countries.items():
        file_path = data_dir / filename
        if file_path.exists():
            datasets[country] = load_country_data(str(file_path), country)
            print(f"Loaded {country}: {len(datasets[country])} records")
        else:
            print(f"Warning: {file_path} not found")
    
    return datasets


def clean_solar_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean solar irradiance data by handling missing values and outliers using Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw solar data dataframe
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (cleaned dataframe, original dataframe for comparison)
    """
    df_original = df.copy()
    df_clean = df.copy()
    
    # Key columns for outlier detection using Z-scores
    outlier_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
    
    # Handle negative values for irradiance (should be >= 0)
    for col in ['GHI', 'DNI', 'DHI']:
        if col in df_clean.columns:
            # Set negative values to 0 (nighttime readings)
            df_clean[col] = df_clean[col].clip(lower=0)
    
    # Remove outliers using Z-score method (Z > 3)
    outlier_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
    
    for col in outlier_cols:
        if col in df_clean.columns:
            # Calculate Z-scores
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / (df_clean[col].std() + 1e-6))
            # Flag outliers where Z > 3
            col_outliers = z_scores > 3
            outlier_mask = outlier_mask | col_outliers
    
    # Remove rows with outliers
    rows_before = len(df_clean)
    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
    rows_after = len(df_clean)
    outliers_removed = rows_before - rows_after
    
    # Key columns for imputation
    key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS', 'WSgust', 'BP', 'Precipitation']
    
    # Handle missing values using median imputation
    for col in key_cols:
        if col in df_clean.columns:
            # Calculate median (excluding NaN)
            median_value = df_clean[col].median()
            # Impute missing values with median
            df_clean[col] = df_clean[col].fillna(median_value)
    
    # Calculate additional metrics
    if 'GHI' in df_clean.columns:
        # Daily solar energy (kWh/m²/day) - approximate conversion
        df_clean['Daily_Solar_Energy'] = df_clean.groupby(['Year', 'Month', 'Day'])['GHI'].transform('sum') / 1000 / 60  # Convert to kWh/m²/day
    
    # Create binary indicators
    df_clean['Is_Daytime'] = (df_clean['GHI'] > 10).astype(int)
    df_clean['Is_Clear_Sky'] = ((df_clean['DNI'] > 0) & (df_clean['DNI'] / (df_clean['GHI'] + 1e-6) > 0.6)).astype(int)
    
    return df_clean, df_original


def profile_data(df: pd.DataFrame, country_name: str, df_original: Optional[pd.DataFrame] = None) -> Dict:
    """
    Generate data profile for a country dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to profile (cleaned)
    country_name : str
        Name of the country
    df_original : pd.DataFrame, optional
        Original dataframe before cleaning (for comparison)
    
    Returns:
    --------
    Dict
        Dictionary containing profile statistics
    """
    profile = {
        'country': country_name,
        'total_records': len(df),
        'date_range': {
            'start': str(df['Timestamp'].min()) if 'Timestamp' in df.columns else None,
            'end': str(df['Timestamp'].max()) if 'Timestamp' in df.columns else None
        },
        'missing_values': df.isnull().sum().to_dict(),
        'missing_values_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
        'columns_high_null': [],
        'describe': {},
        'statistics': {}
    }
    
    # Calculate df.describe() for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile['describe'] = df[numeric_cols].describe().to_dict()
    
    # Identify columns with >5% nulls
    null_percent = df.isnull().sum() / len(df) * 100
    high_null_cols = null_percent[null_percent > 5].index.tolist()
    profile['columns_high_null'] = {col: float(null_percent[col]) for col in high_null_cols}
    
    # Key metrics for solar analysis
    key_metrics = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS', 'WSgust', 'BP', 'Precipitation']
    
    for metric in key_metrics:
        if metric in df.columns:
            profile['statistics'][metric] = {
                'mean': float(df[metric].mean()) if not pd.isna(df[metric].mean()) else None,
                'median': float(df[metric].median()) if not pd.isna(df[metric].median()) else None,
                'std': float(df[metric].std()) if not pd.isna(df[metric].std()) else None,
                'min': float(df[metric].min()) if not pd.isna(df[metric].min()) else None,
                'max': float(df[metric].max()) if not pd.isna(df[metric].max()) else None,
                'q25': float(df[metric].quantile(0.25)) if not pd.isna(df[metric].quantile(0.25)) else None,
                'q75': float(df[metric].quantile(0.75)) if not pd.isna(df[metric].quantile(0.75)) else None
            }
    
    # Solar-specific metrics
    if 'GHI' in df.columns:
        try:
            daily_groups = df.groupby(['Year', 'Month', 'Day'])
            daily_ghi_mean = daily_groups['GHI'].mean().mean()
            daily_ghi_max = daily_groups['GHI'].sum().max()
            clear_sky_days = int(daily_groups['Is_Clear_Sky'].mean().apply(lambda x: x > 0.7).sum()) if 'Is_Clear_Sky' in df.columns else 0
            profile['solar_metrics'] = {
                'avg_daily_ghi': float(daily_ghi_mean) if not pd.isna(daily_ghi_mean) else None,
                'max_daily_ghi': float(daily_ghi_max) if not pd.isna(daily_ghi_max) else None,
                'total_solar_hours': int((df['GHI'] > 10).sum()),
                'clear_sky_days': clear_sky_days
            }
        except Exception:
            profile['solar_metrics'] = {
                'avg_daily_ghi': None,
                'max_daily_ghi': None,
                'total_solar_hours': int((df['GHI'] > 10).sum()),
                'clear_sky_days': 0
            }
    
    # Comparison with original data if provided
    if df_original is not None:
        profile['cleaning_impact'] = {
            'records_before': len(df_original),
            'records_after': len(df),
            'records_removed': len(df_original) - len(df),
            'missing_before': int(df_original.isnull().sum().sum()),
            'missing_after': int(df.isnull().sum().sum())
        }
    
    return profile


def combine_countries(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine datasets from all countries into a single dataframe.
    
    Parameters:
    -----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of country datasets
    
    Returns:
    --------
    pd.DataFrame
        Combined dataframe
    """
    if not datasets:
        return pd.DataFrame()
    
    combined = pd.concat(list(datasets.values()), ignore_index=True)
    return combined

