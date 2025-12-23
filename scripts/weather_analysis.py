"""
================================================================================
Weather Forecasting Dataset - Data Analysis Script

Project: Weather Forecasting Dataset
Category: Time Series Data
Version: 1.0.0
Created: 2024

Author: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277

About RSK World:
RSK World is your one-stop destination for free programming resources,
source code, and development tools.

Description:
This script provides data analysis and visualization functions for the
Weather Forecasting Dataset. It includes functions for loading data,
statistical analysis, and creating visualizations.

Features:
- Temperature and humidity data analysis
- Pressure and precipitation correlation
- Wind speed and direction patterns
- Multiple locations comparison
- Time series visualization

License: Educational Purpose Only
Content used for educational purposes only.

© 2024 RSK World - https://rskworld.in
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load weather data from CSV file.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Weather data DataFrame
    """
    # Skip comment lines at the beginning
    df = pd.read_csv(filepath, comment='#')
    
    # Convert date and time columns
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df


def load_weather_json(filepath: str) -> dict:
    """
    Load weather data from JSON file.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
        
    Returns:
    --------
    dict
        Weather data dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_statistics(df: pd.DataFrame, column: str) -> dict:
    """
    Calculate basic statistics for a column.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    column : str
        Column name to analyze
        
    Returns:
    --------
    dict
        Dictionary with statistical measures
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'range': df[column].max() - df[column].min(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75),
        'iqr': df[column].quantile(0.75) - df[column].quantile(0.25)
    }
    return stats


def analyze_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze weather data by location.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics by location
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    location_stats = df.groupby('location')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    return location_stats


def plot_temperature_trends(df: pd.DataFrame, save_path: str = None):
    """
    Plot temperature trends over time for all locations.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for location in df['location'].unique():
        location_data = df[df['location'] == location]
        ax.plot(location_data['datetime'], location_data['temperature_celsius'], 
                label=location, marker='o', markersize=4, linewidth=2)
    
    ax.set_xlabel('Date & Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Trends by Location\nWeather Forecasting Dataset - rskworld.in', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_humidity_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot humidity distribution by location.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    locations = df['location'].unique()
    humidity_data = [df[df['location'] == loc]['humidity_percent'].dropna() for loc in locations]
    
    bp = ax.boxplot(humidity_data, labels=locations, patch_artist=True)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(locations)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Location', fontsize=12)
    ax.set_ylabel('Humidity (%)', fontsize=12)
    ax.set_title('Humidity Distribution by Location\nWeather Forecasting Dataset - rskworld.in', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation heatmap for numerical variables.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    save_path : str, optional
        Path to save the figure
    """
    numeric_cols = ['temperature_celsius', 'humidity_percent', 'pressure_hpa', 
                   'wind_speed_kmh', 'cloud_cover_percent', 'visibility_km', 'uv_index']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    correlation_matrix = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5, ax=ax, square=True)
    
    ax.set_title('Weather Variables Correlation Matrix\nWeather Forecasting Dataset - rskworld.in', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_daily_patterns(df: pd.DataFrame, variable: str = 'temperature_celsius', save_path: str = None):
    """
    Plot daily patterns for a specific weather variable.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
    variable : str
        Variable to plot
    save_path : str, optional
        Path to save the figure
    """
    if 'datetime' not in df.columns:
        print("Error: datetime column not found")
        return
    
    df['hour'] = df['datetime'].dt.hour
    hourly_avg = df.groupby(['location', 'hour'])[variable].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for location in hourly_avg['location'].unique():
        loc_data = hourly_avg[hourly_avg['location'] == location]
        ax.plot(loc_data['hour'], loc_data[variable], marker='o', 
                label=location, linewidth=2, markersize=6)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Daily {variable.replace("_", " ").title()} Patterns\nWeather Forecasting Dataset - rskworld.in', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def weather_summary_report(df: pd.DataFrame) -> str:
    """
    Generate a text summary report of the weather data.
    
    Author: Molla Samser | Website: https://rskworld.in
    
    Parameters:
    -----------
    df : pd.DataFrame
        Weather data DataFrame
        
    Returns:
    --------
    str
        Summary report text
    """
    report = """
================================================================================
                    WEATHER FORECASTING DATASET - SUMMARY REPORT
================================================================================
    Project: Weather Forecasting Dataset
    Author: Molla Samser | Website: https://rskworld.in
    Email: help@rskworld.in | Phone: +91 93305 39277
================================================================================

DATASET OVERVIEW:
-----------------
    Total Records: {total_records}
    Locations: {locations}
    Date Range: {date_min} to {date_max}
    
TEMPERATURE STATISTICS (°C):
----------------------------
    Mean: {temp_mean:.2f}
    Min: {temp_min:.2f}
    Max: {temp_max:.2f}
    Std Dev: {temp_std:.2f}
    
HUMIDITY STATISTICS (%):
------------------------
    Mean: {humid_mean:.2f}
    Min: {humid_min:.2f}
    Max: {humid_max:.2f}
    Std Dev: {humid_std:.2f}

PRESSURE STATISTICS (hPa):
--------------------------
    Mean: {press_mean:.2f}
    Min: {press_min:.2f}
    Max: {press_max:.2f}

WIND SPEED STATISTICS (km/h):
-----------------------------
    Mean: {wind_mean:.2f}
    Min: {wind_min:.2f}
    Max: {wind_max:.2f}

================================================================================
                © 2024 RSK World - https://rskworld.in
================================================================================
"""
    
    formatted_report = report.format(
        total_records=len(df),
        locations=', '.join(df['location'].unique()),
        date_min=df['datetime'].min().strftime('%Y-%m-%d') if 'datetime' in df.columns else 'N/A',
        date_max=df['datetime'].max().strftime('%Y-%m-%d') if 'datetime' in df.columns else 'N/A',
        temp_mean=df['temperature_celsius'].mean(),
        temp_min=df['temperature_celsius'].min(),
        temp_max=df['temperature_celsius'].max(),
        temp_std=df['temperature_celsius'].std(),
        humid_mean=df['humidity_percent'].mean(),
        humid_min=df['humidity_percent'].min(),
        humid_max=df['humidity_percent'].max(),
        humid_std=df['humidity_percent'].std(),
        press_mean=df['pressure_hpa'].mean(),
        press_min=df['pressure_hpa'].min(),
        press_max=df['pressure_hpa'].max(),
        wind_mean=df['wind_speed_kmh'].mean(),
        wind_min=df['wind_speed_kmh'].min(),
        wind_max=df['wind_speed_kmh'].max()
    )
    
    return formatted_report


def main():
    """
    Main function to demonstrate data analysis capabilities.
    
    Author: Molla Samser | Website: https://rskworld.in
    """
    print("""
================================================================================
            WEATHER FORECASTING DATASET - DATA ANALYSIS
================================================================================
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
================================================================================
    """)
    
    # Define data path
    data_path = '../data/weather_data.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
        print("Please ensure the weather_data.csv file is in the data directory.")
        return
    
    # Load data
    print("Loading weather data...")
    df = load_weather_data(data_path)
    print(f"Loaded {len(df)} records")
    
    # Generate summary report
    print("\n" + weather_summary_report(df))
    
    # Location-wise analysis
    print("\nLocation-wise Temperature Summary:")
    print("-" * 50)
    for location in df['location'].unique():
        loc_data = df[df['location'] == location]
        print(f"{location}:")
        print(f"  Avg Temp: {loc_data['temperature_celsius'].mean():.1f}°C")
        print(f"  Avg Humidity: {loc_data['humidity_percent'].mean():.1f}%")
        print()
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Uncomment the following lines to generate plots
    # plot_temperature_trends(df, 'temperature_trends.png')
    # plot_humidity_distribution(df, 'humidity_distribution.png')
    # plot_correlation_heatmap(df, 'correlation_heatmap.png')
    # plot_daily_patterns(df, 'temperature_celsius', 'daily_patterns.png')
    
    print("\nAnalysis complete!")
    print("© 2024 RSK World - https://rskworld.in")


if __name__ == "__main__":
    main()

