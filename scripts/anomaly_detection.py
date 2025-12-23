"""
================================================================================
Weather Forecasting Dataset - Anomaly Detection

Project: Weather Forecasting Dataset
Category: Time Series Data
Version: 2.0.0
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
Anomaly detection algorithms for identifying unusual weather patterns,
extreme events, and data quality issues.

© 2024 RSK World - https://rskworld.in
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class WeatherAnomalyDetector:
    """
    Detect anomalies in weather data using multiple algorithms.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Algorithms:
    - Statistical methods (Z-score, IQR)
    - Isolation Forest
    - Local Outlier Factor
    - Moving Average Deviation
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the anomaly detector.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.anomalies = {}
        
    def detect_zscore_anomalies(self, column: str, threshold: float = 3.0) -> pd.Series:
        """
        Detect anomalies using Z-score method.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        column : str
            Column to analyze
        threshold : float
            Z-score threshold (default 3.0 for 99.7% confidence)
            
        Returns:
        --------
        pd.Series
            Boolean series indicating anomalies
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        anomalies = z_scores > threshold
        
        # Store results
        self.anomalies[f'{column}_zscore'] = {
            'method': 'Z-Score',
            'threshold': threshold,
            'anomaly_count': anomalies.sum(),
            'anomaly_percentage': (anomalies.sum() / len(anomalies)) * 100
        }
        
        return anomalies
    
    def detect_iqr_anomalies(self, column: str, multiplier: float = 1.5) -> pd.Series:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        column : str
            Column to analyze
        multiplier : float
            IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
            
        Returns:
        --------
        pd.Series
            Boolean series indicating anomalies
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomalies = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        self.anomalies[f'{column}_iqr'] = {
            'method': 'IQR',
            'multiplier': multiplier,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'anomaly_count': anomalies.sum(),
            'anomaly_percentage': (anomalies.sum() / len(anomalies)) * 100
        }
        
        return anomalies
    
    def detect_isolation_forest(self, columns: list, contamination: float = 0.05) -> pd.Series:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        columns : list
            List of columns to use for detection
        contamination : float
            Expected proportion of anomalies
            
        Returns:
        --------
        pd.Series
            Boolean series indicating anomalies
        """
        available_cols = [col for col in columns if col in self.df.columns]
        data = self.df[available_cols].dropna()
        
        scaled_data = self.scaler.fit_transform(data)
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(scaled_data)
        anomalies = predictions == -1
        
        self.anomalies['isolation_forest'] = {
            'method': 'Isolation Forest',
            'contamination': contamination,
            'columns_used': available_cols,
            'anomaly_count': anomalies.sum(),
            'anomaly_percentage': (anomalies.sum() / len(anomalies)) * 100
        }
        
        return pd.Series(anomalies, index=data.index)
    
    def detect_lof(self, columns: list, n_neighbors: int = 20,
                   contamination: float = 0.05) -> pd.Series:
        """
        Detect anomalies using Local Outlier Factor (LOF).
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        columns : list
            List of columns to use for detection
        n_neighbors : int
            Number of neighbors for LOF
        contamination : float
            Expected proportion of anomalies
            
        Returns:
        --------
        pd.Series
            Boolean series indicating anomalies
        """
        available_cols = [col for col in columns if col in self.df.columns]
        data = self.df[available_cols].dropna()
        
        scaled_data = self.scaler.fit_transform(data)
        
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(scaled_data)
        anomalies = predictions == -1
        
        self.anomalies['lof'] = {
            'method': 'Local Outlier Factor',
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'anomaly_count': anomalies.sum(),
            'anomaly_percentage': (anomalies.sum() / len(anomalies)) * 100
        }
        
        return pd.Series(anomalies, index=data.index)
    
    def detect_moving_average_deviation(self, column: str, window: int = 24,
                                        threshold: float = 2.0) -> pd.Series:
        """
        Detect anomalies using moving average deviation.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        column : str
            Column to analyze
        window : int
            Moving average window size
        threshold : float
            Standard deviation multiplier
            
        Returns:
        --------
        pd.Series
            Boolean series indicating anomalies
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")
        
        rolling_mean = self.df[column].rolling(window=window).mean()
        rolling_std = self.df[column].rolling(window=window).std()
        
        deviation = np.abs(self.df[column] - rolling_mean)
        anomalies = deviation > (threshold * rolling_std)
        
        self.anomalies[f'{column}_ma'] = {
            'method': 'Moving Average Deviation',
            'window': window,
            'threshold': threshold,
            'anomaly_count': anomalies.sum(),
            'anomaly_percentage': (anomalies.sum() / len(anomalies)) * 100
        }
        
        return anomalies
    
    def detect_extreme_weather(self) -> pd.DataFrame:
        """
        Detect extreme weather events based on predefined thresholds.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with extreme weather flags
        """
        extreme_events = pd.DataFrame(index=self.df.index)
        
        # Temperature extremes
        if 'temperature_celsius' in self.df.columns:
            extreme_events['heat_wave'] = self.df['temperature_celsius'] > 40
            extreme_events['cold_wave'] = self.df['temperature_celsius'] < 5
        
        # Humidity extremes
        if 'humidity_percent' in self.df.columns:
            extreme_events['very_dry'] = self.df['humidity_percent'] < 20
            extreme_events['very_humid'] = self.df['humidity_percent'] > 95
        
        # Wind extremes
        if 'wind_speed_kmh' in self.df.columns:
            extreme_events['high_wind'] = self.df['wind_speed_kmh'] > 50
            extreme_events['storm'] = self.df['wind_speed_kmh'] > 80
        
        # Visibility
        if 'visibility_km' in self.df.columns:
            extreme_events['poor_visibility'] = self.df['visibility_km'] < 1
            extreme_events['dense_fog'] = self.df['visibility_km'] < 0.5
        
        # Air quality
        if 'air_quality_index' in self.df.columns:
            extreme_events['poor_aqi'] = self.df['air_quality_index'] > 150
            extreme_events['hazardous_aqi'] = self.df['air_quality_index'] > 300
        
        # Heavy precipitation
        if 'precipitation_mm' in self.df.columns:
            extreme_events['heavy_rain'] = self.df['precipitation_mm'] > 50
            extreme_events['extreme_rain'] = self.df['precipitation_mm'] > 100
        
        return extreme_events
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """
        Get summary of all detected anomalies.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            Summary of anomaly detection results
        """
        if not self.anomalies:
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.anomalies).T
        return summary
    
    def generate_report(self) -> str:
        """
        Generate a text report of anomaly detection results.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        str
            Formatted report
        """
        report = """
================================================================================
           WEATHER ANOMALY DETECTION REPORT
================================================================================
    Project: Weather Forecasting Dataset
    Author: Molla Samser | Website: https://rskworld.in
    Email: help@rskworld.in | Phone: +91 93305 39277
================================================================================

SUMMARY OF DETECTED ANOMALIES:
------------------------------
"""
        for name, details in self.anomalies.items():
            report += f"\n{name.upper()}:\n"
            report += f"  Method: {details.get('method', 'N/A')}\n"
            report += f"  Anomalies Found: {details.get('anomaly_count', 0)}\n"
            report += f"  Percentage: {details.get('anomaly_percentage', 0):.2f}%\n"
        
        report += """
================================================================================
                © 2024 RSK World - https://rskworld.in
================================================================================
"""
        return report


def main():
    """
    Main function demonstrating anomaly detection.
    
    Author: Molla Samser | Website: https://rskworld.in
    """
    print("""
================================================================================
        WEATHER FORECASTING DATASET - ANOMALY DETECTION
================================================================================
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
================================================================================
    """)
    
    # Example usage
    data_path = '../data/weather_advanced.csv'
    
    try:
        df = pd.read_csv(data_path, comment='#')
        detector = WeatherAnomalyDetector(df)
        
        # Detect anomalies using different methods
        print("Detecting Z-score anomalies in temperature...")
        detector.detect_zscore_anomalies('temperature_celsius')
        
        print("Detecting IQR anomalies in humidity...")
        detector.detect_iqr_anomalies('humidity_percent')
        
        print("Detecting Isolation Forest anomalies...")
        columns = ['temperature_celsius', 'humidity_percent', 'pressure_hpa', 'wind_speed_kmh']
        detector.detect_isolation_forest(columns)
        
        print("Detecting moving average anomalies...")
        detector.detect_moving_average_deviation('temperature_celsius')
        
        # Get extreme weather events
        print("Detecting extreme weather events...")
        extreme_events = detector.detect_extreme_weather()
        
        # Generate report
        print(detector.generate_report())
        
        # Print extreme events summary
        print("\nExtreme Weather Events Summary:")
        print(extreme_events.sum())
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("© 2024 RSK World - https://rskworld.in")
    print("=" * 70)


if __name__ == "__main__":
    main()

