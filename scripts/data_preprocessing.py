"""
================================================================================
Weather Forecasting Dataset - Data Preprocessing Script

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
This script provides data preprocessing functions for the Weather Forecasting 
Dataset. It includes functions for data cleaning, feature engineering, and 
preparing data for machine learning models.

License: Educational Purpose Only
Content used for educational purposes only.

© 2024 RSK World - https://rskworld.in
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class WeatherDataPreprocessor:
    """
    A class for preprocessing weather forecasting data.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    
    Attributes:
    -----------
    df : pd.DataFrame
        The weather data DataFrame
    scaler : object
        Sklearn scaler object for normalization
    label_encoders : dict
        Dictionary of label encoders for categorical columns
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the WeatherDataPreprocessor.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file
        df : pd.DataFrame, optional
            Existing DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = pd.read_csv(data_path, comment='#')
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.scaler = None
        self.label_encoders = {}
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the weather data by handling missing values and duplicates.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                # Fill with median for numerical columns
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Handle categorical missing values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                # Fill with mode for categorical columns
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print(f"Data cleaned. Shape: {self.df.shape}")
        return self.df
    
    def create_datetime_features(self) -> pd.DataFrame:
        """
        Create datetime-based features from date and time columns.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with new datetime features
        """
        if 'date' in self.df.columns and 'time' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'])
        elif 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        if 'datetime' in self.df.columns:
            self.df['year'] = self.df['datetime'].dt.year
            self.df['month'] = self.df['datetime'].dt.month
            self.df['day'] = self.df['datetime'].dt.day
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical encoding for hour and month
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        print("Datetime features created.")
        return self.df
    
    def encode_wind_direction(self) -> pd.DataFrame:
        """
        Encode wind direction as cyclical features.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded wind direction
        """
        # Wind direction mapping to degrees
        direction_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        if 'wind_direction' in self.df.columns:
            self.df['wind_direction_deg'] = self.df['wind_direction'].map(direction_map)
            # Handle unmapped values
            self.df['wind_direction_deg'].fillna(0, inplace=True)
            
            # Create cyclical features
            self.df['wind_dir_sin'] = np.sin(np.radians(self.df['wind_direction_deg']))
            self.df['wind_dir_cos'] = np.cos(np.radians(self.df['wind_direction_deg']))
        
        print("Wind direction encoded.")
        return self.df
    
    def encode_categorical(self, columns: list = None) -> pd.DataFrame:
        """
        Encode categorical columns using Label Encoding.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to encode. If None, encodes all object columns.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded categorical columns
        """
        if columns is None:
            columns = ['location', 'weather_condition']
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Categorical columns encoded: {columns}")
        return self.df
    
    def normalize_features(self, columns: list = None, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to normalize
        method : str
            Normalization method: 'standard' or 'minmax'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with normalized features
        """
        if columns is None:
            columns = ['temperature_celsius', 'humidity_percent', 'pressure_hpa',
                      'wind_speed_kmh', 'cloud_cover_percent', 'visibility_km']
        
        available_cols = [col for col in columns if col in self.df.columns]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        scaled_data = self.scaler.fit_transform(self.df[available_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in available_cols])
        
        self.df = pd.concat([self.df.reset_index(drop=True), scaled_df], axis=1)
        
        print(f"Features normalized using {method} method.")
        return self.df
    
    def create_lag_features(self, column: str, lags: list = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Create lag features for time series analysis.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        column : str
            Column to create lag features for
        lags : list
            List of lag periods
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with lag features
        """
        if column not in self.df.columns:
            print(f"Column {column} not found.")
            return self.df
        
        for location in self.df['location'].unique():
            mask = self.df['location'] == location
            for lag in lags:
                self.df.loc[mask, f'{column}_lag_{lag}'] = self.df.loc[mask, column].shift(lag)
        
        print(f"Lag features created for {column}.")
        return self.df
    
    def create_rolling_features(self, column: str, windows: list = [3, 6, 12]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        column : str
            Column to create rolling features for
        windows : list
            List of window sizes
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling features
        """
        if column not in self.df.columns:
            print(f"Column {column} not found.")
            return self.df
        
        for location in self.df['location'].unique():
            mask = self.df['location'] == location
            for window in windows:
                self.df.loc[mask, f'{column}_rolling_mean_{window}'] = \
                    self.df.loc[mask, column].rolling(window=window).mean()
                self.df.loc[mask, f'{column}_rolling_std_{window}'] = \
                    self.df.loc[mask, column].rolling(window=window).std()
        
        print(f"Rolling features created for {column}.")
        return self.df
    
    def prepare_for_ml(self, target_column: str, feature_columns: list = None,
                       test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Prepare data for machine learning.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        target_column : str
            Target variable column name
        feature_columns : list, optional
            List of feature columns
        test_size : float
            Test set size ratio
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Drop rows with NaN values
        df_clean = self.df.dropna()
        
        if feature_columns is None:
            # Select numeric columns except target
            feature_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Data prepared for ML:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
        """
        return self.df
    
    def save_processed_data(self, filepath: str):
        """
        Save processed data to CSV.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        filepath : str
            Output file path
        """
        self.df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")


def main():
    """
    Main function demonstrating preprocessing capabilities.
    
    Author: Molla Samser | Website: https://rskworld.in
    """
    print("""
================================================================================
        WEATHER FORECASTING DATASET - DATA PREPROCESSING
================================================================================
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
================================================================================
    """)
    
    # Example usage
    data_path = '../data/weather_data.csv'
    
    try:
        # Initialize preprocessor
        preprocessor = WeatherDataPreprocessor(data_path=data_path)
        
        # Apply preprocessing steps
        preprocessor.clean_data()
        preprocessor.create_datetime_features()
        preprocessor.encode_wind_direction()
        preprocessor.encode_categorical(['location', 'weather_condition'])
        preprocessor.normalize_features()
        preprocessor.create_lag_features('temperature_celsius', lags=[1, 3, 6])
        preprocessor.create_rolling_features('temperature_celsius', windows=[3, 6])
        
        # Get processed data
        processed_df = preprocessor.get_processed_data()
        print(f"\nProcessed data shape: {processed_df.shape}")
        print(f"\nNew columns created:")
        print(processed_df.columns.tolist())
        
        # Prepare for ML
        # X_train, X_test, y_train, y_test = preprocessor.prepare_for_ml(
        #     target_column='temperature_celsius'
        # )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists at the specified path.")
    
    print("\n" + "=" * 80)
    print("© 2024 RSK World - https://rskworld.in")
    print("=" * 80)


if __name__ == "__main__":
    main()

