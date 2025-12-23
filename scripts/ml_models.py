"""
================================================================================
Weather Forecasting Dataset - Machine Learning Models

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
Advanced machine learning models for weather prediction including
LSTM, XGBoost, Random Forest, and ensemble methods.

© 2024 RSK World - https://rskworld.in
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")


class WeatherPredictionModels:
    """
    A comprehensive class for weather prediction using various ML models.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Features:
    - Multiple model support (Linear, RF, XGBoost, LSTM)
    - Automatic feature engineering
    - Cross-validation
    - Hyperparameter tuning
    - Model comparison
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the WeatherPredictionModels class.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        data_path : str, optional
            Path to CSV file
        df : pd.DataFrame, optional
            Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.df = pd.read_csv(data_path, comment='#')
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_features(self, target: str = 'temperature_celsius', 
                        look_back: int = 24) -> tuple:
        """
        Prepare features for ML models with time series consideration.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        target : str
            Target variable to predict
        look_back : int
            Number of previous time steps to use
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        df = self.df.copy()
        
        # Create datetime features
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Select numeric features
        feature_cols = [
            'humidity_percent', 'pressure_hpa', 'wind_speed_kmh',
            'cloud_cover_percent', 'visibility_km', 'uv_index',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Handle missing values
        df = df.dropna(subset=available_features + [target])
        
        X = df[available_features].values
        y = df[target].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.feature_names = available_features
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, y_train) -> LinearRegression:
        """
        Train Linear Regression model.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_ridge_regression(self, X_train, y_train, alpha: float = 1.0) -> Ridge:
        """
        Train Ridge Regression model.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['ridge_regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators: int = 100,
                           max_depth: int = 10) -> RandomForestRegressor:
        """
        Train Random Forest Regressor.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators: int = 100,
                               learning_rate: float = 0.1) -> GradientBoostingRegressor:
        """
        Train Gradient Boosting Regressor.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, n_estimators: int = 100,
                     learning_rate: float = 0.1):
        """
        Train XGBoost Regressor.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Skipping...")
            return None
        
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_svr(self, X_train, y_train, kernel: str = 'rbf', C: float = 1.0) -> SVR:
        """
        Train Support Vector Regressor.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        model = SVR(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        self.models['svr'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate a trained model.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_test : array
            Test features
        y_test : array
            True values
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """
        Train and evaluate all available models.
        
        Author: Molla Samser | Website: https://rskworld.in
        
        Returns:
        --------
        pd.DataFrame
            Comparison of all models
        """
        print("Training Linear Regression...")
        self.train_linear_regression(X_train, y_train)
        self.evaluate_model(self.models['linear_regression'], X_test, y_test, 'Linear Regression')
        
        print("Training Ridge Regression...")
        self.train_ridge_regression(X_train, y_train)
        self.evaluate_model(self.models['ridge_regression'], X_test, y_test, 'Ridge Regression')
        
        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        self.evaluate_model(self.models['random_forest'], X_test, y_test, 'Random Forest')
        
        print("Training Gradient Boosting...")
        self.train_gradient_boosting(X_train, y_train)
        self.evaluate_model(self.models['gradient_boosting'], X_test, y_test, 'Gradient Boosting')
        
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            self.train_xgboost(X_train, y_train)
            self.evaluate_model(self.models['xgboost'], X_test, y_test, 'XGBoost')
        
        print("Training SVR...")
        self.train_svr(X_train, y_train)
        self.evaluate_model(self.models['svr'], X_test, y_test, 'SVR')
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('rmse')
        
        return results_df
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_imp
        else:
            print(f"Model {model_name} does not support feature importance.")
            return None
    
    def hyperparameter_tuning(self, X_train, y_train, model_type: str = 'random_forest'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestRegressor(random_state=42)
        elif model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingRegressor(random_state=42)
        else:
            print(f"Tuning not implemented for {model_type}")
            return None
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters for {model_type}:")
        print(grid_search.best_params_)
        print(f"Best score: {np.sqrt(-grid_search.best_score_):.4f} RMSE")
        
        return grid_search.best_estimator_
    
    def predict(self, X, model_name: str = 'random_forest'):
        """
        Make predictions using a trained model.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        X_scaled = self.scaler.transform(X)
        return self.models[model_name].predict(X_scaled)


class LSTMWeatherPredictor:
    """
    LSTM-based weather prediction model for time series forecasting.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    """
    
    def __init__(self, look_back: int = 24, features: int = 1):
        """
        Initialize LSTM predictor.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.look_back = look_back
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
    
    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        Create sequences for LSTM training.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i:(i + self.look_back)])
            y.append(data[i + self.look_back])
        return np.array(X), np.array(y)
    
    def build_model(self, units: int = 50):
        """
        Build LSTM model architecture.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        self.model = Sequential([
            LSTM(units, return_sequences=True, 
                 input_shape=(self.look_back, self.features)),
            Dropout(0.2),
            LSTM(units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model
    
    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """
        Train LSTM model.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained LSTM model.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X, _ = self.create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)


def main():
    """
    Main function demonstrating ML model capabilities.
    
    Author: Molla Samser | Website: https://rskworld.in
    """
    print("""
================================================================================
        WEATHER FORECASTING DATASET - MACHINE LEARNING MODELS
================================================================================
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
================================================================================
    """)
    
    # Example usage
    data_path = '../data/weather_advanced.csv'
    
    try:
        # Initialize predictor
        predictor = WeatherPredictionModels(data_path=data_path)
        
        # Prepare features
        print("Preparing features...")
        X_train, X_test, y_train, y_test = predictor.prepare_features(
            target='temperature_celsius'
        )
        
        # Train and evaluate all models
        print("\nTraining all models...")
        results = predictor.train_all_models(X_train, X_test, y_train, y_test)
        
        print("\n" + "=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)
        print(results.to_string())
        
        # Get feature importance
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE (Random Forest)")
        print("=" * 70)
        importance = predictor.get_feature_importance('random_forest')
        if importance is not None:
            print(importance.to_string())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists at the specified path.")
    
    print("\n" + "=" * 70)
    print("© 2024 RSK World - https://rskworld.in")
    print("=" * 70)


if __name__ == "__main__":
    main()

