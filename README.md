<!--
================================================================================
Weather Forecasting Dataset - README

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

© 2024 RSK World - https://rskworld.in
================================================================================
-->

# 🌤️ Weather Forecasting Dataset

[![Version](https://img.shields.io/badge/Version-2.0.0-blue.svg)](https://rskworld.in)
[![Category](https://img.shields.io/badge/Category-Time%20Series%20Data-green.svg)](https://rskworld.in)
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow.svg)](https://rskworld.in)
[![License](https://img.shields.io/badge/License-Educational-orange.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Molla%20Samser-purple.svg)](https://rskworld.in)

A comprehensive weather dataset with temperature, humidity, pressure, and precipitation data for weather forecasting models. Perfect for **machine learning**, **time series analysis**, and **climate research**.

---

## 📊 Dataset Overview

This dataset includes historical weather data with temperature, humidity, pressure, precipitation, wind speed, air quality, solar radiation, and other meteorological variables. Perfect for weather forecasting, climate analysis, and time series prediction models.

### Key Features

- 🌡️ **Temperature & humidity data** - Hourly measurements with feels-like temperatures
- 🌧️ **Pressure & precipitation** - Atmospheric pressure and rainfall data
- 💨 **Wind speed & direction** - Including gust measurements
- 📍 **Multiple locations** - 5 major Indian cities
- ⏱️ **Time series format** - Perfect for ML models
- 🌬️ **Air Quality Index (AQI)** - PM2.5, PM10, Ozone, NO2, SO2, CO
- ☀️ **Solar radiation data** - UV index and solar radiation measurements
- 🌱 **Soil conditions** - Temperature and moisture data
- 🔮 **Forecast comparisons** - Predicted vs actual data for model evaluation
- 🚨 **Weather alerts** - Historical alert and warning data

---

## 🗂️ Project Structure

```
weather-forecasting/
├── 📁 data/
│   ├── weather_data.csv              # Main weather data (120+ records)
│   ├── weather_advanced.csv          # Advanced data with 35+ parameters
│   ├── weather_alerts.csv            # Weather alerts and warnings
│   ├── weather_forecast_comparison.csv # Forecast vs actual comparison
│   ├── seasonal_statistics.csv       # Monthly/seasonal statistics
│   ├── locations.csv                 # Location information
│   ├── weather_hourly_extended.csv   # Extended hourly data
│   ├── weather_data.json             # JSON format data
│   └── dataset_info.json             # Dataset metadata
│
├── 📁 scripts/
│   ├── weather_analysis.py           # Data analysis & visualization
│   ├── data_preprocessing.py         # ML preprocessing utilities
│   ├── ml_models.py                  # Machine learning models
│   ├── anomaly_detection.py          # Anomaly detection algorithms
│   └── requirements.txt              # Python dependencies
│
├── 📁 notebooks/
│   └── weather_analysis.ipynb        # Jupyter notebook for analysis
│
├── 📁 api/
│   └── weather_api.py                # Weather API simulation (Flask)
│
├── 📄 index.html                     # Demo page
├── 📄 dashboard.html                 # Advanced analytics dashboard
├── 📄 preview-generator.html         # Preview image generator
├── 📄 README.md                      # This file
├── 📄 LICENSE                        # Educational license
├── 📄 CHANGELOG.md                   # Version history
├── 📄 .gitignore                     # Git ignore rules
├── 🖼️ weather-forecasting.png        # Project preview image
└── 🖼️ weather-forecasting.svg        # Project icon
```

---

## 📈 Data Features

### Basic Weather Data
| Feature | Description | Unit |
|---------|-------------|------|
| temperature_celsius | Air temperature | °C |
| humidity_percent | Relative humidity | % |
| pressure_hpa | Atmospheric pressure | hPa |
| wind_speed_kmh | Wind speed | km/h |
| wind_direction | Wind direction | Cardinal |
| precipitation_mm | Precipitation | mm |
| cloud_cover_percent | Cloud coverage | % |
| visibility_km | Visibility | km |
| uv_index | UV index | 0-11 |

### Advanced Parameters (v2.0)
| Feature | Description | Unit |
|---------|-------------|------|
| feels_like_celsius | Feels-like temperature | °C |
| air_quality_index | Air quality index | AQI |
| pm25 | Fine particulate matter | µg/m³ |
| pm10 | Coarse particulate matter | µg/m³ |
| ozone_ppb | Ground-level ozone | ppb |
| solar_radiation_wm2 | Solar radiation | W/m² |
| soil_temperature | Soil temperature | °C |
| soil_moisture | Soil moisture | % |
| heat_index | Heat index | °C |
| wind_chill | Wind chill factor | °C |
| dew_point | Dew point temperature | °C |

---

## 🤖 Machine Learning Models

The dataset includes ready-to-use ML models:

### Available Models
- **Linear Regression** - Baseline predictions
- **Ridge/Lasso Regression** - Regularized models
- **Random Forest** - Ensemble tree-based
- **Gradient Boosting** - Sequential boosting
- **XGBoost** - Optimized gradient boosting
- **Support Vector Regression** - Kernel-based
- **LSTM Neural Network** - Deep learning for time series

### Usage Example

```python
from scripts.ml_models import WeatherPredictionModels

# Initialize
predictor = WeatherPredictionModels(data_path='data/weather_advanced.csv')

# Prepare features
X_train, X_test, y_train, y_test = predictor.prepare_features(
    target='temperature_celsius'
)

# Train all models and compare
results = predictor.train_all_models(X_train, X_test, y_train, y_test)
print(results)

# Get feature importance
importance = predictor.get_feature_importance('random_forest')
```

---

## 🔍 Anomaly Detection

Detect unusual weather patterns using multiple algorithms:

```python
from scripts.anomaly_detection import WeatherAnomalyDetector

detector = WeatherAnomalyDetector(df)

# Z-score anomalies
detector.detect_zscore_anomalies('temperature_celsius')

# Isolation Forest
detector.detect_isolation_forest(['temperature_celsius', 'humidity_percent'])

# Generate report
print(detector.generate_report())
```

---

## 🌐 Weather API

A Flask-based API simulation for testing:

```bash
# Start the API server
cd api
python weather_api.py

# Endpoints:
# GET /api/weather/current?location=mumbai
# GET /api/weather/forecast?location=delhi&days=5
# GET /api/alerts?location=kolkata
# GET /api/locations
```

---

## 🚀 Quick Start

### 1. Clone or Download

```bash
# Download from RSK World
https://rskworld.in/data-science/datasets/weather-forecasting/
```

### 2. Install Dependencies

```bash
cd weather-forecasting
pip install -r scripts/requirements.txt
```

### 3. Run Analysis

```python
import pandas as pd

# Load data
df = pd.read_csv('data/weather_data.csv', comment='#')

# Explore
print(df.head())
print(df.describe())

# Analysis
print(df.groupby('location')['temperature_celsius'].mean())
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook notebooks/weather_analysis.ipynb
```

### 5. View Dashboard

Open `dashboard.html` in a browser for interactive analytics.

---

## 📍 Locations

| City | State | Latitude | Longitude | Climate |
|------|-------|----------|-----------|---------|
| New Delhi | Delhi | 28.6139 | 77.2090 | Humid Subtropical |
| Mumbai | Maharashtra | 19.0760 | 72.8777 | Tropical Wet-Dry |
| Bangalore | Karnataka | 12.9716 | 77.5946 | Tropical Savanna |
| Kolkata | West Bengal | 22.5726 | 88.3639 | Tropical Wet-Dry |
| Chennai | Tamil Nadu | 13.0827 | 80.2707 | Tropical Wet-Dry |

---

## 📚 Use Cases

1. **Weather Prediction** - Build ML models to predict temperature, humidity, etc.
2. **Time Series Forecasting** - ARIMA, LSTM, Prophet models
3. **Anomaly Detection** - Identify extreme weather events
4. **Climate Analysis** - Study seasonal patterns and trends
5. **Air Quality Monitoring** - Analyze pollution patterns
6. **Energy Consumption** - Correlate weather with energy usage
7. **Agriculture Planning** - Use soil and weather data for farming decisions
8. **Health Impact Studies** - Study weather effects on health

---

## 🛠️ Technologies

- **Python** - Data analysis and ML
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Machine learning
- **TensorFlow/Keras** - Deep learning (LSTM)
- **XGBoost** - Gradient boosting
- **Flask** - API development
- **Chart.js** - Interactive charts
- **HTML/CSS/JS** - Dashboard

---

## 👨‍💻 Author

**Molla Samser**  
Founder & Developer @ RSK World

- 🌐 Website: [https://rskworld.in](https://rskworld.in)
- 📧 Email: help@rskworld.in | support@rskworld.in
- 📱 Phone: +91 93305 39277

### Designer & Tester
**Rima Khatun**

---

## 📄 License

This project is licensed under the Educational License - see the [LICENSE](LICENSE) file for details.

Free for educational and non-commercial use.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

### v2.0.0 (Latest)
- Added advanced weather parameters (AQI, solar, soil data)
- Added ML prediction models
- Added anomaly detection
- Added Weather API simulation
- Added interactive dashboard
- Added Jupyter notebook
- Added seasonal statistics
- Added forecast comparison data

---

## ⭐ Support

If you find this dataset helpful, please give it a star and share with others!

For questions or support, contact us at help@rskworld.in

---

<div align="center">

**© 2024 RSK World - https://rskworld.in**

Made with ❤️ by Molla Samser

</div>
