"""
================================================================================
Weather Forecasting Dataset - Weather API Simulation

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
A Flask-based weather API simulation that provides weather data through
RESTful endpoints. Perfect for testing and learning API development.

Usage:
    python weather_api.py
    
Endpoints:
    GET /api/weather/current?location=<city>
    GET /api/weather/forecast?location=<city>&days=<n>
    GET /api/weather/historical?location=<city>&date=<YYYY-MM-DD>
    GET /api/locations
    GET /api/alerts?location=<city>

© 2024 RSK World - https://rskworld.in
================================================================================
"""

import json
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

# Try importing Flask
try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. Install with: pip install flask")


# Weather data storage
LOCATIONS = {
    "new_delhi": {
        "name": "New Delhi",
        "country": "India",
        "state": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone": "Asia/Kolkata",
        "climate": "Humid Subtropical"
    },
    "mumbai": {
        "name": "Mumbai",
        "country": "India",
        "state": "Maharashtra",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "timezone": "Asia/Kolkata",
        "climate": "Tropical Wet and Dry"
    },
    "bangalore": {
        "name": "Bangalore",
        "country": "India",
        "state": "Karnataka",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "timezone": "Asia/Kolkata",
        "climate": "Tropical Savanna"
    },
    "kolkata": {
        "name": "Kolkata",
        "country": "India",
        "state": "West Bengal",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "timezone": "Asia/Kolkata",
        "climate": "Tropical Wet and Dry"
    },
    "chennai": {
        "name": "Chennai",
        "country": "India",
        "state": "Tamil Nadu",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "timezone": "Asia/Kolkata",
        "climate": "Tropical Wet and Dry"
    }
}

# Base weather patterns by location
WEATHER_PATTERNS = {
    "new_delhi": {
        "temp_base": 25, "temp_variance": 15,
        "humidity_base": 60, "humidity_variance": 30,
        "pressure_base": 1015, "pressure_variance": 10
    },
    "mumbai": {
        "temp_base": 28, "temp_variance": 8,
        "humidity_base": 70, "humidity_variance": 20,
        "pressure_base": 1012, "pressure_variance": 8
    },
    "bangalore": {
        "temp_base": 24, "temp_variance": 10,
        "humidity_base": 55, "humidity_variance": 25,
        "pressure_base": 1015, "pressure_variance": 6
    },
    "kolkata": {
        "temp_base": 26, "temp_variance": 12,
        "humidity_base": 70, "humidity_variance": 25,
        "pressure_base": 1017, "pressure_variance": 8
    },
    "chennai": {
        "temp_base": 29, "temp_variance": 8,
        "humidity_base": 65, "humidity_variance": 25,
        "pressure_base": 1012, "pressure_variance": 6
    }
}

CONDITIONS = ["Clear", "Sunny", "Partly Cloudy", "Cloudy", "Overcast", 
              "Light Rain", "Rain", "Thunderstorm", "Fog", "Haze", "Mist"]


class WeatherAPISimulator:
    """
    Weather API simulator for generating realistic weather data.
    
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    
    def __init__(self):
        """Initialize the weather API simulator."""
        self.locations = LOCATIONS
        self.patterns = WEATHER_PATTERNS
        
    def _normalize_location(self, location: str) -> str:
        """Normalize location name for lookup."""
        return location.lower().replace(" ", "_").replace("-", "_")
    
    def _generate_weather(self, location_key: str, hour: int = None) -> Dict:
        """
        Generate realistic weather data for a location.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        if location_key not in self.patterns:
            return None
        
        pattern = self.patterns[location_key]
        location = self.locations[location_key]
        
        if hour is None:
            hour = datetime.now().hour
        
        # Temperature varies by time of day
        hour_factor = -abs(hour - 14) / 14  # Warmest at 2 PM
        temp = pattern["temp_base"] + hour_factor * 8 + random.uniform(-3, 3)
        
        # Humidity inversely related to temperature
        humidity = pattern["humidity_base"] - hour_factor * 15 + random.uniform(-10, 10)
        humidity = max(20, min(100, humidity))
        
        # Pressure with small variations
        pressure = pattern["pressure_base"] + random.uniform(-pattern["pressure_variance"], 
                                                             pattern["pressure_variance"])
        
        # Wind speed
        wind_speed = random.uniform(2, 25)
        wind_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        wind_direction = random.choice(wind_directions)
        
        # Determine condition based on humidity and random factor
        if humidity > 85:
            condition = random.choice(["Rain", "Light Rain", "Fog"])
        elif humidity > 70:
            condition = random.choice(["Cloudy", "Partly Cloudy", "Overcast"])
        else:
            condition = random.choice(["Clear", "Sunny", "Partly Cloudy"])
        
        return {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "temperature": {
                "celsius": round(temp, 1),
                "fahrenheit": round(temp * 9/5 + 32, 1),
                "feels_like": round(temp - 2 + random.uniform(0, 4), 1)
            },
            "humidity": round(humidity, 1),
            "pressure": round(pressure, 1),
            "wind": {
                "speed_kmh": round(wind_speed, 1),
                "direction": wind_direction,
                "gust_kmh": round(wind_speed * 1.5 + random.uniform(0, 5), 1)
            },
            "visibility_km": round(random.uniform(5, 20), 1),
            "uv_index": max(0, min(11, int(random.uniform(0, 11) * (1 if hour > 6 and hour < 18 else 0.1)))),
            "cloud_cover": round(random.uniform(0, 100), 1),
            "condition": {
                "main": condition,
                "description": f"{condition} conditions in {location['name']}"
            },
            "air_quality": {
                "index": random.randint(30, 200),
                "pm25": round(random.uniform(10, 150), 1),
                "pm10": round(random.uniform(20, 200), 1)
            }
        }
    
    def get_current_weather(self, location: str) -> Dict:
        """
        Get current weather for a location.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        location_key = self._normalize_location(location)
        
        if location_key not in self.locations:
            return {
                "error": "Location not found",
                "available_locations": list(self.locations.keys()),
                "author": "Molla Samser | rskworld.in"
            }
        
        weather = self._generate_weather(location_key)
        weather["_meta"] = {
            "api_version": "2.0",
            "author": "Molla Samser",
            "website": "https://rskworld.in"
        }
        
        return weather
    
    def get_forecast(self, location: str, days: int = 5) -> Dict:
        """
        Get weather forecast for a location.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        location_key = self._normalize_location(location)
        
        if location_key not in self.locations:
            return {
                "error": "Location not found",
                "available_locations": list(self.locations.keys())
            }
        
        forecast = []
        current_date = datetime.now()
        
        for day in range(days):
            date = current_date + timedelta(days=day)
            daily_forecast = {
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "hourly": []
            }
            
            for hour in [6, 9, 12, 15, 18, 21]:
                hourly_weather = self._generate_weather(location_key, hour)
                hourly_weather["hour"] = f"{hour:02d}:00"
                daily_forecast["hourly"].append(hourly_weather)
            
            # Daily summary
            temps = [h["temperature"]["celsius"] for h in daily_forecast["hourly"]]
            daily_forecast["summary"] = {
                "temp_high": max(temps),
                "temp_low": min(temps),
                "temp_avg": round(sum(temps) / len(temps), 1)
            }
            
            forecast.append(daily_forecast)
        
        return {
            "location": self.locations[location_key],
            "forecast_days": days,
            "forecast": forecast,
            "_meta": {
                "api_version": "2.0",
                "author": "Molla Samser",
                "website": "https://rskworld.in"
            }
        }
    
    def get_alerts(self, location: str) -> Dict:
        """
        Get weather alerts for a location.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        location_key = self._normalize_location(location)
        
        if location_key not in self.locations:
            return {"error": "Location not found"}
        
        # Simulate alerts (randomly generate some alerts)
        alerts = []
        
        if random.random() > 0.7:
            alert_types = [
                {"type": "Heat Advisory", "severity": "Moderate"},
                {"type": "Air Quality Alert", "severity": "Moderate"},
                {"type": "Fog Warning", "severity": "Minor"},
                {"type": "Rain Expected", "severity": "Minor"}
            ]
            alert = random.choice(alert_types)
            alerts.append({
                "type": alert["type"],
                "severity": alert["severity"],
                "issued": datetime.now().isoformat(),
                "expires": (datetime.now() + timedelta(hours=12)).isoformat(),
                "description": f"{alert['type']} for {self.locations[location_key]['name']}"
            })
        
        return {
            "location": self.locations[location_key],
            "alerts_count": len(alerts),
            "alerts": alerts,
            "_meta": {
                "author": "Molla Samser",
                "website": "https://rskworld.in"
            }
        }
    
    def get_all_locations(self) -> Dict:
        """
        Get all available locations.
        
        Author: Molla Samser | Website: https://rskworld.in
        """
        return {
            "count": len(self.locations),
            "locations": self.locations,
            "_meta": {
                "author": "Molla Samser",
                "website": "https://rskworld.in"
            }
        }


# Flask App
if FLASK_AVAILABLE:
    app = Flask(__name__)
    weather_api = WeatherAPISimulator()
    
    @app.route('/')
    def home():
        """API Home endpoint."""
        return jsonify({
            "name": "Weather Forecasting API",
            "version": "2.0.0",
            "author": "Molla Samser",
            "website": "https://rskworld.in",
            "email": "help@rskworld.in",
            "endpoints": {
                "current_weather": "/api/weather/current?location=<city>",
                "forecast": "/api/weather/forecast?location=<city>&days=<n>",
                "alerts": "/api/alerts?location=<city>",
                "locations": "/api/locations"
            }
        })
    
    @app.route('/api/weather/current')
    def current_weather():
        """Get current weather."""
        location = request.args.get('location', 'new_delhi')
        return jsonify(weather_api.get_current_weather(location))
    
    @app.route('/api/weather/forecast')
    def forecast():
        """Get weather forecast."""
        location = request.args.get('location', 'new_delhi')
        days = int(request.args.get('days', 5))
        return jsonify(weather_api.get_forecast(location, days))
    
    @app.route('/api/alerts')
    def alerts():
        """Get weather alerts."""
        location = request.args.get('location', 'new_delhi')
        return jsonify(weather_api.get_alerts(location))
    
    @app.route('/api/locations')
    def locations():
        """Get all locations."""
        return jsonify(weather_api.get_all_locations())


def main():
    """
    Main function to run the Weather API.
    
    Author: Molla Samser | Website: https://rskworld.in
    """
    print("""
================================================================================
        WEATHER FORECASTING DATASET - API SERVER
================================================================================
    Author: Molla Samser
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
================================================================================
    """)
    
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Running in simulation mode.")
        print("\nSimulation Example:")
        
        api = WeatherAPISimulator()
        
        print("\n--- Current Weather for New Delhi ---")
        print(json.dumps(api.get_current_weather("New Delhi"), indent=2))
        
        print("\n--- All Locations ---")
        print(json.dumps(api.get_all_locations(), indent=2))
        
        return
    
    print("Starting Weather API Server...")
    print("API Endpoints:")
    print("  - http://localhost:5000/")
    print("  - http://localhost:5000/api/weather/current?location=mumbai")
    print("  - http://localhost:5000/api/weather/forecast?location=mumbai&days=3")
    print("  - http://localhost:5000/api/alerts?location=mumbai")
    print("  - http://localhost:5000/api/locations")
    print("\nPress Ctrl+C to stop the server.")
    
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()

