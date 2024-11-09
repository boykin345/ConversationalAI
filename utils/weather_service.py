# utils/weather_service.py

import requests
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

def get_coordinates(city):
    """Get coordinates for a given city using Open-Meteo Geocoding API."""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return result['latitude'], result['longitude'], result['name'], result.get('country', '')
        return None
    except Exception:
        return None

def get_weather(location=None):
    """Fetch current weather data for a specific location."""
    if not location:
        return "Please specify a location. For example: 'What's the weather in London?'"

    coordinates = get_coordinates(location)
    if not coordinates:
        return f"I couldn't find the location '{location}'. Please try another city."

    lat, lon, city, country = coordinates
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=celsius"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            current_weather = data.get('current_weather', {})
            temperature = current_weather.get('temperature')
            windspeed = current_weather.get('windspeed')

            if temperature is not None:
                weather_description = "warm" if temperature > 20 else "mild" if temperature > 10 else "cold"
                location_name = f"{city}, {country}" if country else city
                return f"The current temperature in {location_name} is {temperature}°C ({weather_description}) with a wind speed of {windspeed} km/h."

        return "I'm sorry, I couldn't fetch the weather information at the moment."
    except Exception:
        return "Sorry, there was an error getting the weather data."

def get_time_in_location(location):
    """Get current time in the specified location."""
    try:
        geolocator = Nominatim(user_agent="chatbot")
        geocode_result = geolocator.geocode(location)
        if geocode_result:
            timezone_finder = TimezoneFinder()
            timezone = timezone_finder.timezone_at(
                lng=geocode_result.longitude, lat=geocode_result.latitude)
            if timezone:
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz).strftime("%I:%M %p")
                return f"The current time in {location} is {current_time}"
        return f"Sorry, I couldn't find the time for {location}."
    except Exception as e:
        print(f"Error in get_time_in_location: {str(e)}")
        return "Sorry, there was an error getting the time."
