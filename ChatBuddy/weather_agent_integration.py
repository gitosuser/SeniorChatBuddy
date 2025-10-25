"""
Weather Agent Integration
Integrates partner's weather assistant into Chat Buddy
"""

import sys
import os
import logging
import requests
import json
from typing import Dict, Any

# Add partner's scripts directory to path
partner_scripts_path = "/Users/aparnaseetharaman/projects/WIBD/SeniorChatBuddy/Senior-Companion-Agent-/scripts"
sys.path.append(partner_scripts_path)

try:
    from weather_assistant import extract_weather_intent, get_open_meteo_forecast
except ImportError as e:
    print(f"Warning: Could not import weather assistant: {e}")
    print("Make sure the partner's weather agent is available at the expected path")


class WeatherAgent:
    """
    Integrated weather agent using partner's implementation
    """
    
    def __init__(self):
        self.default_location = "San Francisco, CA"  # Better fallback location
        self.detected_location = None
        self.location_cache_file = "logs/detected_location.txt"
        self.is_available = self._check_availability()
        self._setup_logging()
        
        # Try to detect user's location (with caching)
        self._detect_location()
        
        if self.is_available:
            print("✅ Weather agent integrated successfully")
            self.logger.info("Weather agent initialized successfully")
        else:
            print("⚠️ Weather agent not available - using fallback")
            self.logger.warning("Weather agent not available - using fallback")
    
    def _setup_logging(self):
        """Setup logging for weather agent operations"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('weather_agent')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler with daily rotation
        log_filename = f"logs/weather_agent_{os.path.basename(os.getcwd())}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _detect_location(self):
        """Detect user's location using IP geolocation with caching"""
        # First, try to load from cache
        cached_location = self._load_cached_location()
        if cached_location:
            self.detected_location = cached_location
            self.logger.info(f"[LOCATION_CACHED] Using cached location: {self.detected_location}")
            print(f"📍 Using cached location: {self.detected_location}")
            return
        
        # If no cache, try to detect location
        try:
            self.logger.info("[LOCATION_DETECTION] Attempting to detect user location...")
            
            # Use ipapi.co (free, no API key required)
            response = requests.get("https://ipapi.co/json/", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                city = data.get('city', '')
                region = data.get('region', '')
                country = data.get('country_name', '')
                
                if city and region:
                    self.detected_location = f"{city}, {region}"
                    self.logger.info(f"[LOCATION_DETECTED] {self.detected_location}")
                    print(f"📍 Detected location: {self.detected_location}")
                    # Cache the successful detection
                    self._cache_location(self.detected_location)
                elif city and country:
                    self.detected_location = f"{city}, {country}"
                    self.logger.info(f"[LOCATION_DETECTED] {self.detected_location}")
                    print(f"📍 Detected location: {self.detected_location}")
                    # Cache the successful detection
                    self._cache_location(self.detected_location)
                else:
                    self.logger.warning("[LOCATION_DETECTION] Could not parse location data")
            else:
                self.logger.warning(f"[LOCATION_DETECTION] API returned status {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"[LOCATION_DETECTION] Failed to detect location: {e}")
            print(f"⚠️ Could not detect location: {e}")
        
        # If detection failed, use default
        if not self.detected_location:
            self.detected_location = self.default_location
            self.logger.info(f"[LOCATION_FALLBACK] Using default location: {self.default_location}")
    
    def _load_cached_location(self):
        """Load cached location from file"""
        try:
            if os.path.exists(self.location_cache_file):
                with open(self.location_cache_file, 'r') as f:
                    cached_location = f.read().strip()
                    if cached_location:
                        return cached_location
        except Exception as e:
            self.logger.warning(f"[LOCATION_CACHE] Failed to load cached location: {e}")
        return None
    
    def _cache_location(self, location):
        """Cache detected location to file"""
        try:
            with open(self.location_cache_file, 'w') as f:
                f.write(location)
            self.logger.info(f"[LOCATION_CACHE] Cached location: {location}")
        except Exception as e:
            self.logger.warning(f"[LOCATION_CACHE] Failed to cache location: {e}")
    
    def _check_availability(self) -> bool:
        """Check if partner's weather agent is available"""
        try:
            # Test import and basic functionality
            test_intent = extract_weather_intent("What's the weather like?")
            return True
        except Exception as e:
            print(f"Weather agent check failed: {e}")
            return False
    
    def handle(self, user_input: str, chat_history: Dict[str, Any]) -> str:
        """
        Handle weather-related user input using partner's weather agent
        
        Args:
            user_input: User's weather question
            chat_history: Conversation history (not used by weather agent)
            
        Returns:
            Weather forecast response
        """
        # Log the incoming user input
        self.logger.info(f"[WEATHER_INPUT] {user_input}")
        
        if not self.is_available:
            self.logger.warning("[WEATHER_FALLBACK] Weather agent not available")
            return self._get_fallback_response()
        
        try:
            # Extract weather intent using partner's NLP
            self.logger.info("[WEATHER_PROCESSING] Extracting weather intent...")
            intent = extract_weather_intent(user_input, self.detected_location)
            
            # Fix: Ensure location is never None
            if not intent["location"]:
                intent["location"] = self.detected_location
                self.logger.info(f"[WEATHER_FIX] Location was None, using detected: {self.detected_location}")
            
            # Fix: Clean location format for Open-Meteo API
            original_location = intent["location"]
            intent["location"] = self._clean_location_for_api(intent["location"])
            if original_location != intent["location"]:
                self.logger.info(f"[WEATHER_FIX] Cleaned location: '{original_location}' → '{intent['location']}'")
            
            # Log the extracted intent details
            self.logger.info(f"[WEATHER_INTENT] location='{intent['location']}', time_phrase='{intent['time_phrase']}', start_hour={intent['start_hour']}, end_hour={intent['end_hour']}")
            
            # Get weather forecast using partner's API integration
            self.logger.info(f"[WEATHER_API] Calling Open-Meteo API for {intent['location']}...")
            forecast = get_open_meteo_forecast(
                intent["location"],
                intent["start_hour"],
                intent["end_hour"]
            )
            
            # Log the API response
            self.logger.info(f"[WEATHER_RESPONSE] {forecast}")
            
            # Return the senior-friendly forecast
            return forecast
            
        except Exception as e:
            error_msg = f"Weather agent error: {e}"
            print(error_msg)
            self.logger.error(f"[WEATHER_ERROR] {error_msg}")
            return self._get_error_response(str(e))
    
    def _get_fallback_response(self) -> str:
        """Fallback response when weather agent is not available"""
        response = (
            "I'm sorry, I'm having trouble accessing weather information right now. "
            "Please try again later, or check your local weather app."
        )
        self.logger.info(f"[WEATHER_FALLBACK_RESPONSE] {response}")
        return response
    
    def _get_error_response(self, error_msg: str) -> str:
        """Error response when weather agent fails"""
        response = (
            "I encountered an issue getting weather information. "
            "Please try rephrasing your question or check back in a few minutes."
        )
        self.logger.info(f"[WEATHER_ERROR_RESPONSE] {response}")
        return response
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about weather agent capabilities"""
        return {
            "available": self.is_available,
            "features": [
                "Automatic location detection with caching",
                "Natural language time parsing",
                "Location extraction with fallback",
                "Real-time weather data",
                "Senior-friendly clothing advice",
                "Temperature and precipitation forecasts"
            ],
            "detected_location": self.detected_location,
            "default_location": self.default_location,
            "data_source": "Open-Meteo API"
        }
    
    def clear_location_cache(self):
        """Clear the cached location to force re-detection"""
        try:
            if os.path.exists(self.location_cache_file):
                os.remove(self.location_cache_file)
                self.logger.info("[LOCATION_CACHE] Cleared cached location")
                print("🗑️ Location cache cleared - will re-detect on next startup")
        except Exception as e:
            self.logger.warning(f"[LOCATION_CACHE] Failed to clear cache: {e}")
            print(f"⚠️ Failed to clear location cache: {e}")
    
    def _clean_location_for_api(self, location):
        """Clean location format for Open-Meteo geocoding API"""
        if not location:
            return location
        
        # Remove state/country suffixes that cause issues
        # "San Francisco, CA" → "San Francisco"
        cleaned = location.split(',')[0].strip()
        
        # Handle common problematic formats
        if cleaned.endswith(', CA'):
            cleaned = cleaned[:-4].strip()
        elif cleaned.endswith(', CA'):
            cleaned = cleaned[:-4].strip()
        
        return cleaned


# Test the integration
if __name__ == "__main__":
    weather_agent = WeatherAgent()
    
    # Test cases
    test_questions = [
        "What's the weather like today?",
        "Will it rain tomorrow morning?",
        "Do I need an umbrella in Newark?",
        "Is it cold outside?"
    ]
    
    print("🧪 Testing Weather Agent Integration")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        response = weather_agent.handle(question, {})
        print(f"A: {response}")
    
    print(f"\n📊 Capabilities: {weather_agent.get_capabilities()}")
