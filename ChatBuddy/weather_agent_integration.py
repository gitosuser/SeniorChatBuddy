"""
Weather Agent Integration
Integrates partner's weather assistant into Chat Buddy
Adds:
  - IP-based location detection + caching
  - Logging
Wraps:
  - Core weather logic from weather_assistant.WeatherAgent
    which returns {"reply": ..., "summary": ...}
"""

import sys
import os
import logging
import requests
from typing import Dict, Any, Optional

# Add partner's scripts directory to path
partner_scripts_path = "/Users/aparnaseetharaman/projects/WIBD/SeniorChatBuddy/Senior-Companion-Agent-/scripts"
sys.path.append(partner_scripts_path)

try:
    # Import the *new* weather assistant module you showed me.
    # It defines:
    #   - extract_weather_intent(...)
    #   - class WeatherAgent with .handle() -> {"reply","summary"}
    from weather_assistant import extract_weather_intent, WeatherAgent as CoreWeatherAgent
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import weather assistant core: {e}")
    print("Make sure weather_assistant.py is on sys.path and defines extract_weather_intent + WeatherAgent")
    CORE_AVAILABLE = False


class WeatherAgent:
    """
    High-level weather agent for ChatAgent.

    Responsibilities:
    - Detect and cache user location (city, state/country) using IP
    - Use core weather assistant (CoreWeatherAgent) to actually produce forecast
    - Return:
        {
            "reply":   <string we tell the user>,
            "summary": <short structured summary for follow-up questions>
        }

    ChatAgent expects this dict shape.
    """

    def __init__(self):
        # Reasonable fallback if we can't detect location
        self.default_location = "San Francisco, CA"

        # Detected/cached location string like "San Francisco, CA"
        self.detected_location: Optional[str] = None

        # Cache file for location so we don't re-IP-lookup on every boot
        self.location_cache_file = "logs/detected_location.txt"

        # Setup logging first so we can log during detection
        self._setup_logging()

        # Check whether we could import the core weather assistant
        self.is_available = CORE_AVAILABLE

        # Try to detect user's location once per process
        self._detect_location()

        if self.is_available:
            print("✅ Weather agent integrated successfully")
            self.logger.info("Weather agent initialized successfully")
        else:
            print("⚠️ Weather agent core not available - using fallback")
            self.logger.warning("Weather agent core not available - using fallback")

        # Create an instance of the core weather agent if available
        self.core_agent = CoreWeatherAgent() if self.is_available else None

    # ------------------------------------------------------------------ #
    # Logging setup
    # ------------------------------------------------------------------ #
    def _setup_logging(self):
        """Setup logging for weather agent operations"""
        os.makedirs('logs', exist_ok=True)

        self.logger = logging.getLogger('weather_agent')
        self.logger.setLevel(logging.INFO)

        log_filename = f"logs/weather_agent_{os.path.basename(os.getcwd())}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Avoid attaching multiple handlers if re-instantiated in same process
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        self.logger.propagate = False

    # ------------------------------------------------------------------ #
    # Location detection + caching
    # ------------------------------------------------------------------ #
    def _detect_location(self):
        """Detect user's location using IP geolocation with caching."""
        # Try cache first
        cached_location = self._load_cached_location()
        if cached_location:
            self.detected_location = cached_location
            self.logger.info(f"[LOCATION_CACHED] Using cached location: {self.detected_location}")
            print(f"📍 Using cached location: {self.detected_location}")
            return

        # Try live IP-based detection
        try:
            self.logger.info("[LOCATION_DETECTION] Attempting to detect user location...")

            # ipapi.co is a free IP geolocation endpoint
            response = requests.get("https://ipapi.co/json/", timeout=5)

            if response.status_code == 200:
                data = response.json()
                city = data.get('city', '')
                region = data.get('region', '')
                country = data.get('country_name', '')

                if city and region:
                    self.detected_location = f"{city}, {region}"
                elif city and country:
                    self.detected_location = f"{city}, {country}"
                else:
                    self.logger.warning("[LOCATION_DETECTION] Could not parse location data")

                if self.detected_location:
                    self.logger.info(f"[LOCATION_DETECTED] {self.detected_location}")
                    print(f"📍 Detected location: {self.detected_location}")
                    self._cache_location(self.detected_location)

            else:
                self.logger.warning(
                    f"[LOCATION_DETECTION] API returned status {response.status_code}"
                )

        except Exception as e:
            self.logger.warning(f"[LOCATION_DETECTION] Failed to detect location: {e}")
            print(f"⚠️ Could not detect location: {e}")

        # Fall back if still nothing
        if not self.detected_location:
            self.detected_location = self.default_location
            self.logger.info(f"[LOCATION_FALLBACK] Using default location: {self.default_location}")

    def _load_cached_location(self) -> Optional[str]:
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

    def _cache_location(self, location: str):
        """Cache detected location to file"""
        try:
            with open(self.location_cache_file, 'w') as f:
                f.write(location)
            self.logger.info(f"[LOCATION_CACHE] Cached location: {location}")
        except Exception as e:
            self.logger.warning(f"[LOCATION_CACHE] Failed to cache location: {e}")

    # ------------------------------------------------------------------ #
    # Public handle() - entry point ChatAgent calls
    # ------------------------------------------------------------------ #
    def handle(self, user_input: str, chat_history: Dict[str, Any]) -> Dict[str, str]:
        """
        Handle a user weather question and return:
            {
              "reply":   <string for the user>,
              "summary": <short structured summary for caching>
            }
        """

        self.logger.info(f"[WEATHER_INPUT] {user_input}")

        # If the core weather logic isn't available, bail safely.
        if not self.is_available or self.core_agent is None:
            self.logger.warning("[WEATHER_FALLBACK] Weather agent core not available")
            fallback_text = self._get_fallback_response()
            return {
                "reply": fallback_text,
                "summary": "Weather unavailable."
            }

        # We still want to pass along a best-guess location for this user.
        # The core agent's extract_weather_intent() will try to infer location
        # from text, but if none is found we will inject our detected_location.
        try:
            # Inspect the user's question using extract_weather_intent just like before,
            # but mainly to see if they named a different location.
            intent = extract_weather_intent(user_input)
            # intent: { "location", "time_phrase", "start_hour", "end_hour" }

            # If there's no location in the question, fallback to detected_location
            if not intent.get("location"):
                intent["location"] = self.detected_location

            # Re-construct an augmented user question if needed so the core agent
            # will generate forecast for the right place.
            #
            # Example:
            #   user_input = "Will it rain tonight?"
            #   detected_location = "San Francisco, CA"
            # We'll synthesize: "Will it rain tonight in San Francisco, CA?"
            #
            augmented_user_input = user_input
            if intent.get("location") and intent["location"] not in user_input:
                augmented_user_input = f"{user_input.strip()} in {intent['location']}?"

            self.logger.info(f"[WEATHER_AUGMENT] {augmented_user_input}")

            # Now call the core agent
            core_result = self.core_agent.handle(
                user_input=augmented_user_input,
                chat_history=chat_history
            )
            # core_result should be:
            #   { "reply": "...", "summary": "..." }

            # Log and return
            self.logger.info(f"[WEATHER_RESPONSE] {core_result['reply']}")
            return {
                "reply": core_result.get("reply", "I'm not sure about the weather right now."),
                "summary": core_result.get("summary", "Weather summary unavailable.")
            }

        except Exception as e:
            error_msg = f"Weather agent error: {e}"
            print(error_msg)
            self.logger.error(f"[WEATHER_ERROR] {error_msg}")

            safe_text = self._get_error_response(str(e))
            return {
                "reply": safe_text,
                "summary": "Weather lookup failed."
            }

    # ------------------------------------------------------------------ #
    # Fallback / error helpers
    # ------------------------------------------------------------------ #
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
            "I ran into an issue getting the weather just now. "
            "You could try asking in a slightly different way, "
            "or check your local weather app."
        )
        self.logger.info(f"[WEATHER_ERROR_RESPONSE] {response}")
        return response

    # ------------------------------------------------------------------ #
    # Capability reporting / cache mgmt
    # ------------------------------------------------------------------ #
    def get_capabilities(self) -> Dict[str, Any]:
        """Return information about weather agent capabilities"""
        return {
            "available": self.is_available,
            "features": [
                "Automatic location detection with caching",
                "Natural language time parsing",
                "Location extraction with fallback",
                "Real-time weather data (Open-Meteo)",
                "Senior-friendly clothing advice",
                "Structured summary for follow-up safety/comfort questions"
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

    # ------------------------------------------------------------------ #
    # (Optional) cleanup for location strings before sending to geocoder
    # Keeping for completeness, though the core agent now handles geocoding.
    # ------------------------------------------------------------------ #
    def _clean_location_for_api(self, location: str) -> str:
        """
        Clean location format for Open-Meteo geocoding API.

        Examples:
          "San Francisco, CA" -> "San Francisco"
          "Newark, CA"        -> "Newark"

        We strip trailing comma pieces, which can confuse the geocoder.
        """
        if not location:
            return location

        cleaned = location.split(',')[0].strip()

        # Basic suffix cleanup (extra defensive)
        if cleaned.endswith(', CA'):
            cleaned = cleaned[:-4].strip()
        elif cleaned.endswith(', CA'):
            cleaned = cleaned[:-4].strip()

        return cleaned


# ------------------------------------------------------------------ #
# Manual test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    agent = WeatherAgent()

    test_questions = [
        "What's the weather like today?",
        "Will it rain tomorrow morning?",
        "Do I need an umbrella?",
        "Is it cold outside?",
        "Is it okay to walk after dinner?"
    ]

    print("🧪 Testing Wrapped Weather Agent")
    print("=" * 50)

    for q in test_questions:
        print(f"\nQ: {q}")
        result = agent.handle(q, {})
        print(f"A.reply:   {result['reply']}")
        print(f"A.summary: {result['summary']}")

    print(f"\n📊 Capabilities: {agent.get_capabilities()}")
