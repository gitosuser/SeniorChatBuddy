"""
Weather Agent Integration
Bridges partner's weather assistant-style logic into a nice Python class.

Key outputs:
    handle(...) -> {
        "reply":   human-facing answer (what we say),
        "summary": short cached summary we reuse for safety questions
    }
"""

import sys
import os
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import calendar
import dateparser
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------------
# Low-level helpers (mostly adapted from your weather_assistant.py)
# ---------------------------------------------------------------------------------

def _extract_entities(text: str, default_location: str) -> Dict[str, Optional[str]]:
    """
    Pull out location (GPE) and time_phrase (DATE/TIME) using spaCy.
    """
    doc = nlp(text)
    location = None
    time_phrase = None

    for ent in doc.ents:
        if ent.label_ == "GPE" and not location:
            location = ent.text
        elif ent.label_ in ["DATE", "TIME"] and not time_phrase:
            time_phrase = ent.text

    if not location and "weather" in text.lower():
        location = default_location

    return {"location": location, "time_phrase": time_phrase}


def _simplify_time_phrase(phrase: Optional[str]) -> Optional[str]:
    if not phrase:
        return None

    phrase = phrase.lower().strip()

    replacements = {
        "tomorrow morning": "tomorrow at 8 AM",
        "tomorrow evening": "tomorrow at 6 PM",
        "tomorrow night": "tomorrow at 9 PM",
        "tonight": "today at 9 PM",
        "this evening": "today at 6 PM",
        "this morning": "today at 8 AM",
        "next friday at 6 pm": "next friday at 6 PM",
        "next friday": "next friday at noon",
        "this weekend": "saturday at noon",
        "evening": "today at 6 PM",
        "morning": "today at 8 AM",
        "afternoon": "today at 2 PM",
        "night": "today at 9 PM"
    }

    if phrase in replacements:
        return replacements[phrase]

    for k, v in replacements.items():
        if k in phrase:
            return v

    return phrase


def _time_phrase_to_hour_window(full_text: str, default_location: str) -> Tuple[int, int]:
    ents = _extract_entities(full_text, default_location)
    tp = ents["time_phrase"]

    if not tp:
        return (0, 2)  # "next couple hours"

    simplified = _simplify_time_phrase(tp)

    dt = dateparser.parse(
        simplified,
        settings={
            "RELATIVE_BASE": datetime.now(),
            "PREFER_DATES_FROM": "future",
        },
    )

    if not dt:
        from dateparser.search import search_dates
        results = search_dates(
            simplified,
            settings={"RELATIVE_BASE": datetime.now()},
        )
        if results:
            dt = results[0][1]
        else:
            return (0, 2)

    # handle "next Friday"
    if "next" in simplified:
        weekday_target = next(
            (i for i, day in enumerate(calendar.day_name)
             if day.lower() in simplified.lower()),
            None
        )
        if weekday_target is not None:
            today = datetime.now()
            days_ahead = (weekday_target - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            dt = today.replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)

    delta = dt - datetime.now()
    start_hour = int(delta.total_seconds() // 3600)
    end_hour = start_hour + 2
    return (max(0, start_hour), max(0, end_hour))


def _extract_weather_intent(text: str, default_location: str) -> Dict[str, Any]:
    ents = _extract_entities(text, default_location)
    location = ents["location"] or default_location
    start_hour, end_hour = _time_phrase_to_hour_window(text, default_location)
    return {
        "location": location,
        "time_phrase": ents["time_phrase"],
        "start_hour": start_hour,
        "end_hour": end_hour,
    }


def _fetch_open_meteo_block(location: str, start_hour: int, end_hour: int) -> Dict[str, Any]:
    """
    Query Open-Meteo for hourly temps + precip for [start_hour, end_hour),
    return averages and resolved location.
    """
    # First geocode
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    geo_resp = requests.get(geo_url).json()

    if "results" not in geo_resp or not geo_resp["results"]:
        return {"ok": False, "error": f"❌ Could not find location: {location}"}

    lat = geo_resp["results"][0]["latitude"]
    lon = geo_resp["results"][0]["longitude"]
    resolved_name = geo_resp["results"][0].get("name", location)

    now = datetime.utcnow()
    start_time = now + timedelta(hours=start_hour)
    end_time = now + timedelta(hours=end_hour)

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation"
        f"&start={start_time.strftime('%Y-%m-%dT%H:00')}"
        f"&end={end_time.strftime('%Y-%m-%dT%H:00')}"
        f"&timezone=auto"
    )

    forecast_resp = requests.get(forecast_url).json()

    if "hourly" not in forecast_resp:
        return {"ok": False, "error": "❌ Could not fetch forecast data."}

    temps_c = forecast_resp["hourly"]["temperature_2m"]
    prec_mm = forecast_resp["hourly"]["precipitation"]

    if not temps_c or not prec_mm:
        return {"ok": False, "error": "❌ Forecast data incomplete."}

    avg_temp_c = sum(temps_c) / len(temps_c)
    avg_prec_mm = sum(prec_mm) / len(prec_mm)

    return {
        "ok": True,
        "avg_temp_c": avg_temp_c,
        "avg_prec_mm": avg_prec_mm,
        "location": resolved_name,
    }


def _c_to_f(c: float) -> float:
    # Fahrenheit = (Celsius * 9/5) + 32
    return (c * 9.0 / 5.0) + 32.0


def _time_of_day_from_window(start_hour: int, end_hour: int) -> str:
    mid = (start_hour + end_hour) // 2
    if 5 <= mid < 11:
        return "morning"
    elif 11 <= mid < 17:
        return "afternoon"
    elif 17 <= mid < 21:
        return "evening"
    else:
        return "night"


def _advice_from_conditions_f(avg_temp_f: float, avg_prec_mm: float) -> str:
    """
    Give simple clothing / rain guidance in °F for seniors.
    """
    bits = []

    # temperature ranges in °F
    if avg_temp_f < 55:
        bits.append("Wear something warm, like a sweater or coat.")
    elif avg_temp_f < 65:
        bits.append("A light jacket should be fine.")
    else:
        bits.append("You should be comfortable in regular clothes.")

    # rain
    if avg_prec_mm > 0.2:
        bits.append("Don't forget an umbrella — there's a good chance of rain.")
    elif avg_prec_mm > 0:
        bits.append("There might be a light drizzle, so keep an umbrella handy.")

    return " ".join(bits)


def _build_user_reply_f(location: str, time_of_day: str, avg_temp_f: float, avg_prec_mm: float) -> str:
    rain_phrase = "some rain" if avg_prec_mm > 0 else "dry skies"
    advisory = _advice_from_conditions_f(avg_temp_f, avg_prec_mm)

    return (
        f"In {location} during the {time_of_day}, "
        f"it'll be around {round(avg_temp_f)}°F with {rain_phrase}.\n"
        f"{advisory}"
    )


def _build_summary_f(location: str, time_of_day: str, avg_temp_f: float, avg_prec_mm: float) -> str:
    rain_desc = "light rain/drizzle likely" if avg_prec_mm > 0 else "no rain expected"

    if avg_temp_f < 55:
        clothing_hint = "bundle up, it's chilly"
    elif avg_temp_f < 65:
        clothing_hint = "light jacket ok"
    else:
        clothing_hint = "regular clothes fine"

    return (
        f"{location}, {time_of_day}: about {round(avg_temp_f)}°F, "
        f"{rain_desc}. {clothing_hint}."
    )


# ---------------------------------------------------------------------------------
# WeatherAgent wrapper class that ChatAgent talks to
# ---------------------------------------------------------------------------------

class WeatherAgent:
    """
    Public contract:
        handle(user_input, chat_history, fallback_location=None) -> {
            "reply": "...",
            "summary": "..."
        }

    Internally:
    - Figures out location/time window
    - Calls Open-Meteo
    - Returns a human-friendly response in °F
    - Returns a compact summary for reuse
    """

    def __init__(self):
        # default if location detection fails
        self.default_location = "San Francisco, CA"

        # IP-based autodetect, cached:
        self.detected_location = None
        self.location_cache_file = "logs/detected_location.txt"

        self._setup_logging()
        self._detect_location()

        print("✅ Weather agent integrated successfully")
        self.logger.info("Weather agent initialized successfully")

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger("weather_agent")
        self.logger.setLevel(logging.INFO)

        log_filename = f"logs/weather_agent_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_filename, encoding="utf-8")
        fh.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)

        if not self.logger.handlers:
            self.logger.addHandler(fh)

        self.logger.propagate = False

    def _detect_location(self):
        """
        Try to fill self.detected_location using cached file or ipapi.co.
        """
        cached = self._load_cached_location()
        if cached:
            self.detected_location = cached
            print(f"📍 Using cached location: {self.detected_location}")
            self.logger.info(f"[LOCATION] cached={self.detected_location}")
            return

        try:
            resp = requests.get("https://ipapi.co/json/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                city = data.get("city", "")
                region = data.get("region", "")
                country = data.get("country_name", "")

                if city and region:
                    self.detected_location = f"{city}, {region}"
                elif city and country:
                    self.detected_location = f"{city}, {country}"

                if self.detected_location:
                    print(f"📍 Detected location: {self.detected_location}")
                    self.logger.info(f"[LOCATION] detected={self.detected_location}")
                    self._cache_location(self.detected_location)
        except Exception as e:
            print(f"⚠️ Could not detect location: {e}")
            self.logger.warning(f"[LOCATION_ERROR] {e}")

        if not self.detected_location:
            self.detected_location = self.default_location
            self.logger.info(f"[LOCATION_FALLBACK] {self.detected_location}")

    def _load_cached_location(self) -> Optional[str]:
        try:
            if os.path.exists(self.location_cache_file):
                with open(self.location_cache_file, "r") as f:
                    s = f.read().strip()
                    if s:
                        return s
        except Exception as e:
            self.logger.warning(f"[CACHE_READ_ERROR] {e}")
        return None

    def _cache_location(self, location: str):
        try:
            with open(self.location_cache_file, "w") as f:
                f.write(location)
            self.logger.info(f"[CACHE_WRITE] {location}")
        except Exception as e:
            self.logger.warning(f"[CACHE_WRITE_ERROR] {e}")

    def _clean_location_for_api(self, location: str) -> str:
        """
        Strip state abbreviations etc. "San Francisco, CA" -> "San Francisco"
        """
        if not location:
            return location
        cleaned = location.split(",")[0].strip()
        return cleaned

    def handle(
        self,
        user_input: str,
        chat_history: Dict[str, Any],
        fallback_location: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Main call used by ChatAgent.
        """
        # Decide which location to assume if user didn't give one:
        base_location = (
            fallback_location
            or self.detected_location
            or self.default_location
        )

        # Extract intent (location + time window)
        wi = _extract_weather_intent(user_input, base_location)
        loc_for_api = wi["location"] or base_location
        loc_for_api = self._clean_location_for_api(loc_for_api)

        start_hour = wi["start_hour"]
        end_hour = wi["end_hour"]

        block = _fetch_open_meteo_block(loc_for_api, start_hour, end_hour)
        if not block["ok"]:
            err = block["error"]
            self.logger.warning(f"[WEATHER_FAIL] {err}")
            return {
                "reply": err,
                "summary": f"Weather unknown for {loc_for_api} right now."
            }

        avg_temp_c = block["avg_temp_c"]
        avg_temp_f = _c_to_f(avg_temp_c)
        avg_prec_mm = block["avg_prec_mm"]
        resolved_name = block["location"]

        tod = _time_of_day_from_window(start_hour, end_hour)

        reply_text = _build_user_reply_f(
            resolved_name,
            tod,
            avg_temp_f,
            avg_prec_mm
        )

        summary_text = _build_summary_f(
            resolved_name,
            tod,
            avg_temp_f,
            avg_prec_mm
        )

        return {
            "reply": reply_text,
            "summary": summary_text,
        }

    # capability reporting if you want it later
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "detected_location": self.detected_location,
            "data_source": "Open-Meteo hourly forecast",
            "advice_units": "°F and plain-language rain/clothing advice"
        }


    def get_user_location(self) -> str:
        """
        Return the best-known user location string for other agents (directory, chat, etc.)
        """
        return (
        	self.detected_location
        	or self.default_location
        	or "your area"
    	)


