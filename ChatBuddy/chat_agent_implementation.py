"""
Chat Agent Implementation - Router + Weather + Memory
Senior Citizens Chat Buddy
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic

from weather_agent_integration import WeatherAgent
from phi_router_llm import PhiRouterLLM


# -----------------------------------------------------------------------------
# Small, friendly general chat helper (Claude)
# -----------------------------------------------------------------------------

class LLMFallbackAgent:
    """
    Handles warm, empathetic small talk and general conversation.
    """

    SYSTEM_MESSAGE = """You are a friendly, patient chat companion for senior citizens.

Your personality:
- Warm, kind, encouraging, and calm
- Never rushed, never judgmental
- You show genuine interest

Your style:
- Use plain, everyday language
- Keep answers short (2 to 4 sentences)
- Be emotionally supportive and reassuring
- Ask gentle follow-up questions

Important:
- Do NOT give medical, legal, or financial advice
- If user seems distressed or unsafe, encourage them to reach out to family or a trusted professional
- Never pretend you are human; you are a virtual companion
"""

    def __init__(self):
        # Anthropic via LangChain
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=200,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

    def handle(self, user_input: str, memory: ConversationBufferMemory) -> str:
        """
        Generate a safe, kind response using the running memory.
        """
        try:
            conversation = ConversationChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=memory,
                verbose=False,
            )

            raw = conversation.predict(input=user_input)
            return self._post_process(raw)

        except Exception as e:
            print(f"[LLM_FALLBACK_ERROR] {e}")
            return self._safe_fallback_line()

    def _post_process(self, text: str) -> str:
        """
        Light cleanup: trim markdown **, keep it short-ish.
        """
        text = text.replace("**", "").replace("*", "")

        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        if len(sentences) > 5:
            text = ". ".join(sentences[:4]) + "."

        return text.strip()

    def _safe_fallback_line(self) -> str:
        options = [
            "I'm thinking about what you said. Could you say that one more time so I understand you clearly?",
            "I want to get this right for you. Would you mind saying it in a slightly different way?",
            "I'm still learning. Could you tell me more about what you mean?"
        ]
        import random
        return random.choice(options)


# -----------------------------------------------------------------------------
# ChatAgent orchestrator
# -----------------------------------------------------------------------------

class ChatAgent:
    """
    Orchestrates:
      • routing (weather / advise-from-weather / directory / chat)
      • calling the WeatherAgent
      • calling the fallback social LLM
      • tracking memory and cached weather summary
    """

    def __init__(self, small_router_llm: PhiRouterLLM, directory_agent: Optional[Any] = None):
        # logging set up first
        self._setup_logging()

        # memory of convo turns
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,
        )

        # subagents
        self.router_llm = small_router_llm         # local Phi mini
        self.fallback_agent = LLMFallbackAgent()   # Claude small talk
        self.weather_agent = WeatherAgent()        # Open-Meteo powered
        self.directory_agent = directory_agent     # currently None, stub below

        # we keep latest weather summary for "is it safe to walk?" followups
        self.last_weather_summary: Optional[str] = None

        # track session id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[SESSION_START] {self.session_id}")

        print("✅ Chat agent initialized successfully")

    # -------------------------------------------------------------------------
    # Public: Flask calls this
    # -------------------------------------------------------------------------

    def process_message(self, user_input: str) -> str:
        """
        Main entry point called by Flask.
        Returns final text we send back to the UI.
        """

        print(f"[USER_INPUT] {user_input}")
        self.logger.info(f"[USER] {user_input}")

        # 1. Quick heuristic override for obvious weather questions.
        #    This protects seniors. We don't *only* trust a tiny LLM router.
        forced_route = self._looks_like_weather_request(user_input)
        if forced_route:
            print(f"[ROUTER_OVERRIDE] Forcing route to {forced_route} based on heuristic match.")
            route_action = forced_route
        else:
            # 2. Otherwise ask router_llm (Phi mini)
            recent_chat_turns = self._summarize_recent_chat_turns()
            print(f"[ROUTER_PROMPT] recent_chat_turns=\n{recent_chat_turns}\n"
                  f"last_weather_summary={self.last_weather_summary}\n"
                  f"user_input={user_input}\n")

            router_result = self.router_llm.invoke(
                user_input=user_input,
                last_weather_summary=self.last_weather_summary,
                recent_chat_turns=recent_chat_turns,
            )

            print(f"[ROUTER_RAW_GENERATION] {router_result.content}")
            route_action = router_result.content.strip().split()[0].upper()

        print(f"[ROUTER_ACTION] {route_action}")
        self.logger.info(f"[ROUTER_ACTION] {route_action}")

        # 3. Branch on what to do next
        if route_action == "FETCH_WEATHER":
            # Call weather agent (live fetch)
            weather_info = self.weather_agent.handle(
                user_input=user_input,
                chat_history=self.memory.load_memory_variables({})
            )
            # weather_info is { "reply": "...", "summary": "..." }

            self.last_weather_summary = weather_info.get("summary", None)
            answer = weather_info.get("reply", "I couldn't get the weather just now.")

            print(f"[WEATHER_REPLY] {answer}")
            print(f"[WEATHER_SUMMARY_CACHED] {self.last_weather_summary}")
            self.logger.info(f"[WEATHER_REPLY] {answer}")
            self.logger.info(f"[WEATHER_SUMMARY] {self.last_weather_summary}")

        elif route_action == "ADVISE_FROM_WEATHER":
            # Use cached summary for safety/comfort advice
            answer = self._answer_using_cached_weather(user_input)

            print(f"[ADVICE_REPLY] {answer}")
            self.logger.info(f"[ADVICE_REPLY] {answer}")

        elif route_action == "FETCH_DIRECTORY":
            # placeholder directory lookup
            answer = self._handle_directory_request(user_input)

            print(f"[DIRECTORY_REPLY] {answer}")
            self.logger.info(f"[DIRECTORY_REPLY] {answer}")

        else:
            # fallback chat (Claude small talk agent)
            answer = self.fallback_agent.handle(
                user_input=user_input,
                memory=self.memory,
            )

            print(f"[CHAT_REPLY] {answer}")
            self.logger.info(f"[CHAT_REPLY] {answer}")

        # 4. Save user + assistant turn into memory
        self.memory.save_context(
            {"input": user_input},
            {"output": answer}
        )

        return answer

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _looks_like_weather_request(self, text: str) -> Optional[str]:
        """
        Heuristic: If user is clearly asking about weather/conditions/what to wear,
        skip the router and go straight to FETCH_WEATHER.

        Also: if user asks "is it okay to walk", and we *already* have a cached
        summary, we can jump straight to ADVISE_FROM_WEATHER.
        """
        lowered = text.lower()

        weather_keywords = [
            "weather",
            "rain",
            "raining",
            "umbrella",
            "forecast",
            "temperature",
            "hot outside",
            "cold outside",
            "how hot",
            "how cold",
            "how warm",
            "how cool",
            "will it be nice",
            "is it nice out",
            "is it nice outside",
        ]

        safety_keywords = [
            "okay to walk",
            "is it safe to walk",
            "should i go for a walk",
            "do i need a jacket",
            "should i wear a jacket",
            "should i bring an umbrella",
        ]

        # First, if it's clearly a fresh forecast ask:
        if any(k in lowered for k in weather_keywords):
            return "FETCH_WEATHER"

        # Second, if it's a safety/comfort follow-up AND we have weather summary:
        if any(k in lowered for k in safety_keywords) and self.last_weather_summary:
            return "ADVISE_FROM_WEATHER"

        # Otherwise, no forced route
        return None

    def _summarize_recent_chat_turns(self, max_turns: int = 6) -> str:
        """
        Build a compact text transcript of recent turns for the router model.
        """
        loaded = self.memory.load_memory_variables({})
        turns: List[Any] = loaded.get("chat_history", [])

        out_lines: List[str] = []
        for msg in turns[-max_turns:]:
            role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
            out_lines.append(f"{role}: {msg.content}")

        return "\n".join(out_lines) if out_lines else "(no previous conversation)"

    def _answer_using_cached_weather(self, user_input: str) -> str:
        """
        Take the cached self.last_weather_summary and turn it into direct advice
        for questions like "Is it okay to walk?" or "Do I need a jacket?".
        We keep this extremely lightweight and reassuring.
        """
        if not self.last_weather_summary:
            return (
                "I'm not completely sure yet, because I haven't checked the weather "
                "for you in this chat. You can ask me, for example: "
                "\"What's the weather like right now?\" and I'll take a look."
            )

        # Gentle advisory tone:
        return (
            f"From what I saw earlier: {self.last_weather_summary} "
            "So if you're going out, dress for that and take it slow. "
            "If you feel unsure about footing or wind, it's always okay "
            "to wait a little or bring someone with you."
        )

    def _handle_directory_request(self, user_input: str) -> str:
        """
        Directory lookups would go here.
        Right now, we haven't wired a working directory agent, so be honest.
        """
        return (
            "I can try to help you find places like pharmacies or clinics, "
            "but I don't have phone book lookups fully set up yet. "
            "You can tell me the name of the place and the city, and I'll do my best to help."
        )

    def get_conversation_history(self):
        """
        Used by /api/history in Flask.
        """
        return self.memory.load_memory_variables({})

    def reset_conversation(self):
        """
        Manual reset in case you add an endpoint later.
        """
        self.memory.clear()
        self.last_weather_summary = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[RESET] New session {self.session_id}")
        print("[DEBUG] Conversation reset.")

    # -------------------------------------------------------------------------
    # Logging setup
    # -------------------------------------------------------------------------

    def _setup_logging(self):
        """
        Create file logger with daily file rotation style filename.
        """
        os.makedirs("logs", exist_ok=True)

        self.logger = logging.getLogger("chat_buddy")
        self.logger.setLevel(logging.INFO)

        log_filename = f"logs/chat_sessions_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        self.logger.propagate = False


# -----------------------------------------------------------------------------
# Manual CLI smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    router = PhiRouterLLM()
    agent = ChatAgent(
        small_router_llm=router,
        directory_agent=None,
    )

    print("=== Senior Chat Buddy CLI test ===\n")

    tests = [
        "Hello there!",
        "Thinking of going for a walk.",
        "Will the weather be nice?",
        "Is it okay to walk outside, or too cold?",
        "Can you give me the phone number for Walgreens in Fremont?",
    ]

    for t in tests:
        print(f"\nUSER: {t}")
        bot = agent.process_message(t)
        print(f"BOT:  {bot}")

    print("\n=== Conversation Memory ===")
    hist = agent.get_conversation_history()
    for msg in hist["chat_history"]:
        role = "User" if getattr(msg, "type", "") == "human" else "Bot"
        print(f"{role}: {msg.content}")

