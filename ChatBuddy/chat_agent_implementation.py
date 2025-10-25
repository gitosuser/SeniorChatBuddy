"""
Chat Agent Implementation - Router + Weather + Memory
Senior Citizens Chat Buddy
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional, Any, List

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic

from weather_agent_integration import WeatherAgent
from phi_router_llm import PhiRouterLLM


class LLMFallbackAgent:
    SYSTEM_MESSAGE = """You are a friendly, patient chat companion for senior citizens.

Your personality:
- Warm, empathetic, and encouraging
- Patient and clear in communication
- Respectful and never condescending
- Good listener who shows genuine interest

Communication style:
- Use simple, clear language
- Avoid technical jargon
- Keep responses concise (2-4 sentences typically)
- Show empathy and emotional awareness
- Ask gentle follow-up questions to keep conversation going
- Do not sound clinical or like a weather report unless they ask directly

Safety / boundaries:
- Never pretend to be human
- Don't provide medical, legal, or financial advice
- If someone seems distressed, show care and suggest talking to family or a professional
- Be honest if you don't know something

Context you may receive each turn:
- USER_LOCATION: A plain-English location string like "San Francisco, California".
- WEATHER_CONTEXT: A short summary like "San Francisco, afternoon: about 64°F, no rain expected. light jacket ok."

How to use these:
- You MAY mention the location casually, like "around San Francisco" or "in your area".
- You MAY offer practical comfort/safety advice based on the weather, especially if they might go outside (walk, errands, etc.).
- NEVER say you learned this from IP or tracking.
- NEVER guess weather if no WEATHER_CONTEXT was provided.
- If WEATHER_CONTEXT is not provided, don't mention weather unless the user asks.

Be warm, encouraging, and practical.
"""

    def __init__(self):
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


    def handle(
        self,
        user_input: str,
        memory: ConversationBufferMemory,
        user_location: str,
        weather_context: str | None,
    ) -> str:
        """Generate response using LangChain with conversation context
        plus soft context about location & weather.
        """

        # Build contextual preface for this turn
        # We only include WEATHER_CONTEXT block if we actually have one.
        context_lines = []
        if user_location:
            context_lines.append(f"USER_LOCATION: {user_location}")
        if weather_context:
            context_lines.append(f"WEATHER_CONTEXT: {weather_context}")

        if context_lines:
            # Wrap it so Claude clearly knows this is metadata, not the user's words
            contextual_prefix = (
                "Here is helpful context about the user's situation:\n"
                + "\n".join(context_lines)
                + "\n\nNow respond to the user's message below.\n\n"
            )
        else:
            contextual_prefix = ""

        try:
            conversation = ConversationChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=memory,
                verbose=False
            )

            response = conversation.predict(
                input=contextual_prefix + user_input
            )
            return self._post_process(response)

        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._get_fallback_response()




    def _post_process(self, text: str) -> str:
        text = text.replace("**", "").replace("*", "")
        sentences = [s.strip() for s in text.split(". ") if s.strip()]
        if len(sentences) > 5:
            text = ". ".join(sentences[:4]) + "."
        return text.strip()

    def _safe_fallback_line(self) -> str:
        import random
        options = [
            "I'm thinking about what you said. Could you say that one more time so I understand you clearly?",
            "I want to get this right for you. Would you mind saying it in a slightly different way?",
            "I'm still learning. Could you tell me more about what you mean?"
        ]
        return random.choice(options)


class ChatAgent:
    """
    Coordinates:
      - tiny local router (PhiRouterLLM)
      - weather agent (Open-Meteo w/ location + °F)
      - small talk fallback (Claude)
      - memory buffer
    """

    def __init__(self, small_router_llm: PhiRouterLLM, directory_agent: Optional[Any] = None):
        self._setup_logging()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,
        )

        self.router_llm = small_router_llm
        self.fallback_agent = LLMFallbackAgent()

        self.weather_agent = WeatherAgent()
        self.user_location = self.weather_agent.get_user_location()
        self.last_weather_summary: Optional[str] = None

        self.directory_agent = directory_agent


        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[SESSION_START] {self.session_id}")

        print("✅ Chat agent initialized successfully")

    def process_message(self, user_input: str) -> str:
        print(f"[USER_INPUT] {user_input}")
        self.logger.info(f"[USER] {user_input}")

        # 1. Heuristic override for safety: obvious weather -> force FETCH_WEATHER.
        forced_route = self._looks_like_weather_request(user_input)
        if forced_route:
            print(f"[ROUTER_OVERRIDE] Forcing route to {forced_route} based on heuristic match.")
            route_action = forced_route
        else:
            # 2. Ask router model
            recent_chat_turns = self._summarize_recent_chat_turns()
            print(f"[ROUTER_PROMPT] recent_chat_turns=\n{recent_chat_turns}\n"
                  f"last_weather_summary={self.last_weather_summary}\n"
                  f"user_input={user_input}\n")

            router_result = self.router_llm.invoke(
                user_input=user_input,
                last_weather_summary=self.last_weather_summary,
                recent_chat_turns=recent_chat_turns,
            )

            raw_router_text = router_result.content
            print(f"[ROUTER_RAW_GENERATION] {raw_router_text}")

            # normalize the action
            route_action = self._normalize_router_action(raw_router_text)

        print(f"[ROUTER_ACTION] {route_action}")
        self.logger.info(f"[ROUTER_ACTION] {route_action}")

        # 3. Branch
        if route_action == "FETCH_WEATHER":
            weather_info = self.weather_agent.handle(
                user_input=user_input,
                chat_history=self.memory.load_memory_variables({}),
                # we explicitly pass WeatherAgent.detected_location
                fallback_location=self.weather_agent.detected_location,
            )
            self.last_weather_summary = weather_info.get("summary")
            answer = weather_info.get("reply", "I'm sorry, I couldn't get the weather just now.")

            print(f"[WEATHER_REPLY] {answer}")
            print(f"[WEATHER_SUMMARY_CACHED] {self.last_weather_summary}")
            self.logger.info(f"[WEATHER_REPLY] {answer}")
            self.logger.info(f"[WEATHER_SUMMARY] {self.last_weather_summary}")

        elif route_action == "ADVISE_FROM_WEATHER":
            answer = self._answer_using_cached_weather(user_input)
            print(f"[ADVICE_REPLY] {answer}")
            self.logger.info(f"[ADVICE_REPLY] {answer}")

        elif route_action == "FETCH_DIRECTORY":
            answer = self._handle_directory_request(user_input)
            print(f"[DIRECTORY_REPLY] {answer}")
            self.logger.info(f"[DIRECTORY_REPLY] {answer}")

        else:
            answer = self.fallback_agent.handle(
		user_input=user_input,
    		memory=self.memory,
    		user_location=self.user_location,
    		weather_context=self.last_weather_summary,
            )
            print(f"[CHAT_REPLY] {answer}")
            self.logger.info(f"[CHAT_REPLY] {answer}")

        # 4. Save turn to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": answer}
        )

        return answer

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _normalize_router_action(self, raw_text: str) -> str:
        """
        Router sometimes returns stuff like 'Answer:FETCH_WE'.
        We'll coerce that into one of:
          FETCH_WEATHER
          ADVISE_FROM_WEATHER
          FETCH_DIRECTORY
          CHAT
        """
        if not raw_text:
            return "CHAT"

        t = raw_text.strip().upper()

        if "FETCH_WE" in t or "FETCH WEATHER" in t:
            return "FETCH_WEATHER"
        if "ADVISE" in t or "ADVISE_FROM_WEATHER" in t:
            return "ADVISE_FROM_WEATHER"
        if "DIRECTORY" in t:
            return "FETCH_DIRECTORY"
        if "CHAT" in t:
            return "CHAT"

        # default
        return "CHAT"

    def _looks_like_weather_request(self, text: str) -> Optional[str]:
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
            "ok to walk",
            "is it safe to walk",
            "should i go for a walk",
            "do i need a jacket",
            "should i wear a jacket",
            "should i bring an umbrella",
        ]

        if any(k in lowered for k in weather_keywords):
            return "FETCH_WEATHER"

        if any(k in lowered for k in safety_keywords) and self.last_weather_summary:
            return "ADVISE_FROM_WEATHER"

        return None

    def _summarize_recent_chat_turns(self, max_turns: int = 6) -> str:
        loaded = self.memory.load_memory_variables({})
        turns: List[Any] = loaded.get("chat_history", [])
        out_lines = []
        for msg in turns[-max_turns:]:
            role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
            out_lines.append(f"{role}: {msg.content}")
        return "\n".join(out_lines) if out_lines else "(no previous conversation)"

    def _answer_using_cached_weather(self, user_input: str) -> str:
        if not self.last_weather_summary:
            return (
                "I haven't checked the weather for you yet this session. "
                "You can ask me something like, \"What's the weather like right now?\" "
                "and I'll take a look for you."
            )

        return (
            f"From what I saw earlier: {self.last_weather_summary} "
            "So if you're heading out, dress for that and take your time. "
            "If it seems slippery or windy, it's okay to wait or bring someone with you."
        )

    def _handle_directory_request(self, user_input: str) -> str:
        return (
            "I can try to help you look up places like pharmacies or clinics, "
            "but my phone book lookup isn't fully set up yet. "
            "If you tell me the name of the place and the city, I'll do my best."
        )

    def get_conversation_history(self):
        return self.memory.load_memory_variables({})

    def reset_conversation(self):
        self.memory.clear()
        self.last_weather_summary = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[RESET] New session {self.session_id}")
        print("[DEBUG] Conversation reset.")

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger("chat_buddy")
        self.logger.setLevel(logging.INFO)

        log_filename = f"logs/chat_sessions_{datetime.now().strftime('%Y%m%d')}.log"
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


if __name__ == "__main__":
    router = PhiRouterLLM()
    agent = ChatAgent(
        small_router_llm=router,
        directory_agent=None,
    )

    print("=== Senior Chat Buddy CLI test ===\n")
    tests = [
        "Thinking of going for a walk.",
        "Will the weather be nice?",
        "You tell me what the weather is like",
        "So is it ok to walk?",
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

