"""
Chat Agent Implementation - Router + Tool Agents
Senior Citizens Chat Buddy
"""

import re
import random
import logging
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ============================================================================ #
# Intent Classification (unchanged except for comments)
# ============================================================================ #

class IntentResult(BaseModel):
    """Structured output for intent classification"""
    intent: str = Field(description="The detected intent: 'weather', 'greeting', 'farewell', or 'general'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of classification")


class IntentClassifier:
    """
    Hybrid intent classifier: patterns first, LLM for ambiguous cases.
    We will pass its result to the router as a hint — but router is the source of truth.
    """
    INTENT_PATTERNS = {
        'weather': {
            'keywords': ['weather', 'rain', 'snow', 'temperature', 'hot', 'cold', 
                         'sunny', 'cloudy', 'forecast', 'climate', 'warm', 'cool'],
            'patterns': [
                r'\b(will it|is it going to|going to) (rain|snow)\b',
                r'\bhow (hot|cold|warm|cool)\b',
                r'\bwhat\'?s the (weather|temperature|forecast)\b',
                r'\b(raining|snowing|sunny|cloudy) (today|tomorrow|outside)\b'
            ]
        },
        'greeting': {
            'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                         'good evening', 'greetings'],
            'patterns': [
                r'^(hello|hi|hey|greetings)',
                r'good (morning|afternoon|evening|night)'
            ]
        },
        'farewell': {
            'keywords': ['bye', 'goodbye', 'see you', 'talk later', 'farewell'],
            'patterns': [
                r'\b(bye|goodbye|see you|talk to you later)\b'
            ]
        }
    }
    
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=IntentResult)
        
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a senior citizen chat buddy.
            
Available intents:
- weather: Questions or comments about weather, temperature, forecast
- greeting: Hellos, greetings, initial contact
- farewell: Goodbyes, ending conversation
- general: Anything else (casual chat, stories, questions, etc.)

Classify the user's message and provide confidence (0.0-1.0).

{format_instructions}"""),
            ("user", "{user_input}")
        ])
    
    def classify(self, user_input: str) -> Tuple[str, float]:
        """Returns (intent_name, confidence_score)"""
        # Fast pattern matching first
        intent, confidence = self._pattern_match(user_input)
        
        # Use LLM if uncertain
        if confidence < 0.7:
            intent, confidence = self._llm_classify(user_input)
        
        return intent, confidence
    
    def _pattern_match(self, text: str) -> Tuple[str, float]:
        """Fast pattern-based classification"""
        text_lower = text.lower()
        scores = {}
        
        for intent_name, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for kw in patterns['keywords'] if kw in text_lower)
            if keyword_matches > 0:
                score = min(0.3 + (keyword_matches * 0.2), 0.9)
            
            # Check regex patterns (higher confidence)
            for pattern in patterns['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score = max(score, 0.95)
                    break
            
            if score > 0:
                scores[intent_name] = score
        
        if scores:
            best_intent = max(scores, key=scores.get)
            return best_intent, scores[best_intent]
        
        return 'general', 0.3
    
    def _llm_classify(self, text: str) -> Tuple[str, float]:
        """Use LangChain LLM for ambiguous cases"""
        try:
            chain = self.intent_prompt | self.llm | self.parser
            result = chain.invoke({
                "user_input": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result.intent, result.confidence
        except Exception as e:
            print(f"LLM classification error: {e}")
            return 'general', 0.5


# ============================================================================ #
# LLM Fallback Agent (voice of the assistant)
# ============================================================================ #

class LLMFallbackAgent:
    """General conversational agent using LangChain.
       This is the warm, empathetic, senior-friendly assistant voice."""
    
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
- Ask follow-up questions to keep conversation flowing

Important guidelines:
- Never pretend to be human
- Don't provide medical, legal, or financial advice
- If someone seems distressed, show empathy and suggest talking to family/professional
- Be honest if you don't know something
"""

    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=200
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_MESSAGE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
    
    def handle(self, user_input: str, memory: ConversationBufferMemory) -> str:
        """Generate response using LangChain with conversation context"""
        try:
            conversation = ConversationChain(
                llm=self.llm,
                prompt=self.prompt,
                memory=memory,
                verbose=False
            )
            
            response = conversation.predict(input=user_input)
            return self._post_process(response)
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._get_fallback_response()
    
    def _post_process(self, response: str) -> str:
        """Post-process for senior-friendly output"""
        # Remove markdown
        response = response.replace('**', '').replace('*', '')
        
        # Keep concise
        sentences = response.split('. ')
        if len(sentences) > 5:
            response = '. '.join(sentences[:4]) + '.'
        
        return response.strip()
    
    def _get_fallback_response(self) -> str:
        """Safe fallback if LLM fails"""
        responses = [
            "I'm having a bit of trouble right now. Could you tell me that again?",
            "Let me think about that for a moment. Could you rephrase what you said?",
            "I want to make sure I understand you correctly. Could you say that another way?"
        ]
        return random.choice(responses)


# ============================================================================ #
# Weather Agent Integration
# ============================================================================ #

# WeatherAgent is expected to return:
# {
#   "reply":   <string, what we tell the user>,
#   "summary": <string, short structured summary to cache>
# }
from weather_agent_integration import WeatherAgent


# ============================================================================ #
# RouterAgent (NEW)
# ============================================================================ #

class RouterAgent:
    """
    This agent decides the NEXT ACTION the system should take.
    It uses a small local model (Phi-4-mini-instruct or similar) that you run locally.

    Allowed actions:
    - FETCH_WEATHER
    - ADVISE_FROM_WEATHER
    - FETCH_DIRECTORY
    - ADVISE_FROM_DIRECTORY
    - GENERAL_CHAT
    """

    SYSTEM_PROMPT = """You are a routing controller.
You will be given:
1. Recent conversation between the user and assistant.
2. The user's latest message.
3. Cached info summaries from previous tool calls (like weather, directory).
4. An intent hint from a classifier.

You MUST choose EXACTLY ONE of these actions:

FETCH_WEATHER:
- User is asking for a weather forecast, temperature, rain, etc. We need NEW weather.

ADVISE_FROM_WEATHER:
- We ALREADY gave weather.
- User is now asking if it is okay, safe, comfortable, or wise to do something (walk, go out) given that weather.
- Example: "So is it okay to walk?" "Should I still go?"

FETCH_DIRECTORY:
- User is asking to look up / find a phone number, address, business, or service.

ADVISE_FROM_DIRECTORY:
- We ALREADY gave directory/contact info.
- User is now asking if it's safe/legit/what to do next with that info.

GENERAL_CHAT:
- Friendly conversation, memories, feelings, chit-chat, or anything else.

Rules:
- If the user is clearly asking for reassurance or advice about what to do next based on info we already provided, use ADVISE_FROM_* instead of FETCH_*.
- Output ONLY the action token, with no punctuation and no explanation.
"""

    def __init__(self, small_llm):
        """
        small_llm should be a lightweight local inference wrapper around phi-4-mini-instruct.
        It needs an .invoke(prompt:str) -> object with .content or str(result).
        """
        self.llm = small_llm

    def decide(
        self,
        chat_history_messages: List[Any],
        last_user_msg: str,
        last_weather_summary: Optional[str],
        last_directory_summary: Optional[str],
        intent_hint: Optional[str]
    ) -> str:
        """
        Build a compact routing prompt and ask the small router model which action to take.
        """
        # Take only last N turns to keep router prompt tiny
        condensed_history_lines = []
        for m in chat_history_messages[-6:]:
            role = "USER" if getattr(m, "type", "") == "human" else "ASSISTANT"
            condensed_history_lines.append(f"{role}: {m.content}")
        history_text = "\n".join(condensed_history_lines)

        router_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"RECENT CONVERSATION:\n{history_text}\n\n"
            f"USER MESSAGE:\n{last_user_msg}\n\n"
            f"WEATHER SUMMARY CACHE:\n{last_weather_summary or 'NONE'}\n\n"
            f"DIRECTORY SUMMARY CACHE:\n{last_directory_summary or 'NONE'}\n\n"
            f"INTENT HINT:\n{intent_hint or 'NONE'}\n\n"
            f"YOUR ACTION:"
        )

        # Call the small local model
        raw = self.llm.invoke(router_prompt)
        text = getattr(raw, "content", str(raw))
        action = text.strip().split()[0]  # first token should be the action label
        return action


# ============================================================================ #
# ChatAgent Orchestrator (UPDATED TO USE ROUTER)
# ============================================================================ #

class ChatAgent:
    """
    Main orchestrator:
    - keeps conversation memory
    - classifies intent as a hint
    - calls RouterAgent (phi-4-mini-instruct locally) to select NEXT ACTION
    - dispatches to WeatherAgent, advice-from-weather, directory agent, or fallback chat
    """

    def __init__(self, small_router_llm=None, directory_agent=None):
        # ------------------------------------------------------------------ #
        # Setup logging
        self._setup_logging()

        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )

        # LLM for intent classification (Anthropic, high accuracy)
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0
        )

        # Intent classifier
        self.intent_classifier = IntentClassifier(llm=self.llm)

        # Domain agents
        self.weather_agent = WeatherAgent()
        self.directory_agent = directory_agent  # may be None for now

        # Fallback "voice"
        self.llm_fallback = LLMFallbackAgent()

        # Router model (phi-4-mini-instruct or similar)
        # small_router_llm is an injected lightweight model wrapper
        self.router_agent = RouterAgent(small_llm=small_router_llm)

        # Caches for "advise_from_*"
        self.last_weather_summary: Optional[str] = None
        self.last_directory_summary: Optional[str] = None

        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"New chat session started: {self.session_id}")

    # ---------------------------------------------------------------------- #
    # Helper: ask fallback LLM to give advice based on cached weather
    def _advise_from_weather(self, user_input: str) -> str:
        """
        Take cached self.last_weather_summary and user_input like
        'So is it ok to walk?' and ask fallback LLM to produce supportive,
        safety-aware guidance for a senior.
        """
        if not self.last_weather_summary:
            # if we somehow got routed here with no cache, safest fallback:
            return "From what I understand, it should be okay, but I'd like to double-check the weather for you. Would you like me to look it up again?"

        advisory_prompt = (
            "You are helping an older adult decide what to do based on the weather.\n\n"
            "Here is the most recent weather summary you already gave them:\n"
            f"{self.last_weather_summary}\n\n"
            "Here is what they just asked:\n"
            f"{user_input}\n\n"
            "Answer in a warm, simple, reassuring way. Give practical advice "
            "(like 'bring a light jacket' or 'the sidewalk might be a little wet'). "
            "Keep it to 2-4 short sentences. Do NOT just restate the forecast numbers unless needed."
        )

        # We'll reuse llm_fallback.handle() logic, but we want to control prompt.
        # We'll create a tiny fake memory containing recent chat so tone stays consistent.
        fake_memory = _EphemeralMemoryShim(self.memory)

        return self.llm_fallback.handle(
            user_input=advisory_prompt,
            memory=fake_memory
        )

    # You will later do the same pattern for directory:
    def _advise_from_directory(self, user_input: str) -> str:
        if not self.last_directory_summary:
            return "Here's what I suggest: if you're unsure about that phone number, it's okay to wait and call someone you trust first. I can try to look it up if you want."
        advisory_prompt = (
            "You are helping an older adult decide what to do about a phone number or contact.\n\n"
            "Here is the contact / directory info you already found for them:\n"
            f"{self.last_directory_summary}\n\n"
            "Here is what they just asked:\n"
            f"{user_input}\n\n"
            "Give calm, practical guidance in 2-4 short sentences, in plain language. "
            "Focus on safety, scam awareness, and reassurance. Avoid panic."
        )

        fake_memory = _EphemeralMemoryShim(self.memory)
        return self.llm_fallback.handle(
            user_input=advisory_prompt,
            memory=fake_memory
        )

    # ---------------------------------------------------------------------- #
    def process_message(self, user_input: str) -> str:
        """Main entry point for processing a user message end-to-end."""

        # Log user text
        self.logger.info(f"[USER] {user_input}")

        # Grab conversation history so far
        convo_vars = self.memory.load_memory_variables({})
        chat_history_messages = convo_vars["chat_history"]

        # 1. Lightweight intent classification (hint for router)
        intent_hint, confidence = self.intent_classifier.classify(user_input)
        self.logger.info(f"[INTENT] {intent_hint} (confidence: {confidence:.2f})")

        # 2. Ask router which action to take next
        route_action = self.router_agent.decide(
            chat_history_messages=chat_history_messages,
            last_user_msg=user_input,
            last_weather_summary=self.last_weather_summary,
            last_directory_summary=self.last_directory_summary,
            intent_hint=intent_hint
        )
        self.logger.info(f"[ROUTER] {route_action}")

        # 3. Dispatch based on router decision
        if route_action == "FETCH_WEATHER":
            result = self.weather_agent.handle(
                user_input=user_input,
                chat_history=convo_vars
            )
            # result is dict {"reply":..., "summary":...}
            response = result.get("reply", "I'm not sure about the weather right now.")
            self.last_weather_summary = result.get("summary", None)

        elif route_action == "ADVISE_FROM_WEATHER":
            response = self._advise_from_weather(user_input)

        elif route_action == "FETCH_DIRECTORY":
            if self.directory_agent is None:
                response = (
                    "Let me try to help you find that number. "
                    "I'm still learning to look things up, but you can tell me the name of the person or place."
                )
            else:
                d_result = self.directory_agent.handle(
                    user_input=user_input,
                    chat_history=convo_vars
                )
                response = d_result.get("reply", "I'll do my best to help with that contact.")
                self.last_directory_summary = d_result.get("summary", None)

        elif route_action == "ADVISE_FROM_DIRECTORY":
            response = self._advise_from_directory(user_input)

        else:  # GENERAL_CHAT or fallback
            response = self.llm_fallback.handle(
                user_input=user_input,
                memory=self.memory
            )

        # 4. Save turn to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        # 5. Log bot response
        self.logger.info(f"[BOT] {response}")

        return response

    # ---------------------------------------------------------------------- #
    def get_conversation_history(self):
        """Retrieve conversation history from LangChain memory"""
        return self.memory.load_memory_variables({})
    
    def reset_conversation(self):
        """Clear memory and start a new conversation session"""
        self.memory.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[RESET] Conversation cleared - new session: {self.session_id}")
        print("[DEBUG] Conversation memory cleared - starting new session")

    # ---------------------------------------------------------------------- #
    def _setup_logging(self):
        """Setup file-based logging for conversations"""
        os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger('chat_buddy')
        self.logger.setLevel(logging.INFO)

        log_filename = f"logs/chat_sessions_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # avoid duplicate handlers in interactive sessions
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

        self.logger.propagate = False


# ============================================================================ #
# EphemeralMemoryShim (NEW)
# ============================================================================ #

class _EphemeralMemoryShim(ConversationBufferMemory):
    """
    We sometimes want to call llm_fallback with a custom synthesized prompt
    (like the weather advice prompt), but still let it "think" it sees
    recent chat history so tone stays consistent.

    We don't want to permanently mutate main memory for that intermediate prompt.

    This shim wraps the real memory's chat_history for read,
    but discards any save_context writes.
    """
    def __init__(self, real_memory: ConversationBufferMemory):
        # We don't call super().__init__ with full args; we just mimic the API.
        self._real_memory = real_memory

    def load_memory_variables(self, _):
        return self._real_memory.load_memory_variables({})

    def save_context(self, *_args, **_kwargs):
        # swallow writes; do nothing
        return
        

# ============================================================================ #
# Usage Example
# ============================================================================ #

if __name__ == "__main__":

    from phi_router_llm import PhiRouterLLM

    # Instantiate the local Phi router model (downloads GGUF on first run)
    router_llm = PhiRouterLLM(
        # You can override model_path etc. here if you want a specific quant file
        # model_path="/absolute/path/to/your/local/phi4mini-q4.gguf"
    )

    # Optionally stub a directory agent if you have one later.
    class DummyDirectoryAgent:
        def handle(self, user_input: str, chat_history: Dict[str, Any]) -> Dict[str, str]:
            return {
                "reply": "Here's the main office number: (555) 123-4567. Be sure to call during normal hours.",
                "summary": "User asked for office number. Gave (555) 123-4567."
            }

    agent = ChatAgent(
        small_router_llm=router_llm,
        directory_agent=DummyDirectoryAgent()
    )

    print("=== Senior Chat Buddy ===\n")

    # Example 1: Greeting
    user_input = "Hello there!"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")

    # Example 2: Weather request
    user_input = "Can you tell me the weather tonight?"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")

    # Example 3: Follow-up advice about walking
    user_input = "So is it okay if I still go for a walk?"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")

    # Example 4: Directory lookup
    user_input = "Can you find the number for Dr. Patel's office?"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")

    # Show conversation history
    print("=== Conversation History ===")
    history = agent.get_conversation_history()
    for msg in history['chat_history']:
        role = "User" if msg.type == "human" else "Bot"
        print(f"{role}: {msg.content}")
