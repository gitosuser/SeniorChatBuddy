"""
Chat Agent Implementation - Hybrid LangChain Approach
Senior Citizens Chat Buddy
"""

import re
import random
import logging
import os
from datetime import datetime
from typing import Tuple
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# ============================================================================
# Intent Classification
# ============================================================================

class IntentResult(BaseModel):
    """Structured output for intent classification"""
    intent: str = Field(description="The detected intent: 'weather', 'greeting', 'farewell', or 'general'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of classification")


class IntentClassifier:
    """Hybrid intent classifier: patterns first, LLM for ambiguous cases"""
    
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


# ============================================================================
# LLM Fallback Agent
# ============================================================================

class LLMFallbackAgent:
    """General conversational agent using LangChain"""
    
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


# ============================================================================
# Weather Agent Integration
# ============================================================================

# Import the integrated weather agent
from weather_agent_integration import WeatherAgent


# ============================================================================
# Main Chat Agent
# ============================================================================

class ChatAgent:
    """
    Main orchestrator using LangChain for memory and LLM,
    but keeping simple routing logic
    """
    
    def __init__(self):
        # Setup logging
        self._setup_logging()
        # LangChain memory - handles conversation history automatically
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
        
        # LangChain LLM for intent classification
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0
        )
        
        # Intent classifier
        self.intent_classifier = IntentClassifier(llm=self.llm)
        
        # Topic-specific agents
        self.topic_agents = {
            'weather': WeatherAgent(),
            # Add more agents here as you build them
        }
        
        # LLM fallback
        self.llm_fallback = LLMFallbackAgent()
        
        # Configuration
        self.intent_confidence_threshold = 0.6
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"New chat session started: {self.session_id}")
    
    def process_message(self, user_input: str) -> str:
        """Main entry point for processing user messages"""
        
        # Log user input
        self.logger.info(f"[USER] {user_input}")
        
        # 1. Detect intent
        intent, confidence = self.intent_classifier.classify(user_input)
        print(f"[DEBUG] Intent: {intent} (confidence: {confidence:.2f})")
        self.logger.info(f"[INTENT] {intent} (confidence: {confidence:.2f})")
        
        # 2. Route to appropriate handler
        if intent in self.topic_agents and confidence > self.intent_confidence_threshold:
            # Use specific topic agent
            response = self.topic_agents[intent].handle(
                user_input=user_input,
                chat_history=self.memory.load_memory_variables({})
            )
        else:
            # Fallback to general LLM conversation
            response = self.llm_fallback.handle(
                user_input=user_input,
                memory=self.memory
            )
        
        # 3. Save to memory (LangChain handles both user and assistant messages)
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        # Log bot response
        self.logger.info(f"[BOT] {response}")
        
        return response
    
    def get_conversation_history(self):
        """Retrieve conversation history from LangChain memory"""
        return self.memory.load_memory_variables({})
    
    def reset_conversation(self):
        """Clear memory and start a new conversation session"""
        self.memory.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"[RESET] Conversation cleared - new session: {self.session_id}")
        print("[DEBUG] Conversation memory cleared - starting new session")
    
    def _setup_logging(self):
        """Setup file-based logging for conversations"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('chat_buddy')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler with daily rotation
        log_filename = f"logs/chat_sessions_{datetime.now().strftime('%Y%m%d')}.log"
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


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Initialize chat agent
    agent = ChatAgent()
    
    # Simulate conversation
    print("=== Senior Chat Buddy ===\n")
    
    # Example 1: Greeting
    user_input = "Hello there!"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
    
    # Example 2: Ambiguous weather comment
    user_input = "It's quite cold today"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
    
    # Example 3: General conversation
    user_input = "I was thinking about my grandchildren"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
    
    # Example 4: Clear weather intent
    user_input = "Will it rain tomorrow?"
    response = agent.process_message(user_input)
    print(f"User: {user_input}")
    print(f"Bot: {response}\n")
    
    # Show conversation history
    print("=== Conversation History ===")
    history = agent.get_conversation_history()
    for msg in history['chat_history']:
        role = "User" if msg.type == "human" else "Bot"
        print(f"{role}: {msg.content}")
