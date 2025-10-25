"""
Test Suite for Chat Agent Implementation
Senior Citizens Chat Buddy
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_agent_implementation import (
    IntentClassifier, 
    LLMFallbackAgent, 
    WeatherAgent, 
    ChatAgent,
    IntentResult
)


class TestIntentClassifier(unittest.TestCase):
    """Test cases for IntentClassifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.classifier = IntentClassifier(self.mock_llm)
    
    def test_pattern_match_weather_keywords(self):
        """Test weather intent detection with keywords"""
        test_cases = [
            ("What's the weather like?", "weather"),
            ("It's raining outside", "weather"),
            ("The temperature is hot today", "weather"),
            ("Will it snow tomorrow?", "weather"),
            ("It's sunny and warm", "weather")
        ]
        
        for text, expected_intent in test_cases:
            with self.subTest(text=text):
                intent, confidence = self.classifier._pattern_match(text)
                self.assertEqual(intent, expected_intent)
                self.assertGreater(confidence, 0.5)
    
    def test_pattern_match_weather_regex(self):
        """Test weather intent detection with regex patterns"""
        test_cases = [
            ("Will it rain tomorrow?", "weather"),
            ("Is it going to snow?", "weather"),
            ("How hot is it today?", "weather"),
            ("What's the temperature?", "weather"),
            ("It's raining today", "weather")
        ]
        
        for text, expected_intent in test_cases:
            with self.subTest(text=text):
                intent, confidence = self.classifier._pattern_match(text)
                self.assertEqual(intent, expected_intent)
                self.assertGreaterEqual(confidence, 0.95)  # Regex patterns have high confidence
    
    def test_pattern_match_greeting(self):
        """Test greeting intent detection"""
        test_cases = [
            ("Hello there!", "greeting"),
            ("Hi, how are you?", "greeting"),
            ("Good morning", "greeting"),
            ("Good afternoon", "greeting"),
            ("Hey there", "greeting")
        ]
        
        for text, expected_intent in test_cases:
            with self.subTest(text=text):
                intent, confidence = self.classifier._pattern_match(text)
                self.assertEqual(intent, expected_intent)
                self.assertGreater(confidence, 0.5)
    
    def test_pattern_match_farewell(self):
        """Test farewell intent detection"""
        test_cases = [
            ("Goodbye", "farewell"),
            ("See you later", "farewell"),
            ("Talk to you later", "farewell"),
            ("Bye for now", "farewell")
        ]
        
        for text, expected_intent in test_cases:
            with self.subTest(text=text):
                intent, confidence = self.classifier._pattern_match(text)
                self.assertEqual(intent, expected_intent)
                self.assertGreater(confidence, 0.5)
    
    def test_pattern_match_general(self):
        """Test general intent for non-matching text"""
        test_cases = [
            "I was thinking about my grandchildren",
            "Tell me a story",
            "What's your favorite color?",
            "I'm feeling lonely today"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                intent, confidence = self.classifier._pattern_match(text)
                self.assertEqual(intent, "general")
                self.assertEqual(confidence, 0.3)
    
    def test_classify_with_high_confidence(self):
        """Test classify method with high confidence pattern match"""
        intent, confidence = self.classifier.classify("Hello there!")
        self.assertEqual(intent, "greeting")
        self.assertGreater(confidence, 0.7)
        # Should not call LLM since confidence is high
        self.mock_llm.invoke.assert_not_called()
    
    @patch('chat_agent_implementation.PydanticOutputParser')
    def test_classify_with_low_confidence_uses_llm(self, mock_parser_class):
        """Test classify method falls back to LLM for low confidence"""
        # Mock the parser and result
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.get_format_instructions.return_value = "format instructions"
        
        mock_result = IntentResult(intent="general", confidence=0.8, reasoning="test")
        mock_parser.parse.return_value = mock_result
        
        # Mock the chain invoke
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_result
        
        with patch.object(self.classifier, 'intent_prompt') as mock_prompt:
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            
            intent, confidence = self.classifier.classify("ambiguous text")
            self.assertEqual(intent, "general")
            self.assertEqual(confidence, 0.8)
    
    def test_classify_llm_error_handling(self):
        """Test LLM error handling in classify method"""
        # Mock LLM to raise exception
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        
        intent, confidence = self.classifier.classify("ambiguous text")
        self.assertEqual(intent, "general")
        self.assertEqual(confidence, 0.5)


class TestLLMFallbackAgent(unittest.TestCase):
    """Test cases for LLMFallbackAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = LLMFallbackAgent()
        self.mock_memory = Mock()
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.prompt)
        self.assertEqual(self.agent.llm.temperature, 0.7)
        self.assertEqual(self.agent.llm.max_tokens, 200)
    
    def test_post_process_removes_markdown(self):
        """Test post-processing removes markdown formatting"""
        test_cases = [
            ("**Bold text**", "Bold text"),
            ("*Italic text*", "Italic text"),
            ("**Bold** and *italic*", "Bold and italic"),
            ("Normal text", "Normal text")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.agent._post_process(input_text)
                self.assertEqual(result, expected)
    
    def test_post_process_limits_sentences(self):
        """Test post-processing limits response length"""
        long_response = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."
        result = self.agent._post_process(long_response)
        
        # Should be limited to 4 sentences
        sentences = result.split('. ')
        self.assertLessEqual(len(sentences), 4)
    
    def test_get_fallback_response(self):
        """Test fallback response generation"""
        responses = []
        for _ in range(10):  # Test multiple calls
            response = self.agent._get_fallback_response()
            responses.append(response)
        
        # All responses should be from the predefined list
        expected_responses = [
            "I'm having a bit of trouble right now. Could you tell me that again?",
            "Let me think about that for a moment. Could you rephrase what you said?",
            "I want to make sure I understand you correctly. Could you say that another way?"
        ]
        
        for response in responses:
            self.assertIn(response, expected_responses)
    
    @patch('chat_agent_implementation.ConversationChain')
    def test_handle_success(self, mock_conversation_chain):
        """Test successful response generation"""
        mock_conversation = Mock()
        mock_conversation.predict.return_value = "Hello! How are you today?"
        mock_conversation_chain.return_value = mock_conversation
        
        response = self.agent.handle("Hello", self.mock_memory)
        
        self.assertEqual(response, "Hello! How are you today?")
        mock_conversation_chain.assert_called_once()
        mock_conversation.predict.assert_called_once_with(input="Hello")
    
    @patch('chat_agent_implementation.ConversationChain')
    def test_handle_error_fallback(self, mock_conversation_chain):
        """Test error handling with fallback response"""
        mock_conversation_chain.side_effect = Exception("Chain error")
        
        response = self.agent.handle("Hello", self.mock_memory)
        
        # Should return a fallback response
        expected_responses = [
            "I'm having a bit of trouble right now. Could you tell me that again?",
            "Let me think about that for a moment. Could you rephrase what you said?",
            "I want to make sure I understand you correctly. Could you say that another way?"
        ]
        self.assertIn(response, expected_responses)


class TestWeatherAgent(unittest.TestCase):
    """Test cases for WeatherAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = WeatherAgent()
        self.mock_chat_history = {"messages": []}
    
    def test_handle_returns_mock_response(self):
        """Test weather agent returns mock response"""
        response = self.agent.handle("What's the weather?", self.mock_chat_history)
        
        self.assertEqual(response, "The weather is sunny with a high of 75°F today.")
    
    def test_handle_with_different_inputs(self):
        """Test weather agent with various inputs"""
        test_inputs = [
            "Will it rain?",
            "How hot is it?",
            "What's the temperature?",
            "Is it sunny?"
        ]
        
        for user_input in test_inputs:
            with self.subTest(user_input=user_input):
                response = self.agent.handle(user_input, self.mock_chat_history)
                self.assertEqual(response, "The weather is sunny with a high of 75°F today.")


class TestChatAgent(unittest.TestCase):
    """Test cases for main ChatAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('chat_agent_implementation.ChatAnthropic'):
            self.agent = ChatAgent()
    
    def test_initialization(self):
        """Test chat agent initialization"""
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.intent_classifier)
        self.assertIsNotNone(self.agent.llm_fallback)
        self.assertIn('weather', self.agent.topic_agents)
        self.assertEqual(self.agent.intent_confidence_threshold, 0.6)
    
    @patch.object(IntentClassifier, 'classify')
    def test_process_message_weather_intent(self, mock_classify):
        """Test processing message with weather intent"""
        mock_classify.return_value = ("weather", 0.8)
        
        response = self.agent.process_message("What's the weather?")
        
        self.assertEqual(response, "The weather is sunny with a high of 75°F today.")
        mock_classify.assert_called_once_with("What's the weather?")
    
    @patch.object(IntentClassifier, 'classify')
    @patch.object(LLMFallbackAgent, 'handle')
    def test_process_message_general_intent(self, mock_fallback_handle, mock_classify):
        """Test processing message with general intent"""
        mock_classify.return_value = ("general", 0.8)
        mock_fallback_handle.return_value = "Hello! How are you today?"
        
        response = self.agent.process_message("Hello there!")
        
        self.assertEqual(response, "Hello! How are you today?")
        mock_classify.assert_called_once_with("Hello there!")
        mock_fallback_handle.assert_called_once()
    
    @patch.object(IntentClassifier, 'classify')
    def test_process_message_low_confidence(self, mock_classify):
        """Test processing message with low confidence uses fallback"""
        mock_classify.return_value = ("weather", 0.4)  # Below threshold
        
        with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
            mock_fallback.return_value = "I'm not sure about the weather."
            
            response = self.agent.process_message("Maybe it's raining?")
            
            self.assertEqual(response, "I'm not sure about the weather.")
            mock_fallback.assert_called_once()
    
    def test_memory_save_context(self):
        """Test that conversation is saved to memory"""
        with patch.object(self.agent.memory, 'save_context') as mock_save:
            with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
                mock_classify.return_value = ("general", 0.8)
                
                with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                    mock_fallback.return_value = "Test response"
                    
                    self.agent.process_message("Test input")
                    
                    mock_save.assert_called_once_with(
                        {"input": "Test input"},
                        {"output": "Test response"}
                    )
    
    def test_get_conversation_history(self):
        """Test retrieving conversation history"""
        with patch.object(self.agent.memory, 'load_memory_variables') as mock_load:
            mock_load.return_value = {"chat_history": []}
            
            history = self.agent.get_conversation_history()
            
            self.assertEqual(history, {"chat_history": []})
            mock_load.assert_called_once_with({})


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('chat_agent_implementation.ChatAnthropic'):
            self.agent = ChatAgent()
    
    def test_empty_input(self):
        """Test handling of empty input"""
        with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
            mock_classify.return_value = ("general", 0.5)
            
            with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                mock_fallback.return_value = "I didn't catch that. Could you repeat?"
                
                response = self.agent.process_message("")
                
                self.assertEqual(response, "I didn't catch that. Could you repeat?")
    
    def test_very_long_input(self):
        """Test handling of very long input"""
        long_input = "This is a very long input " * 100
        
        with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
            mock_classify.return_value = ("general", 0.5)
            
            with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                mock_fallback.return_value = "That's quite a lot to say!"
                
                response = self.agent.process_message(long_input)
                
                self.assertEqual(response, "That's quite a lot to say!")
    
    def test_special_characters_input(self):
        """Test handling of special characters"""
        special_inputs = [
            "Hello! @#$%^&*()",
            "What's the weather? 🌤️",
            "I'm feeling 😊 today",
            "Test with émojis and spéciál chârs"
        ]
        
        for special_input in special_inputs:
            with self.subTest(input=special_input):
                with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
                    mock_classify.return_value = ("general", 0.5)
                    
                    with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                        mock_fallback.return_value = "I understand."
                        
                        response = self.agent.process_message(special_input)
                        
                        self.assertEqual(response, "I understand.")
    
    def test_none_input(self):
        """Test handling of None input"""
        with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
            mock_classify.return_value = ("general", 0.5)
            
            with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                mock_fallback.return_value = "I didn't catch that."
                
                response = self.agent.process_message(None)
                
                self.assertEqual(response, "I didn't catch that.")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('chat_agent_implementation.ChatAnthropic'):
            self.agent = ChatAgent()
    
    @patch.object(IntentClassifier, 'classify')
    def test_full_conversation_flow(self, mock_classify):
        """Test complete conversation flow"""
        # Mock different intents for different messages
        mock_classify.side_effect = [
            ("greeting", 0.9),
            ("weather", 0.8),
            ("general", 0.7),
            ("farewell", 0.9)
        ]
        
        with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
            mock_fallback.return_value = "Hello! How are you today?"
            
            # Simulate conversation
            messages = [
                "Hello there!",
                "What's the weather like?",
                "Tell me about your day",
                "Goodbye!"
            ]
            
            responses = []
            for message in messages:
                response = self.agent.process_message(message)
                responses.append(response)
            
            # Verify all messages were processed
            self.assertEqual(len(responses), 4)
            self.assertEqual(mock_classify.call_count, 4)
    
    def test_memory_persistence(self):
        """Test that memory persists across multiple messages"""
        with patch.object(self.agent.intent_classifier, 'classify') as mock_classify:
            mock_classify.return_value = ("general", 0.7)
            
            with patch.object(self.agent.llm_fallback, 'handle') as mock_fallback:
                mock_fallback.return_value = "I remember our conversation."
                
                # Send multiple messages
                self.agent.process_message("Hello")
                self.agent.process_message("How are you?")
                self.agent.process_message("What's your name?")
                
                # Verify memory was called for each message
                self.assertEqual(mock_classify.call_count, 3)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestIntentClassifier,
        TestLLMFallbackAgent,
        TestWeatherAgent,
        TestChatAgent,
        TestEdgeCases,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
