# Senior Citizens Chat Buddy 🌟

A friendly, accessible chat companion designed specifically for senior citizens, featuring a web-based interface with intelligent conversation routing and senior-friendly design.

## 🎯 Features

### **Core Functionality**
- **Intelligent Intent Classification** - Automatically detects weather, greeting, farewell, and general conversation intents
- **Hybrid AI Approach** - Fast pattern matching with LLM fallback for ambiguous cases
- **Conversation Memory** - Maintains context throughout the conversation
- **Senior-Friendly Design** - Large fonts, high contrast, simple navigation

### **Web Interface**
- **Responsive Design** - Works on desktop, tablet, and mobile devices
- **Accessibility Features** - Screen reader support, keyboard navigation, high contrast mode
- **Real-time Chat** - Instant responses with typing indicators
- **Quick Actions** - Pre-defined conversation starters
- **Error Handling** - Graceful error recovery with user-friendly messages

### **Technical Features**
- **Flask Backend** - RESTful API with proper error handling
- **LangChain Integration** - Advanced conversation management
- **Anthropic Claude** - High-quality AI responses
- **Comprehensive Testing** - Full test suite with mocking
- **Local Development** - Easy setup for development and testing

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- Anthropic API key (for Claude AI)

### **Installation**

1. **Clone or download the project**
   ```bash
   cd /Users/aparnaseetharaman/projects/WIBD/ChatBuddy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
   ```
   
   **Get your Anthropic API key:**
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create an account and get your API key
   - Replace `your_api_key_here` with your actual key

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   - Navigate to: `http://localhost:5000`
   - Start chatting with your Senior Chat Buddy! 👴

## 📁 Project Structure

```
ChatBuddy/
├── app.py                          # Flask web server
├── chat_agent_implementation.py    # Core chat agent logic
├── test_chat_agent.py             # Comprehensive test suite
├── run_tests.py                   # Test runner script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── templates/
│   └── index.html                 # Main HTML template
└── static/
    ├── css/
    │   └── style.css              # Senior-friendly styles
    └── js/
        └── app.js                 # Frontend JavaScript
```

## 🧪 Running Tests

### **Run All Tests**
```bash
python run_tests.py
```

### **Run Specific Test Categories**
```bash
# Test intent classification
python -m unittest test_chat_agent.TestIntentClassifier -v

# Test LLM fallback
python -m unittest test_chat_agent.TestLLMFallbackAgent -v

# Test main chat agent
python -m unittest test_chat_agent.TestChatAgent -v

# Test edge cases
python -m unittest test_chat_agent.TestEdgeCases -v
```

### **Test Coverage**
The test suite covers:
- ✅ Intent classification (pattern matching + LLM fallback)
- ✅ Message routing and processing
- ✅ Error handling and edge cases
- ✅ Memory management
- ✅ API endpoints
- ✅ Integration between components

## 🎨 Customization

### **Adding New Intents**
1. **Update IntentClassifier patterns** in `chat_agent_implementation.py`:
   ```python
   INTENT_PATTERNS = {
       'weather': { ... },
       'greeting': { ... },
       'farewell': { ... },
       'your_new_intent': {
           'keywords': ['keyword1', 'keyword2'],
           'patterns': [r'regex_pattern']
       }
   }
   ```

2. **Create corresponding agent**:
   ```python
   class YourNewAgent:
       def handle(self, user_input: str, chat_history: dict) -> str:
           # Your agent logic here
           return "Response from your agent"
   ```

3. **Register in ChatAgent**:
   ```python
   self.topic_agents = {
       'weather': WeatherAgent(),
       'your_new_intent': YourNewAgent(),
   }
   ```

### **Styling Customization**
- **Colors**: Edit CSS variables in `static/css/style.css`
- **Fonts**: Change font family in CSS
- **Layout**: Modify responsive breakpoints
- **Accessibility**: Adjust contrast ratios and font sizes

### **Backend Customization**
- **API Endpoints**: Add new routes in `app.py`
- **Error Handling**: Customize error responses
- **Authentication**: Add user authentication if needed
- **Database**: Integrate persistent storage

## 🔧 Development

### **Local Development Server**
```bash
# Run with auto-reload
python app.py

# Or use Flask's development server
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

### **Code Quality**
```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Run tests with coverage
pytest --cov=chat_agent_implementation test_chat_agent.py
```

### **Debugging**
- **Backend logs**: Check console output for API errors
- **Frontend logs**: Open browser DevTools (F12) for JavaScript errors
- **Network issues**: Check API endpoints in Network tab
- **AI responses**: Monitor debug output in console

## 🌐 API Endpoints

### **POST /api/chat**
Send a message and get a response.
```json
{
  "message": "Hello, how are you?"
}
```

### **GET /api/history**
Get conversation history.

### **POST /api/clear**
Clear conversation history.

### **GET /api/health**
Check system health and agent availability.

## 🎯 Senior-Friendly Features

### **Design Principles**
- **Large, clear fonts** (16px+ base size)
- **High contrast colors** (dark text on light background)
- **Generous spacing** between elements
- **Large clickable areas** (minimum 44px touch targets)
- **Simple, uncluttered interface**
- **Clear visual hierarchy**

### **Accessibility Features**
- **Screen reader support** with ARIA labels
- **Keyboard navigation** (Tab, Enter, Escape)
- **Focus indicators** for keyboard users
- **High contrast mode** support
- **Reduced motion** support for users with vestibular disorders
- **Voice input** capability (Web Speech API)

### **User Experience**
- **Quick action buttons** for common phrases
- **Character counter** for message length
- **Typing indicators** for better feedback
- **Error recovery** with helpful messages
- **Conversation persistence** across sessions

## 🚀 Deployment

### **Production Considerations**
1. **Environment Variables**: Set `ANTHROPIC_API_KEY` in production
2. **Security**: Change Flask secret key
3. **HTTPS**: Use SSL certificates for production
4. **Database**: Add persistent storage for conversation history
5. **Monitoring**: Add logging and health checks
6. **Scaling**: Use Gunicorn or similar WSGI server

### **Docker Deployment** (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write comprehensive tests
- Update documentation as needed

## 📝 License

This project is designed for educational and personal use. Please ensure you comply with Anthropic's API terms of service when using Claude AI.

## 🆘 Troubleshooting

### **Common Issues**

**"Chat agent not available"**
- Check your Anthropic API key
- Verify internet connection
- Check console for error messages

**"Failed to get response"**
- Check API key validity
- Verify Anthropic account has credits
- Check network connectivity

**Styling issues**
- Clear browser cache
- Check CSS file is loading
- Verify file paths are correct

**Tests failing**
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify API key is set

### **Getting Help**
- Check the console output for error messages
- Review the test suite for expected behavior
- Check Anthropic API documentation
- Ensure all dependencies are properly installed

## 🎉 Success!

You now have a fully functional Senior Citizens Chat Buddy with:
- ✅ Web-based interface
- ✅ Intelligent conversation routing
- ✅ Senior-friendly design
- ✅ Comprehensive testing
- ✅ Local development setup

**Happy chatting!** 👴💬🤖
