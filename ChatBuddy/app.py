"""
Flask Web Server for Senior Citizens Chat Buddy
Local Development Setup
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import os

# Import our chat agent
from chat_agent_implementation import ChatAgent

app = Flask(__name__)
app.secret_key = 'senior-chat-buddy-dev-key'  # Change in production
CORS(app)  # Enable CORS for development

# Global chat agent instance
chat_agent = None

def get_or_create_chat_agent():
    """Get or create chat agent instance"""
    global chat_agent
    if chat_agent is None:
        try:
            chat_agent = ChatAgent()
            print("✅ Chat agent initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing chat agent: {e}")
            chat_agent = None
    return chat_agent

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        # Get chat agent
        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500
        
        # Process message
        response = agent.process_message(user_message)
        
        # Create message objects
        user_msg = {
            'id': str(uuid.uuid4()),
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        }
        
        bot_msg = {
            'id': str(uuid.uuid4()),
            'type': 'bot',
            'content': response,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'user_message': user_msg,
            'bot_message': bot_msg
        })
        
    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500
        
        # Get conversation history from LangChain memory
        history = agent.get_conversation_history()
        
        # Convert LangChain messages to our format
        messages = []
        if 'chat_history' in history:
            for msg in history['chat_history']:
                message = {
                    'id': str(uuid.uuid4()),
                    'type': 'user' if msg.type == 'human' else 'bot',
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()  # LangChain doesn't store timestamps
                }
                messages.append(message)
        
        return jsonify({
            'success': True,
            'messages': messages
        })
        
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500
        
        # Clear LangChain memory
        agent.memory.clear()
        
        return jsonify({
            'success': True,
            'message': 'Conversation history cleared'
        })
        
    except Exception as e:
        print(f"Error clearing conversation history: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    agent = get_or_create_chat_agent()
    return jsonify({
        'success': True,
        'status': 'healthy' if agent is not None else 'unhealthy',
        'agent_available': agent is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Senior Citizens Chat Buddy Web Server")
    print("=" * 50)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Initialize chat agent
    get_or_create_chat_agent()
    
    print("📡 Server will be available at: http://localhost:5001")
    print("🔧 Running in development mode")
    print("=" * 50)
    
    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )
