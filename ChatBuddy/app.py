"""
Flask Web Server for Senior Citizens Chat Buddy
Local Development Setup
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import os
import traceback

# Import our chat agent + tiny router
from chat_agent_implementation import ChatAgent
from phi_router_llm import PhiRouterLLM

app = Flask(__name__)
app.secret_key = 'senior-chat-buddy-dev-key'  # TODO: change for prod
CORS(app)  # Enable CORS for development

# Global chat agent singleton
chat_agent = None


def get_or_create_chat_agent():
    """
    Lazily create (or return cached) ChatAgent.
    This is called by every endpoint so the server can restart routes
    without re-downloading models if the process is still alive.
    """
    global chat_agent
    if chat_agent is None:
        try:
            router_llm = PhiRouterLLM()  # local CPU routing model
            chat_agent = ChatAgent(
                small_router_llm=router_llm,
                directory_agent=None  # placeholder for now
            )
            print("✅ Chat agent initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing chat agent: {e}")
            # also dump traceback for debugging during dev
            traceback.print_exc()
            chat_agent = None
    return chat_agent


@app.route('/')
def index():
    """Serve main chat interface (basic HTML/JS client)."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle one chat turn:
    - read user's message
    - run it through ChatAgent.process_message()
    - return both user + bot bubbles in a consistent shape
    """
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400

        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500

        bot_reply = agent.process_message(user_message)

        user_msg = {
            'id': str(uuid.uuid4()),
            'type': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        }

        bot_msg = {
            'id': str(uuid.uuid4()),
            'type': 'bot',
            'content': bot_reply,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'success': True,
            'user_message': user_msg,
            'bot_message': bot_msg
        }), 200

    except Exception as e:
        print("Error processing chat message:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Return the full conversation so far as the frontend expects it.
    We convert LangChain messages -> simple {id,type,content,timestamp}.
    """
    try:
        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500

        # Pull structured memory from LangChain
        history = agent.get_conversation_history()

        messages = []
        # LangChain stores under key "chat_history": [HumanMessage, AIMessage, ...]
        if 'chat_history' in history:
            for msg in history['chat_history']:
                messages.append({
                    'id': str(uuid.uuid4()),
                    'type': 'user' if getattr(msg, "type", "") == 'human' else 'bot',
                    'content': msg.content,
                    # We don't have per-turn timestamps in LC memory, so use "now"
                    'timestamp': datetime.now().isoformat()
                })

        return jsonify({
            'success': True,
            'messages': messages
        }), 200

    except Exception as e:
        print("Error getting conversation history:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Forget the current conversation (server-side memory reset)."""
    try:
        agent = get_or_create_chat_agent()
        if agent is None:
            return jsonify({
                'success': False,
                'error': 'Chat agent not available'
            }), 500

        agent.memory.clear()
        # also reset weather context so we don't hallucinate from stale summary
        if hasattr(agent, "last_weather_summary"):
            agent.last_weather_summary = None

        return jsonify({
            'success': True,
            'message': 'Conversation history cleared'
        }), 200

    except Exception as e:
        print("Error clearing conversation history:")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Shallow health signal for front-end / status page."""
    agent = get_or_create_chat_agent()
    return jsonify({
        'success': True,
        'status': 'healthy' if agent is not None else 'unhealthy',
        'agent_available': agent is not None,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(_error):
    """Handle 404 so frontend gets JSON instead of HTML."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(_error):
    """Handle 500 so frontend gets JSON instead of HTML."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("🚀 Starting Senior Citizens Chat Buddy Web Server")
    print("==================================================")

    # Make sure basic folders exist (so templates/static don't 404 on first run)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    # Warm up agent (loads phi-3.5-mini-instruct, etc.)
    get_or_create_chat_agent()

    print("📡 Server will be available at: http://localhost:5001")
    print("🔧 Running in development mode")
    print("==================================================")

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )

