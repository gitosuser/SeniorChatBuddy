# Senior Citizens Chat Buddy
**Empathy through AI.**

A modular, senior-friendly conversational assistant designed to support older adults and individuals living alone through friendly dialogue, helpful information, and simple, accessible interaction.

This repository contains:
- The production code for the chatbot (Flask app, RouterLLM, LangChain orchestration, and intent agents)
- The presentation deck describing the problem, architecture, UX, social impact, and roadmap
- (Coming soon) A short demo video

---

## 🌟 Project Goal

To be a helpful assistant and conversation buddy for seniors and people living alone, encouraging:
- Companionship  
- Engagement  
- Mental wellness  

The system is designed to be approachable, empathetic, and easy to use — not “just another chatbot,” but a presence.

---

## 🧠 What the System Does

### 1. Accessible Conversation
The chat interface uses:
- Large, high-contrast fonts
- Clear “Help” and “Clear” buttons
- Starter prompts like “Say Hello,” “Ask about the Weather,” “Start Conversation”
- Simple layout that reduces cognitive load

### 2. Router + Intent Agents
User messages go through:
1. An intent detection module (pattern-based + LLM support)
2. A routing layer (“RouterLLM”) that decides which specialized agent should handle the request
3. A topic-specific agent such as:
   - **Weather agent**
   - **Directory / local info agent**
   - etc.

If an incoming message doesn’t match a known intent, it’s handled by a general LLM fallback to keep the conversation natural.

### 3. Memory & Context
The system uses LangChain to:
- Maintain multi-turn conversation memory
- Preserve context across turns
- Support more natural, less repetitive conversation

---

## 🏗 Repository Layout

```text
SeniorChatBuddy/
├── ChatBuddy/
│   ├── app.py
│   ├── chat_agent_implementation.py
│   ├── phi_router_llm.py / router logic
│   ├── weather_agent_integration.py
│   ├── static/ , templates/
│   └── README.md          <-- How to run the full app locally (Flask, etc.)
│
├── Senior-Companion-Agent/
│   ├── directory_agent.py
│   ├── directory_agent_integration.py
│   ├── restaurant_and_pharmacy_check.py
│   ├── weather_assistant.py
│   └── README.md          <-- Details of each intent agent
│
├── Presentation/
│   ├── Senior_Chat_Buddy_Presentation_Final_v10.pdf
│   └── (Coming soon) demo_video.mp4
│
└── README.md              <-- You are here
```

### Folder roles

- **ChatBuddy/**  
  Core chat experience:
  - Flask web server / API
  - RouterLLM
  - Main ChatAgent
  - Conversation state / memory logic
  - Frontend (HTML/CSS/JS) for senior-friendly UI  
  This folder is where you run the app. See `ChatBuddy/README.md` for setup, environment variables, and local run instructions.

- **Senior-Companion-Agent/**  
  Intent-specific “skills”:
  - Weather agent (forecasts, conditions)
  - Directory / local info agent
  - Other helpers like pharmacy / restaurant lookups  
  Each agent can be registered with the router, and new agents can be added without changing the overall app structure.

- **Presentation/**  
  Material for review and judging:
  - Final PDF presentation deck (system overview, flowcharts, screenshots, roadmap, social impact)
  - A short demo video (to be added)

---

## 🔍 Architecture (High-Level)

1. **Flask App (`app.py`)**  
   Exposes REST endpoints (`/api/chat`, `/api/clear`, `/api/history`, health checks).  
   Also serves the UI (templates + static assets).

2. **RouterLLM**  
   - Classifies the user’s intent.
   - Chooses which agent to call (weather, directory, etc.).
   - Falls back to a general LLM when needed.
   - Persists conversation history using LangChain, so the system can “remember” context.

3. **Intent Agents**  
   Modular helpers for specific domains.  
   Each agent implements a simple interface so it can be plugged in or swapped out:
   - WeatherAgent
   - DirectoryAgent
   - Future: CalendarAgent, MedicationReminderAgent, etc.

4. **LLM Fallback**  
   If we can’t confidently match intent, the message is forwarded to a general-purpose LLM for safe, conversational handling.

---

## 🧑‍🤝‍🧑 Impact

This project is built around senior use cases:
- Reducing isolation through conversation
- Providing simple, immediate utility (weather, directory info)
- Giving control and clarity (“Clear,” “Help,” obvious starter buttons)
- Building toward voice interaction and reminders

---

## 📂 Presentation & Demo

Inside `Presentation/`:
- `Senior_Chat_Buddy_Presentation_Final_v10.pdf`:  
  Pitch deck covering:
  - Problem statement and social impact
  - User interface design for seniors
  - Architecture diagrams (RouterLLM + agents)
  - Accessibility decisions
  - Roadmap and next steps
  - Contribution breakdown

- `demo_video.mp4` (coming soon):  
  A 2-minute walkthrough of:
  - Starting a new chat
  - Asking for help
  - Getting weather
  - Seeing how the system responds in natural language

---

## 👩🏽‍💻 Contributors

- **Aishwarya Ravisankar**  
  - Chose the specific intents and their scope
  - Implemented the intent agents (Weather, Directory, etc.)  
  - Integrated domain-specific logic for helpful answers  
  - Focused on agent behaviors and “what to say back” in each domain

- **Aparna Seetharaman**  
  - Built the RouterLLM and ChatAgent  
  - Integrated LangChain memory and multi-turn intent handling  
  - Built the Flask app and senior-friendly chat interface  
  - Drove overall architecture and orchestration

---

## 🔮 Roadmap

Planned enhancements:
- Voice-enabled front end for natural verbal conversation
- Calendar / schedule support (reminders, appointments)
- Lightweight “How are you doing today?” wellness check-ins
- Usage & feedback dashboard (engagement, sentiment trends) to measure social impact and improve tone

---

## 📜 License / Usage

This work is provided for demonstration, judging, and evaluation.  
Please ensure any use of LLM APIs complies with their respective terms of service.
