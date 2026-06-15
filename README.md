# JARVIS AI Assistant

JARVIS is a multi-agent AI system powered by LangGraph and Google's Gemini models. The repository consists of three specialized agents, each capable of fulfilling distinct roles through natural language interactions.

## 🤖 Agents Included

1. **Search Agent** (`/search_agent`)
   - Uses the Tavily API to browse the internet for the most up-to-date and real-time information.
   - Cleans and renders outputs beautifully in the terminal.
2. **Email Agent** (`/email_agent`)
   - Interacts with your Gmail account to read, filter, and reply to emails seamlessly.
3. **Calendar Agent** (`/calendar_agent`)
   - Manages your Google Calendar to view upcoming schedules, create events, and postpone or delete meetings.

## 🚀 Installation & Setup

Follow these steps to set up the entire JARVIS codebase locally.

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd JARVIS
```

### 2. Set up a Python Virtual Environment
It's recommended to isolate your dependencies using a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages for the three agents:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
You must set up your API keys and credentials locally.
1. Create a `.env` file in the root of the `JARVIS` directory.
2. Add your keys:
   ```env
   GEMINI_API_KEY="your-gemini-api-key"
   TAVILY_API_KEY="your-tavily-api-key"
   ```
*(Note: This `.env` file is ignored by Git, ensuring your keys stay private).*

### 5. Setup Google OAuth Credentials (For Email & Calendar)
For the Email and Calendar agents to function, they require Google Workspace access.
1. Create an application in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Gmail API** and **Google Calendar API**.
3. Generate OAuth Client ID credentials and download the `credentials.json` file.
4. Place `credentials.json` in the appropriate agent directories (the first time you run them, they will prompt you to authenticate via your browser and will generate a `token.json` file).

## 💻 Usage

To chat with any of the agents, simply run their respective `chat.py` scripts.

**Run the Search Agent:**
```bash
python search_agent/chat.py
```

**Run the Email Agent:**
```bash
python email_agent/chat.py
```

**Run the Calendar Agent:**
```bash
python calendar_agent/chat.py
```
