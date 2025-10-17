# 🕳️ BLACKHOLE AI

<div align="center">

![BLACKHOLE AI](https://img.shields.io/badge/BLACKHOLE-AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PC9zdmc+)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Flet](https://img.shields.io/badge/Flet-0.23.0%2B-purple?style=flat-square)](https://flet.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square)](https://github.com)

**A powerful multi-model AI chat application with intelligent routing, seamless fallback, and documentation-aware code generation**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#%EF%B8%8F-configuration) • [Screenshots](#-screenshots)

</div>

---

## 🌟 Features

### 🤖 **Multi-Model AI Support**
- **Google Gemini** - Best for coding, math, and analysis
- **OpenAI GPT-4o Mini** - Creative writing and conversation
- **Claude 3.5 Haiku** - Advanced writing and safety

### 🧠 **Intelligent Query Routing**
- Automatic model selection based on query type
- Complexity analysis for optimal performance
- Query classification (coding, writing, math, general)
- Manual model override option

### 📚 **Smart Documentation Search**
- Automatic detection of code-related queries
- Searches official documentation sources
- Prevents deprecated code generation
- Sources: Python.org, OpenAI, Anthropic, Google AI, MDN, and more

### 🔄 **Seamless Fallback System**
- Invisible fallback to Gemini when primary model fails
- User never knows fallback occurred
- Maintains original model attribution in UI
- Handles API errors gracefully

### 🎨 **Professional User Interface**
- Modern dark theme with ChatGPT-style UX
- Copy-to-clipboard for code blocks
- Syntax highlighting for multiple languages
- Proper markdown formatting
- Professional typography and spacing

### 💾 **Advanced Features**
- Persistent chat history with SQLite
- Context-aware conversations
- Chat summarization every 12 turns
- Multiple simultaneous chats
- Chat search and management
- Export functionality

### 🌐 **Web Search Integration**
- Automatic web search for time-sensitive queries
- DuckDuckGo HTML scraping
- Current events and real-time information
- Source citations in responses

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/pravi7072/Blackhole.git
cd Blackhole
```

### Step 2: Install Dependencies

```bash
pip install flet>=0.23.0 requests google-generativeai langchain-openai langchain-anthropic python-dotenv pyperclip
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Get Your API Keys:**
- **Google Gemini**: [https://ai.google.dev/](https://ai.google.dev/)
- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [https://console.anthropic.com/](https://console.anthropic.com/)

---

## 🚀 Usage

### Running the Application

```bash
python BlackholeFile.py
```

### Running Self-Tests

```bash
python BlackholeFile.py --selftest
```

### Basic Operations

#### Creating a New Chat
- Click the **"New Chat"** button in the sidebar
- Or press `F1` keyboard shortcut

#### Sending Messages
- Type your message in the input field
- Use `Ctrl+Enter` for multi-line input

#### Switching Models
- Toggle **"Smart Routing"** off for manual selection
- Choose from available models in dropdown
- Toggle back on for automatic routing

#### Managing Chats
- **Rename**: Click menu (⋮) → Rename
- **Delete**: Click menu (⋮) → Delete
- **Search**: Click on any chat to view history

#### Copying Code
- Click the **copy icon** on any code block
- Receive visual confirmation of successful copy

---

## ⚙️ Configuration

### Model Configuration

Edit `MODEL_CAPABILITIES` in the code to customize model behavior:

```python
MODEL_CAPABILITIES = {
    "google": {
        "name": "Google Gemini",
        "strengths": ["coding", "math", "research"],
        "complexity_score": 9,
        ...
    },
    ...
}
```

### Query Patterns

Customize query classification in `QUERY_PATTERNS`:

```python
QUERY_PATTERNS = {
    "coding": {
        "keywords": ["code", "python", "javascript", ...],
        "patterns": [r"write.*code", ...],
        ...
    },
    ...
}
```

### Database Settings

```python
DATABASE_FILE = "blackhole_conversations_pro.db"
MAX_TURNS_CONTEXT = 12  # Number of previous messages to include
SUMMARY_EVERY_N_TURNS = 12  # Summarize conversation every N turns
```

---

## 📁 Project Structure

```
blackhole-ai/
├── blackhole-final-spacing-fixed.py  # Main application file
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (create this)
├── .env.example                      # Example environment file
├── README.md                         # This file
├── LICENSE                           # MIT License
└── blackhole_conversations.db   # SQLite database (auto-generated)
```

---

## 🎨 Screenshots

### Main Interface
<div align="center">
  <img src="https://github.com/pravi7072/Blackhole/tree/main/screeshots/main-interface.png" alt="Main Interface" width="800"/>
  <p><em>Modern chat interface with intelligent routing indicators</em></p>
</div>

### Code Generation
<div align="center">
  <img src="https://github.com/pravi7072/Blackhole/tree/main/screeshots/code-generation.png" alt="Code Generation" width="800"/>
  <p><em>Professional code blocks with syntax highlighting and copy buttons</em></p>
</div>

### Model Selection
<div align="center">
  <img src="https://github.com/pravi7072/Blackhole/tree/main/screeshots/model-selection.png" alt="Model Selection" width="800"/>
  <p><em>Smart routing with manual override capability</em></p>
</div>

---

## 🔍 How It Works

### Intelligent Routing System

1. **Query Analysis**
   - Classifies query type (coding, writing, math, general)
   - Analyzes query complexity
   - Determines confidence scores

2. **Model Selection**
   - Matches query type with model strengths
   - Considers model availability
   - Applies user preferences

3. **Documentation Search** (For Code Queries)
   - Detects code-related keywords
   - Searches official documentation
   - Provides latest syntax and versions
   - Prevents deprecated code

4. **Response Generation**
   - Builds context with conversation history
   - Integrates search results
   - Generates response with selected model
   - Falls back to Gemini if needed (seamlessly)

### Seamless Fallback Mechanism

```
User Query → Primary Model Selected
                    ↓
            [Try Primary Model]
                    ↓
              Model Fails?
                    ↓
         YES ← [Fallback to Gemini] → NO
          ↓                              ↓
[Use Gemini Response]          [Use Primary Response]
          ↓                              ↓
[Show Original Model Name] ← UI Display → [Show Primary Model Name]
```

**User Experience**: Completely transparent - users never know fallback occurred!

---

## 🛠️ Advanced Features

### Context Management
- Maintains last 12 conversation turns
- Creates rolling summaries every 12 turns
- Injects current date/time automatically
- Preserves conversation continuity

### Memory System
- SQLite-based persistent storage
- Chat-specific memory contexts
- Conversation summaries
- System primers for specialized behavior

### Web Search Integration
- Time-sensitive query detection
- DuckDuckGo HTML scraping
- Current events and real-time data
- Automatic source attribution

### Code Quality Features
- Syntax highlighting for 8+ languages
- Professional code formatting
- Copy-to-clipboard functionality
- Language-specific color coding
- Inline code backtick support

---

## 📊 Performance

- **Average Response Time**: 2-4 seconds
- **Context Window**: Up to 12 previous turns
- **Supported Languages**: Python, JavaScript, Java, HTML, CSS, SQL, Bash, JSON
- **Concurrent Chats**: Unlimited
- **Database Size**: ~50KB per 1000 messages

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints
- Write descriptive commit messages

---

## 🐛 Known Issues

- Web search may be rate-limited by DuckDuckGo
- Large conversation histories may slow down context building
- API rate limits apply to individual model providers

---

## 📝 Roadmap

- [ ] Voice input support
- [ ] Image generation integration
- [ ] Plugin system for extensions
- [ ] Cloud synchronization
- [ ] Mobile app version
- [ ] Multi-language UI support
- [ ] Advanced search and filtering
- [ ] Export to PDF/Markdown
- [ ] Custom model fine-tuning
- [ ] Team collaboration features

---

## 🔒 Security

- API keys stored in environment variables
- Local database with no cloud storage
- No telemetry or data collection
- All conversations stored locally
- Optional encryption for sensitive data

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Flet Framework** - For the excellent Python UI framework
- **Google Gemini** - For powerful AI capabilities
- **OpenAI** - For GPT models and API
- **Anthropic** - For Claude models
- **LangChain** - For AI orchestration tools
- **DuckDuckGo** - For privacy-focused search

---

## 💬 Support
- **Email**: pravinthete8080@gmail.com

---

## 📞 Contact

**Your Name**
- GitHub: [@pravi7072](https://github.com/pravi7072)
- LinkedIn: [Pravin Thete](https://www.linkedin.com/in/pravin-thete-631977315/)

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by Pravin Thete**

[⬆ Back to Top](#-blackhole-ai)

</div>
