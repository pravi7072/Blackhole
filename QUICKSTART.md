# 🚀 Quick Start Guide - BLACKHOLE AI

Get up and running in 5 minutes!

## Step 1: Install Python Dependencies

```bash
pip install flet>=0.23.0 requests google-generativeai langchain-openai langchain-anthropic python-dotenv pyperclip
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Step 2: Get Your API Keys

### Google Gemini (Required for Fallback)
1. Go to [https://ai.google.dev/](https://ai.google.dev/)
2. Click "Get API Key"
3. Create a new API key
4. Copy the key

### OpenAI (Optional)
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key

### Anthropic Claude (Optional)
1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Navigate to API Keys
3. Create a new key
4. Copy the key

## Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your favorite editor
```

Add your keys:
```env
GOOGLE_API_KEY=your_actual_gemini_key_here
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
```

> **Note**: At minimum, you need the Google Gemini API key for fallback functionality.

## Step 4: Run the Application

```bash
python BlackholeFile.py
```

The application will:
- ✅ Load your API keys
- ✅ Initialize available models
- ✅ Create the database
- ✅ Open the UI window

## Step 5: Start Chatting!

### First Steps:
1. **Create a chat**: Click "New Chat" button
2. **Type a message**: Enter your query in the input field
3. **Send**: Press Enter or click send button
4. **View response**: AI will respond with intelligent routing

### Try These Example Queries:

#### Code Generation:
```
write python code for training a CNN for image classification
```
→ Will search official docs and generate up-to-date code

#### Current Information:
```
what is the latest version of gemini released in 2025
```
→ Will search the web for current information

#### General Questions:
```
explain quantum computing in simple terms
```
→ Will use appropriate model based on complexity

## Keyboard Shortcuts

- `Ctrl+Enter` - New line in message
- `F1` - Create new chat
- `Escape` - Focus input field

## Tips for Best Results

### 1. Use Smart Routing (Default)
- Leave "Smart Routing" toggle ON
- System automatically selects best model

### 2. For Code Queries
- Be specific about language and requirements
- System will search official documentation
- Gets latest syntax and non-deprecated code

### 3. For Manual Control
- Toggle "Smart Routing" OFF
- Select specific model from dropdown
- Useful for testing model differences

### 4. Copy Code Easily
- Click copy icon on any code block
- Get visual confirmation of copy

## Troubleshooting

### "No API models available"
- Check your `.env` file has correct API keys
- Verify keys are not expired
- Restart the application

### Web search not working
- Check internet connection
- DuckDuckGo may have rate limits
- Try again after a few seconds

### Model fails but no response
- Gemini fallback should work automatically
- Check Gemini API key is valid
- Restart application if needed

### Database errors
- Delete `blackhole_conversations.db` file
- Restart application (will recreate database)

## Testing Your Setup

Run self-tests to verify everything works:

```bash
python BlackholeFile.py --selftest
```

This will test:
- ✅ Query classification
- ✅ Documentation search
- ✅ Web search
- ✅ Model routing
- ✅ Complexity analysis

## Next Steps

- ⭐ Star the repo on GitHub
- 📖 Read full [README.md](README.md)
- 🤝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- 💬 Join discussions for questions

## Common Use Cases

### 1. Learning to Code
```
teach me python basics with examples
```

### 2. Debugging Code
```
fix this python error: [paste your error]
```

### 3. Research Assistant
```
explain the latest developments in AI transformers
```

### 4. Writing Help
```
write a professional email to request a meeting
```

### 5. Math & Calculations
```
solve this calculus problem: [your problem]
```

## Performance Tips

- Keep conversations focused (auto-summarizes every 12 turns)
- Use new chat for different topics
- Delete old chats you don't need
- Restart app if it gets slow

## Support

Need help?
- 📧 Check [GitHub Issues](https://github.com/yourusername/blackhole-ai/issues)
- 💬 Join [Discussions](https://github.com/yourusername/blackhole-ai/discussions)
- 📖 Read detailed [README](README.md)

---

**Happy chatting with BLACKHOLE AI! 🕳️✨**
