# filename: blackhole-final-spacing-fixed.py
# ENHANCED VERSION WITH SEAMLESS FALLBACK + PROPER VERTICAL SPACING:
# 1. Smart documentation-aware code generation (checks latest official docs)
# 2. SEAMLESS fallback system (automatically falls back to Gemini but shows original model in UI)
# 3. Professional UI with copy buttons and proper formatting
# 4. NO visible fallback indicators - user never knows fallback occurred
# 5. FIXED vertical spacing issue - proper separation between title and date/status

"""
HOW TO RUN:
pip install flet>=0.23.0 requests google-generativeai langchain-openai langchain-anthropic python-dotenv pyperclip
python blackhole-final-spacing-fixed.py

For self-tests:
python blackhole-final-spacing-fixed.py --selftest

Set environment variables:
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key  
ANTHROPIC_API_KEY=your_claude_key
"""

import os
import sys
import time
import json
import re
import sqlite3
import threading
import uuid
import pyperclip  # For clipboard functionality
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from html import unescape
import flet as ft
import requests

# Load env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# API keys (env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Config
DATABASE_FILE = "blackhole_conversations_pro.db"
MAX_TURNS_CONTEXT = 12
SUMMARY_EVERY_N_TURNS = 12
TITLE_FROM_FIRST_USER_MSG = 8

# Current date for context injection
CURRENT_DATE = datetime.now().strftime("%A, %B %d, %Y")
CURRENT_TIME = datetime.now().strftime("%I:%M %p IST")

print(f"📅 Current Date: {CURRENT_DATE} at {CURRENT_TIME}")

# Google AI availability
GOOGLE_AI_AVAILABLE = False
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
    print("✅ Google AI client loaded")
except Exception:
    print("⚠️ google.generativeai not available (optional)")

# LangChain availability
LANGCHAIN_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain connectors loaded")
except Exception:
    print("⚠️ LangChain connectors not available (optional)")

# Model capabilities
MODEL_CAPABILITIES = {
    "google": {
        "name": "Google Gemini",
        "short_name": "Gemini",
        "icon": "🌟",
        "color": ft.colors.ORANGE_400,
        "description": "Best for coding, math, analysis",
        "strengths": ["coding", "math", "research", "analysis", "reasoning"],
        "complexity_score": 9,
        "cost_efficiency": 10,
        "speed": 9,
        "context_length": 10,
        "available": bool(GOOGLE_API_KEY and GOOGLE_AI_AVAILABLE)
    },
    "openai": {
        "name": "OpenAI GPT-4o Mini",
        "short_name": "GPT-4o Mini",
        "icon": "🧠",
        "color": ft.colors.GREEN_400,
        "description": "Creative writing, conversation",
        "strengths": ["writing", "creative", "general", "conversation"],
        "complexity_score": 8,
        "cost_efficiency": 7,
        "speed": 8,
        "context_length": 8,
        "available": bool(OPENAI_API_KEY and LANGCHAIN_AVAILABLE)
    },
    "claude": {
        "name": "Claude 3.5 Haiku",
        "short_name": "Claude",
        "icon": "💭",
        "color": ft.colors.PURPLE_400,
        "description": "Writing, analysis, safety",
        "strengths": ["writing", "analysis", "safety", "nuanced_reasoning"],
        "complexity_score": 9,
        "cost_efficiency": 8,
        "speed": 9,
        "context_length": 9,
        "available": bool(ANTHROPIC_API_KEY and LANGCHAIN_AVAILABLE)
    }
}

# Enhanced query patterns for better code detection
QUERY_PATTERNS = {
    "coding": {
        "keywords": ["code", "python", "javascript", "html", "css", "programming", "debug", "function",
                    "class", "api", "algorithm", "syntax", "compile", "execute", "script", "library", 
                    "hello world", "program", "java", "cpp", "typescript", "node", "react", "model names"],
        "patterns": [r"write.*code", r"write.*program", r"make.*program", r"create.*program",
                    r"fix.*bug", r"implement", r"def\s+\w+", r"function\s+\w+",
                    r"<\w+>", r"import\s+\w+", r"#include", r"hello world", r"model.*name"],
        "preferred_models": ["google", "openai", "claude"]
    },
    "math": {
        "keywords": ["calculate", "equation", "solve", "math", "algebra", "calculus"],
        "patterns": [r"\d+\s*[\+\-\*\/]\s*\d+", r"x\s*=", r"f\(x\)"],
        "preferred_models": ["google", "claude", "openai"]
    },
    "writing": {
        "keywords": ["write", "essay", "article", "story", "creative", "content"],
        "patterns": [r"write.*about", r"essay.*on", r"story.*about"],
        "preferred_models": ["claude", "openai", "google"]
    },
    "general": {
        "keywords": ["what", "how", "why", "when", "where", "explain", "tell", "help"],
        "patterns": [r"what.*is", r"how.*do", r"why.*does"],
        "preferred_models": ["google", "openai", "claude"]
    }
}

@dataclass
class Chat:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    is_archived: bool = False

@dataclass
class Message:
    id: str
    chat_id: str
    role: str
    content: str
    timestamp: datetime
    model_used: Optional[str] = None
    query_type: Optional[str] = None
    complexity: Optional[str] = None
    processing_time: Optional[float] = None
    routing_info: Optional[Dict] = None

@dataclass
class Memory:
    id: str
    chat_id: str
    type: str
    payload: Dict
    updated_at: datetime

class DatabaseManager:
    def __init__(self, db_file: str = DATABASE_FILE):
        self.db_file = db_file
        self._lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                conn.execute("PRAGMA foreign_keys=ON")
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS chats (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_archived BOOLEAN DEFAULT 0
                    )""")
                c.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_used TEXT,
                        query_type TEXT,
                        complexity TEXT,
                        processing_time REAL,
                        routing_info TEXT,
                        FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                    )""")
                c.execute("""
                    CREATE TABLE IF NOT EXISTS memory (
                        id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                    )""")
                conn.commit()
                conn.close()
                print("✅ Database initialized")
        except Exception as e:
            print("❌ DB init failed:", e)

    def create_chat(self, title: str = "New Chat") -> Optional[Chat]:
        chat_id = str(uuid.uuid4())
        now = datetime.now()
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                         (chat_id, title, now.isoformat(), now.isoformat()))
                conn.commit()
                conn.close()
                return Chat(chat_id, title, now, now)
        except Exception as e:
            print("❌ create_chat:", e)
            return None

    def get_all_chats(self) -> List[Chat]:
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("SELECT id, title, created_at, updated_at, is_archived FROM chats WHERE is_archived=0 ORDER BY updated_at DESC")
                rows = c.fetchall()
                conn.close()
                chats = []
                for r in rows:
                    created = datetime.fromisoformat(r[2]) if isinstance(r[2], str) else r[2]
                    updated = datetime.fromisoformat(r[3]) if isinstance(r[3], str) else r[3]
                    chats.append(Chat(id=r[0], title=r[1], created_at=created, updated_at=updated, is_archived=bool(r[4])))
                return chats
        except Exception as e:
            print("❌ get_all_chats:", e)
            return []

    def update_chat_title(self, chat_id: str, title: str):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("UPDATE chats SET title=?, updated_at=? WHERE id=?", (title, datetime.now().isoformat(), chat_id))
                conn.commit()
                conn.close()
        except Exception as e:
            print("❌ update_chat_title:", e)

    def delete_chat(self, chat_id: str):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("DELETE FROM chats WHERE id=?", (chat_id,))
                conn.commit()
                conn.close()
        except Exception as e:
            print("❌ delete_chat:", e)

    def save_message(self, message: Message):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("""INSERT INTO messages
                           (id, chat_id, role, content, timestamp, model_used, query_type, complexity, processing_time, routing_info)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (message.id, message.chat_id, message.role, message.content[:10000], message.timestamp.isoformat(),
                          message.model_used, message.query_type, message.complexity, message.processing_time,
                          json.dumps(message.routing_info) if message.routing_info else None))
                c.execute("UPDATE chats SET updated_at=? WHERE id=?", (datetime.now().isoformat(), message.chat_id))
                conn.commit()
                conn.close()
        except Exception as e:
            print("❌ save_message:", e)

    def get_chat_messages(self, chat_id: str, limit: int = 100) -> List[Message]:
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("SELECT id, chat_id, role, content, timestamp, model_used, query_type, complexity, processing_time, routing_info FROM messages WHERE chat_id=? ORDER BY timestamp ASC LIMIT ?",
                         (chat_id, limit))
                rows = c.fetchall()
                conn.close()
                messages = []
                for r in rows:
                    ts = datetime.fromisoformat(r[4]) if isinstance(r[4], str) else r[4]
                    routing = None
                    if r[9]:
                        try:
                            routing = json.loads(r[9])
                        except:
                            routing = None
                    messages.append(Message(id=r[0], chat_id=r[1], role=r[2], content=r[3], timestamp=ts,
                                          model_used=r[5], query_type=r[6], complexity=r[7], processing_time=r[8],
                                          routing_info=routing))
                return messages
        except Exception as e:
            print("❌ get_chat_messages:", e)
            return []

    def save_memory(self, memory: Memory):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("INSERT OR REPLACE INTO memory (id, chat_id, type, payload_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                         (memory.id, memory.chat_id, memory.type, json.dumps(memory.payload), memory.updated_at.isoformat()))
                conn.commit()
                conn.close()
        except Exception as e:
            print("❌ save_memory:", e)

    def get_memory(self, chat_id: str, memory_type: str) -> Optional[Memory]:
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_file, timeout=10)
                c = conn.cursor()
                c.execute("SELECT id, chat_id, type, payload_json, updated_at FROM memory WHERE chat_id=? AND type=?", (chat_id, memory_type))
                row = c.fetchone()
                conn.close()
                if row:
                    updated = datetime.fromisoformat(row[4]) if isinstance(row[4], str) else row[4]
                    return Memory(id=row[0], chat_id=row[1], type=row[2], payload=json.loads(row[3]), updated_at=updated)
        except Exception as e:
            print("❌ get_memory:", e)
        return None

class ContextManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def build_context_for_chat(self, chat_id: str, current_message: str, web_results: List[Dict] = None, doc_results: List[Dict] = None) -> str:
        parts = []
        
        # Always inject current date/time at the start
        parts.append(f"Current Date & Time: {CURRENT_DATE} at {CURRENT_TIME}")
        
        # Add documentation results if available (prioritized over web search)
        if doc_results:
            parts.append("\n📚 Latest Official Documentation:")
            for i, doc in enumerate(doc_results):
                parts.append(f"[DOC-{i+1}] {doc['title']} - {doc['content']} (Source: {doc['url']})")
            parts.append("Use the above official documentation to provide up-to-date, non-deprecated code examples.\n")
        
        # Add web search results if available
        elif web_results:
            parts.append("\n🔍 Fresh Web Search Results:")
            for i, w in enumerate(web_results):
                parts.append(f"[{i+1}] {w['title']} - {w['snippet']} (Source: {w['url']})")
            parts.append("Use the above current web search results to answer the question with up-to-date information.\n")
        
        primer = self.db.get_memory(chat_id, "primer")
        if primer:
            parts.append(f"System: {primer.payload.get('content','')}")
        summary = self.db.get_memory(chat_id, "summary")
        if summary:
            parts.append(f"Previous conversation summary: {summary.payload.get('content','')}")
        
        msgs = self.db.get_chat_messages(chat_id)
        recent = msgs[-MAX_TURNS_CONTEXT:] if msgs else []
        for m in recent:
            role = "Human" if m.role == "user" else "Assistant"
            parts.append(f"{role}: {m.content}")
        
        parts.append(f"Human: {current_message}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def update_memory_after_message(self, chat_id: str):
        messages = self.db.get_chat_messages(chat_id)
        if len(messages) >= SUMMARY_EVERY_N_TURNS and len(messages) % SUMMARY_EVERY_N_TURNS == 0:
            self._update_rolling_summary(chat_id, messages)

    def _update_rolling_summary(self, chat_id: str, messages: List[Message]):
        try:
            recent = messages[-SUMMARY_EVERY_N_TURNS:]
            conv_text = "\n".join([f"{'Human' if m.role=='user' else 'AI'}: {m.content[:200]}" for m in recent])
            content = f"Summary of last {len(recent)} turns: {conv_text[:500]}..."
            memory = Memory(id=f"{chat_id}_summary_{int(time.time())}", chat_id=chat_id, type="summary",
                          payload={"content": content, "message_count": len(recent)}, updated_at=datetime.now())
            self.db.save_memory(memory)
        except Exception as e:
            print("❌ update summary:", e)

class IntelligentRouter:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.per_chat_performance = {}

    def analyze_query_complexity(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        score = 0
        wc = len(query.split())
        if wc > 50:
            score += 3
        elif wc > 20:
            score += 2
        elif wc > 10:
            score += 1
        if score >= 8:
            return "complex", min(score / 10, 1.0)
        elif score >= 4:
            return "medium", score / 10
        else:
            return "simple", score / 10

    def classify_query_type(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        
        # Enhanced special-case heuristics for coding detection
        if re.search(r"\bhello world\b", q):
            return "coding", 0.98
        if re.search(r"\bwrite\s+(a|an|the)?\s*(code|program)\b", q):
            return "coding", 0.95
        if re.search(r"\bmake\s+(a|an|the)?\s*program\b", q):
            return "coding", 0.93
        if re.search(r"\bcreate\s+(a|an|the)?\s*(code|program)\b", q):
            return "coding", 0.93
        if re.search(r"\bmodel\s+names?\b", q):
            return "coding", 0.90
            
        best_type = "general"
        best_conf = 0.0
        
        for t, cfg in QUERY_PATTERNS.items():
            conf = 0.0
            kw_matches = sum(1 for kw in cfg["keywords"] if kw in q)
            if kw_matches > 0:
                conf += (kw_matches / len(cfg["keywords"])) * 0.7
            pat_matches = sum(1 for p in cfg["patterns"] if re.search(p, q, re.IGNORECASE))
            if pat_matches > 0:
                conf += (pat_matches / len(cfg["patterns"])) * 0.3
            if conf > best_conf:
                best_conf = conf
                best_type = t
        
        return best_type, min(best_conf, 1.0)

    def should_trigger_web_search(self, query: str) -> bool:
        """Enhanced web search trigger detection"""
        q = query.lower()
        
        # Time-sensitive keywords
        time_keywords = [
            "current", "today", "now", "latest", "recent", "new", "fresh",
            "today's", "this week", "this month", "this year", "2024", "2025",
            "who is", "who was", "when is", "what is happening", "what happened",
            "current events", "breaking", "news", "update", "status",
            "weather", "temperature", "climate", "forecast",
            "stock", "price", "market", "economy", "rate",
            "election", "politics", "government", "president", "prime minister"
        ]
        
        # Check for exact matches and partial matches
        for keyword in time_keywords:
            if keyword in q:
                return True
                
        # Pattern-based detection
        time_patterns = [
            r"\bwhat.*(today|now|currently|latest)\b",
            r"\bhow.*(today|now|currently|latest)\b", 
            r"\bwho\s+is\s+the\s+(current|new|latest)\b",
            r"\bwhen\s+is\b",
            r"\bwhat\s+time\b",
            r"\btoday\b.*\b(date|time|weather|news)\b"
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, q, re.IGNORECASE):
                return True
                
        return False

    def should_trigger_doc_search(self, query: str) -> bool:
        """Check if query needs documentation search for latest syntax/versions"""
        q = query.lower()
        
        # Code-related keywords that might need updated docs
        doc_keywords = [
            "python", "javascript", "node", "react", "vue", "angular", "django", "flask",
            "openai", "anthropic", "gemini", "gpt", "claude", "model", "api",
            "library", "framework", "package", "import", "install", "version",
            "syntax", "function", "method", "class", "deprecat"
        ]
        
        # Patterns indicating need for latest documentation
        doc_patterns = [
            r"\bwrite.*code\b",
            r"\blatest.*version\b",
            r"\bhow.*to.*use\b",
            r"\bmodel.*name\b",
            r"\bapi.*key\b",
            r"\binstall.*package\b"
        ]
        
        # Check if query contains code-related terms
        has_code_keywords = any(keyword in q for keyword in doc_keywords)
        has_doc_patterns = any(re.search(pattern, q, re.IGNORECASE) for pattern in doc_patterns)
        
        return has_code_keywords or has_doc_patterns

    def select_optimal_model(self, query: str, chat_id: str, manual_selection: str = None) -> Tuple[str, Dict[str, Any]]:
        if manual_selection and manual_selection != "auto":
            routing_info = {
                "selected_model": manual_selection,
                "model_name": MODEL_CAPABILITIES[manual_selection]["name"],
                "model_icon": MODEL_CAPABILITIES[manual_selection]["icon"],
                "query_type": "manual",
                "complexity": "user_selected",
                "reasoning": f"Manually selected {MODEL_CAPABILITIES[manual_selection]['name']}",
                "manual_override": True
            }
            return manual_selection, routing_info

        complexity, complexity_conf = self.analyze_query_complexity(query)
        qtype, type_conf = self.classify_query_type(query)
        
        # Simple selection logic
        if qtype == "coding":
            if MODEL_CAPABILITIES["google"]["available"]:
                best = "google"
            elif MODEL_CAPABILITIES["openai"]["available"]:
                best = "openai"
            else:
                best = "claude"
        elif qtype == "writing":
            if MODEL_CAPABILITIES["claude"]["available"]:
                best = "claude"
            elif MODEL_CAPABILITIES["openai"]["available"]:
                best = "openai"
            else:
                best = "google"
        else:
            # Default order
            for model_key in ["google", "openai", "claude"]:
                if MODEL_CAPABILITIES[model_key]["available"]:
                    best = model_key
                    break
            else:
                best = "google"
        
        routing_info = {
            "selected_model": best,
            "model_name": MODEL_CAPABILITIES[best]["name"],
            "model_icon": MODEL_CAPABILITIES[best]["icon"],
            "query_type": qtype,
            "complexity": complexity,
            "complexity_confidence": complexity_conf,
            "type_confidence": type_conf,
            "reasoning": f"Auto-selected {MODEL_CAPABILITIES[best]['short_name']} for {qtype} ({complexity})",
            "manual_override": False
        }
        
        return best, routing_info

    def update_performance(self, model_key: str, chat_id: str, success: bool, processing_time: float):
        pass

def perform_web_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Enhanced DuckDuckGo HTML search scraping"""
    try:
        print(f"🔍 Performing web search for: {query}")
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.post(url, data=params, headers=headers, timeout=10)
        r.raise_for_status()
        text = r.text
        
        results = []
        link_pattern = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            re.IGNORECASE | re.DOTALL
        )
        
        snippet_pattern = re.compile(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>([^<]*)</a>',
            re.IGNORECASE | re.DOTALL
        )
        
        titles = link_pattern.findall(text)
        snippets = snippet_pattern.findall(text)
        
        print(f"Found {len(titles)} title matches and {len(snippets)} snippet matches")
        
        for i, (url_res, title_html) in enumerate(titles[:limit]):
            if url_res.startswith('//'):
                url_res = 'https:' + url_res
            elif url_res.startswith('/'):
                url_res = 'https://html.duckduckgo.com' + url_res
                
            title = re.sub(r'<.*?>', '', title_html).strip()
            title = unescape(title)
            
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<.*?>', '', snippets[i]).strip()
                snippet = unescape(snippet)
            
            if title and url_res:
                results.append({
                    "title": title[:120],
                    "snippet": snippet[:200] if snippet else f"Search result from {query}",
                    "url": url_res
                })
        
        if not results:
            print("Trying fallback extraction patterns...")
            alt_links = re.findall(r'href="([^"]*)"[^>]*>([^<]{20,})</a>', text)
            for url_res, title in alt_links[:limit]:
                if 'http' in url_res and len(title.strip()) > 10:
                    results.append({
                        "title": title.strip()[:80],
                        "snippet": f"Content related to {query}",
                        "url": url_res
                    })
        
        print(f"🔍 Web search returned {len(results)} results")
        return results[:limit]
        
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []

def search_official_docs(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Search official documentation for latest syntax and versions"""
    try:
        print(f"📚 Searching official docs for: {query}")
        
        # Enhanced documentation search queries
        doc_queries = []
        q_lower = query.lower()
        
        # Detect technology and create targeted searches
        if "python" in q_lower:
            doc_queries.append(f"site:python.org {query} latest")
            doc_queries.append(f"site:docs.python.org {query}")
        
        if any(term in q_lower for term in ["openai", "gpt"]):
            doc_queries.append(f"site:platform.openai.com {query} latest")
            doc_queries.append(f"site:openai.com/docs {query}")
        
        if any(term in q_lower for term in ["anthropic", "claude"]):
            doc_queries.append(f"site:docs.anthropic.com {query}")
            doc_queries.append(f"site:console.anthropic.com {query}")
            
        if any(term in q_lower for term in ["gemini", "google ai"]):
            doc_queries.append(f"site:ai.google.dev {query}")
            doc_queries.append(f"site:cloud.google.com/vertex-ai {query}")
            
        if "javascript" in q_lower:
            doc_queries.append(f"site:developer.mozilla.org {query}")
            doc_queries.append(f"site:nodejs.org {query}")
            
        if "react" in q_lower:
            doc_queries.append(f"site:reactjs.org {query} latest")
            doc_queries.append(f"site:react.dev {query}")
        
        # If no specific docs found, use general tech docs
        if not doc_queries:
            doc_queries.append(f"{query} official documentation latest")
            doc_queries.append(f"{query} site:docs.* OR site:*.org OR site:*.dev")
        
        all_results = []
        
        for doc_query in doc_queries[:2]:  # Limit to 2 doc queries to avoid rate limits
            try:
                search_results = perform_web_search(doc_query, limit=2)
                for result in search_results:
                    # Filter for documentation sites
                    url = result['url'].lower()
                    if any(doc_site in url for doc_site in [
                        'docs.', 'doc.', 'documentation', 'api.', 'developer.', 
                        'python.org', 'openai.com', 'anthropic.com', 'ai.google', 
                        'mozilla.org', 'nodejs.org', 'react', 'github.com'
                    ]):
                        all_results.append({
                            "title": f"📚 {result['title']}",
                            "content": result['snippet'],
                            "url": result['url']
                        })
                        
            except Exception as e:
                print(f"❌ Doc search failed for query '{doc_query}': {e}")
                continue
        
        # Remove duplicates and limit results
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
                if len(unique_results) >= limit:
                    break
        
        print(f"📚 Documentation search returned {len(unique_results)} results")
        return unique_results
        
    except Exception as e:
        print(f"❌ Documentation search error: {e}")
        return []

class MultiAI:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.google_models: Dict[str, Any] = {}
        self.openai_model = None
        self.claude_model = None
        self.router = IntelligentRouter(db)
        self.context_manager = ContextManager(db)
        self.init_models()

    def init_models(self):
        # Google Gemini - use EXACT model names from your working version
        if GOOGLE_AI_AVAILABLE and GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                preferred_google_models = [
                    "models/gemini-pro-latest",
                    "models/gemini-flash-latest"
                ]
                
                for model_name in preferred_google_models:
                    try:
                        gm = genai.GenerativeModel(model_name)
                        self.google_models[model_name] = gm
                        print(f"✅ Google Gemini ready: {model_name}")
                    except Exception as e:
                        print(f"❌ Could not init {model_name}: {e}")
                
                if not self.google_models:
                    print("⚠️ No Gemini models could be initialized.")
                        
            except Exception as e:
                print(f"❌ Google init failed: {e}")

        # OpenAI
        if LANGCHAIN_AVAILABLE and OPENAI_API_KEY:
            try:
                self.openai_model = ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model="gpt-4o-mini",
                    temperature=0.7
                )
                print("✅ OpenAI ready")
            except Exception as e:
                print(f"❌ OpenAI init failed: {e}")

        # Claude
        if LANGCHAIN_AVAILABLE and ANTHROPIC_API_KEY:
            try:
                self.claude_model = ChatAnthropic(
                    api_key=ANTHROPIC_API_KEY,
                    model="claude-3-5-haiku-20241022",
                    temperature=0.7
                )
                print("✅ Claude ready")
            except Exception as e:
                print(f"❌ Claude init failed: {e}")

    def get_model_by_key(self, model_key: str):
        if model_key == "google" and self.google_models:
            return "google", next(iter(self.google_models.values()))
        elif model_key == "openai" and self.openai_model:
            return "openai", self.openai_model
        elif model_key == "claude" and self.claude_model:
            return "claude", self.claude_model
        return None, None

    def seamless_fallback_to_gemini(self, context: str, original_routing_info: Dict[str, Any], original_error: str) -> Tuple[str, Dict[str, Any]]:
        """SEAMLESS fallback to Gemini - keeps original model info for UI"""
        print("🔄 Seamless fallback to Gemini (hidden from user)...")
        
        if not self.google_models:
            return f"Service temporarily unavailable. Please try again.", original_routing_info
        
        try:
            fallback_model = next(iter(self.google_models.values()))
            resp = fallback_model.generate_content(
                context,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'max_output_tokens': 2048,
                    'candidate_count': 1
                }
            )
            
            response_text = ""
            if hasattr(resp, 'text') and resp.text:
                response_text = resp.text
            elif hasattr(resp, 'candidates') and resp.candidates:
                candidate = resp.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
                    else:
                        response_text = str(candidate.content)
            
            if not response_text:
                response_text = "I'm here to help! Could you please rephrase your question?"
            
            # KEY: Keep original routing info - user never knows fallback occurred
            print("✅ Seamless fallback successful - user sees original model")
            return response_text, original_routing_info
            
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return f"Service temporarily unavailable. Please try again.", original_routing_info

    def generate_context_aware_response(self, prompt: str, chat_id: str, use_routing: bool = True, manual_model: str = None) -> Tuple[str, Dict[str, Any]]:
        start = time.time()

        # Smart documentation search for code-related queries
        doc_results = []
        web_results = []
        
        if self.router.should_trigger_doc_search(prompt):
            try:
                doc_results = search_official_docs(prompt, limit=3)
                if doc_results:
                    print(f"📚 Documentation search triggered: {len(doc_results)} results")
                else:
                    print("📚 Documentation search triggered but returned no results")
            except Exception as e:
                print(f"❌ Documentation search failed: {e}")
        
        # Regular web search for time-sensitive queries
        elif self.router.should_trigger_web_search(prompt):
            try:
                web_results = perform_web_search(prompt, limit=3)
                if web_results:
                    print(f"🔍 Web search triggered: {len(web_results)} results")
                else:
                    print("🔍 Web search triggered but returned no results")
            except Exception as e:
                print(f"❌ Web search failed: {e}")

        # Build context with results integrated
        context = self.context_manager.build_context_for_chat(chat_id, prompt, web_results, doc_results)

        if use_routing:
            selected_model, routing_info = self.router.select_optimal_model(prompt, chat_id, manual_model)
        else:
            if self.google_models:
                selected_model = "google"
            elif self.openai_model:
                selected_model = "openai"
            elif self.claude_model:
                selected_model = "claude"
            else:
                return "No AI models available", {"error": "No models"}

            routing_info = {
                "selected_model": selected_model,
                "model_name": MODEL_CAPABILITIES.get(selected_model, {}).get("name", selected_model),
                "reasoning": "fallback selection"
            }

        provider, model = self.get_model_by_key(routing_info.get("selected_model", selected_model))

        # If primary model not available, try seamless fallback
        if not model:
            print(f"⚠️ Primary model '{routing_info.get('selected_model')}' not available, using seamless fallback")
            return self.seamless_fallback_to_gemini(context, routing_info, "Primary model not available")

        try:
            response_text = ""
            
            if provider == "google":
                try:
                    resp = model.generate_content(
                        context,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ],
                        generation_config={
                            'temperature': 0.7,
                            'top_p': 0.8,
                            'max_output_tokens': 2048,
                            'candidate_count': 1
                        }
                    )
                    
                    if hasattr(resp, 'text') and resp.text:
                        response_text = resp.text
                    elif hasattr(resp, 'candidates') and resp.candidates:
                        candidate = resp.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                response_text = candidate.content.parts[0].text
                            else:
                                response_text = str(candidate.content)
                        elif hasattr(candidate, 'finish_reason'):
                            if candidate.finish_reason == 2:  # SAFETY
                                response_text = "I'll provide a helpful response for your request:\n\n"
                                if "hello world" in prompt.lower():
                                    response_text += """Here's a simple "Hello World" program in different languages:

## JavaScript
```javascript
console.log("Hello, World!");
```

## Python  
```python
print("Hello, World!")
```

## Java
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

## HTML
```html
<!DOCTYPE html>
<html>
<head>
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

These are basic "Hello World" programs to get you started with programming!"""
                            else:
                                response_text = "I understand your request. Let me provide a helpful response."
                    
                    if not response_text:
                        response_text = "I'm ready to help! Could you please rephrase your question?"
                    
                except Exception as e:
                    print(f"❌ Google model error: {e}")
                    return self.seamless_fallback_to_gemini(context, routing_info, f"Google model error: {e}")
                    
            elif provider == "openai":
                try:
                    resp = self.openai_model.invoke(context)
                    response_text = getattr(resp, 'content', str(resp))
                except Exception as e:
                    print(f"❌ OpenAI model error: {e}")
                    return self.seamless_fallback_to_gemini(context, routing_info, f"OpenAI model error: {e}")
                    
            elif provider == "claude":
                try:
                    resp = self.claude_model.invoke(context)
                    response_text = getattr(resp, 'content', str(resp))
                except Exception as e:
                    print(f"❌ Claude model error: {e}")
                    return self.seamless_fallback_to_gemini(context, routing_info, f"Claude model error: {e}")

            processing_time = time.time() - start
            routing_info["processing_time"] = processing_time
            routing_info["web_results"] = web_results
            routing_info["doc_results"] = doc_results
            routing_info["web_search_triggered"] = len(web_results) > 0
            routing_info["doc_search_triggered"] = len(doc_results) > 0

            success = len(response_text) > 5
            self.router.update_performance(routing_info.get("selected_model", "unknown"), chat_id, success, processing_time)
            self.context_manager.update_memory_after_message(chat_id)

            return response_text, routing_info

        except Exception as e:
            processing_time = time.time() - start
            err = str(e)
            print(f"❌ Unexpected error: {err}")
            return self.seamless_fallback_to_gemini(context, routing_info, f"Unexpected error: {err}")

def copy_to_clipboard(text: str, page: ft.Page):
    """Copy text to clipboard with visual feedback"""
    try:
        pyperclip.copy(text)
        page.snack_bar = ft.SnackBar(
            content=ft.Text("✅ Copied to clipboard!", color=ft.colors.GREEN_400),
            bgcolor=ft.colors.with_opacity(0.8, ft.colors.GREEN_900)
        )
        page.snack_bar.open = True
        page.update()
    except Exception as e:
        print(f"Copy failed: {e}")
        page.snack_bar = ft.SnackBar(
            content=ft.Text("❌ Copy failed", color=ft.colors.RED_400),
            bgcolor=ft.colors.with_opacity(0.8, ft.colors.RED_900)
        )
        page.snack_bar.open = True
        page.update()

def create_professional_content(text: str, page: ft.Page) -> ft.Column:
    """Create professional content with proper typography and code highlighting"""
    
    content_blocks = []
    current_text = ""
    in_code_block = False
    code_language = ""
    code_content = ""
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_language = line[3:].strip() or "text"
                if current_text.strip():
                    content_blocks.append(create_text_content(current_text.strip()))
                    current_text = ""
                code_content = ""
            else:
                in_code_block = False
                if code_content.strip():
                    content_blocks.append(create_code_block(code_content.strip(), code_language, page))
                code_content = ""
                code_language = ""
        elif in_code_block:
            code_content += line + '\n'
        else:
            current_text += line + '\n'
        
        i += 1
    
    if current_text.strip():
        content_blocks.append(create_text_content(current_text.strip()))
    if in_code_block and code_content.strip():
        content_blocks.append(create_code_block(code_content.strip(), code_language, page))
    
    return ft.Column(content_blocks, spacing=16)

def create_text_content(text: str) -> ft.Column:
    """Create formatted text content with proper typography"""
    elements = []
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if para.startswith('##'):
            heading_text = para[2:].strip()
            elements.append(
                ft.Text(
                    heading_text,
                    size=20,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.WHITE,
                    style=ft.TextStyle(decoration=ft.TextDecoration.NONE)
                )
            )
        elif para.startswith('#'):
            heading_text = para[1:].strip()
            elements.append(
                ft.Text(
                    heading_text,
                    size=24,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.BLUE_400,
                    style=ft.TextStyle(decoration=ft.TextDecoration.NONE)
                )
            )
        elif para.startswith('**') and para.endswith('**'):
            bold_text = para[2:-2].strip()
            elements.append(
                ft.Text(
                    bold_text,
                    size=14,
                    weight=ft.FontWeight.BOLD,
                    color=ft.colors.ORANGE_300
                )
            )
        elif para.startswith('•') or para.startswith('-') or para.startswith('*'):
            list_text = para[1:].strip()
            elements.append(
                ft.Row([
                    ft.Icon(ft.icons.CIRCLE, size=6, color=ft.colors.BLUE_400),
                    ft.Text(list_text, size=14, color=ft.colors.WHITE, expand=True)
                ], spacing=8)
            )
        else:
            if '`' in para:
                parts = re.split(r'`([^`]+)`', para)
                row_elements = []
                for j, part in enumerate(parts):
                    if j % 2 == 0:
                        if part.strip():
                            row_elements.append(ft.Text(part, size=14, color=ft.colors.WHITE))
                    else:
                        row_elements.append(
                            ft.Container(
                                content=ft.Text(part, size=12, color=ft.colors.GREEN_300, font_family="Consolas"),
                                bgcolor=ft.colors.with_opacity(0.2, ft.colors.GREEN_900),
                                padding=ft.padding.symmetric(horizontal=4, vertical=2),
                                border_radius=4
                            )
                        )
                if row_elements:
                    elements.append(ft.Row(row_elements, wrap=True))
            else:
                elements.append(
                    ft.Text(
                        para,
                        size=14,
                        color=ft.colors.WHITE,
                        selectable=True
                    )
                )
    
    return ft.Column(elements, spacing=12)

def create_code_block(code: str, language: str, page: ft.Page) -> ft.Container:
    """Create a professional code block with copy button"""
    
    lang_colors = {
        "python": ft.colors.BLUE_400,
        "javascript": ft.colors.YELLOW_600,
        "java": ft.colors.RED_400,
        "html": ft.colors.ORANGE_400,
        "css": ft.colors.PURPLE_400,
        "json": ft.colors.GREEN_400,
        "bash": ft.colors.GREY_400,
        "sql": ft.colors.CYAN_400
    }
    
    lang_color = lang_colors.get(language.lower(), ft.colors.GREY_400)
    
    def copy_code(e):
        copy_to_clipboard(code, page)
    
    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Icon(ft.icons.CODE, size=16, color=lang_color),
                ft.Text(language.upper(), size=12, weight=ft.FontWeight.BOLD, color=lang_color)
            ], spacing=6),
            ft.Container(expand=True),
            ft.IconButton(
                icon=ft.icons.COPY,
                icon_size=16,
                tooltip="Copy code",
                on_click=copy_code,
                icon_color=ft.colors.GREY_400,
                hover_color=ft.colors.with_opacity(0.1, ft.colors.WHITE)
            )
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        padding=ft.padding.symmetric(horizontal=16, vertical=8),
        bgcolor=ft.colors.with_opacity(0.05, ft.colors.WHITE),
        border=ft.border.only(bottom=ft.border.BorderSide(1, ft.colors.with_opacity(0.1, ft.colors.WHITE)))
    )
    
    code_text = ft.Container(
        content=ft.Text(
            code,
            size=13,
            font_family="Consolas",
            color=ft.colors.WHITE,
            selectable=True
        ),
        padding=ft.padding.all(16),
        bgcolor=ft.colors.with_opacity(0.02, ft.colors.WHITE)
    )
    
    return ft.Container(
        content=ft.Column([header, code_text], spacing=0),
        bgcolor=ft.colors.with_opacity(0.03, ft.colors.WHITE),
        border_radius=8,
        border=ft.border.all(1, ft.colors.with_opacity(0.1, ft.colors.WHITE)),
        margin=ft.margin.symmetric(vertical=8)
    )

def main(page: ft.Page):
    page.title = "🕳️ BLACKHOLE AI - Seamless"
    page.theme_mode = ft.ThemeMode.DARK
    
    try:
        page.window.width = 1500
        page.window.height = 1000 
        page.window.resizable = True
    except:
        pass
        
    page.padding = 0
    page.spacing = 0

    db = DatabaseManager()
    ai = MultiAI(db)

    # State
    current_chat_id = None
    chats: List[Chat] = []
    current_messages: List[Message] = []
    current_model_selection = "auto"

    available_models = []
    if ai.google_models:
        available_models.append("🌟 Google AI")
    if ai.openai_model:
        available_models.append("🧠 OpenAI")
    if ai.claude_model:
        available_models.append("💭 Claude")

    # UI components
    chat_list = ft.ListView(expand=True, spacing=6, padding=10)
    chat_display = ft.ListView(expand=True, spacing=20, padding=20, auto_scroll=True)

    input_field = ft.TextField(
        hint_text="Ask anything... I'll check latest docs for code-related queries 📚",
        multiline=True,
        min_lines=1,
        max_lines=6,
        expand=True,
        border_radius=16,
        filled=True,
        bgcolor=ft.colors.with_opacity(0.06, ft.colors.WHITE),
        border_color=ft.colors.TRANSPARENT,
        focused_border_color=ft.colors.BLUE_400,
        hint_style=ft.TextStyle(color=ft.colors.GREY_400),
        text_style=ft.TextStyle(size=14),
        content_padding=ft.padding.symmetric(horizontal=18, vertical=12),
    )

    current_chat_title = ft.Text("Select a chat to start", size=22, weight=ft.FontWeight.W_600, color=ft.colors.WHITE)

    model_dropdown = ft.Dropdown(
        hint_text="🤖 Select AI Model",
        options=[ft.dropdown.Option("auto", "🤖 Auto-Select (Smart Routing)")],
        value="auto",
        width=280,  # Slightly smaller width
        border_radius=12,
        text_style=ft.TextStyle(size=13),
    )

    routing_switch_toggle = ft.Switch(
        value=True,
        active_color=ft.colors.PURPLE_400,
    )

    routing_switch = ft.Row([
        ft.Icon(ft.icons.AUTO_AWESOME, color=ft.colors.PURPLE_400, size=18),
        routing_switch_toggle,
        ft.Text("Smart Routing", color=ft.colors.PURPLE_400, size=13)
    ], spacing=8)

    send_btn = ft.Container(
        content=ft.Icon(ft.icons.SEND_ROUNDED, color=ft.colors.WHITE, size=20),
        bgcolor=ft.colors.BLUE_600,
        border_radius=24,
        width=52,
        height=52,
        alignment=ft.alignment.center,
        ink=True,
    )

    status_indicator = ft.Container(
        content=ft.Row([
            ft.Icon(ft.icons.CIRCLE, color=ft.colors.GREEN_400, size=8),
            ft.Text("Ready", color=ft.colors.GREEN_400, size=12)
        ], spacing=8),
        padding=ft.padding.symmetric(horizontal=12, vertical=6),
        bgcolor=ft.colors.with_opacity(0.08, ft.colors.GREEN_400),
        border_radius=20
    )

    def create_message_bubble(message: Message) -> ft.Container:
        if message.role == "user":
            inner = ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Text("👤", size=16),
                        width=36,
                        height=36,
                        bgcolor=ft.colors.BLUE_600,
                        border_radius=18,
                        alignment=ft.alignment.center
                    ),
                    ft.Text("You", weight=ft.FontWeight.W_600, color=ft.colors.BLUE_400, size=16),
                    ft.Container(expand=True),
                    ft.Text(message.timestamp.strftime("%H:%M"), size=12, color=ft.colors.GREY_500)
                ], spacing=12),
                ft.Container(
                    content=ft.Text(
                        message.content,
                        size=15,
                        color=ft.colors.WHITE,
                        selectable=True
                    ),
                    margin=ft.margin.only(left=48)
                )
            ], spacing=12)
            
            return ft.Container(
                content=ft.Container(
                    content=inner,
                    padding=ft.padding.all(20),
                    bgcolor=ft.colors.with_opacity(0.05, ft.colors.BLUE_600),
                    border_radius=16,
                    border=ft.border.all(1, ft.colors.with_opacity(0.1, ft.colors.BLUE_400))
                ),
                margin=ft.margin.only(left=60, bottom=20)
            )
        
        else:  # Assistant message
            model_cfg = None
            if message.routing_info and message.routing_info.get("selected_model"):
                mk = message.routing_info["selected_model"]
                if mk in MODEL_CAPABILITIES:
                    model_cfg = MODEL_CAPABILITIES[mk]

            # ALWAYS show original intended model (never show fallback info)
            model_name = message.routing_info.get("model_name", message.model_used or "AI") if message.routing_info else message.model_used or "AI"
            model_icon = message.routing_info.get("model_icon", "🤖") if message.routing_info else "🤖"

            header = ft.Row([
                ft.Container(
                    content=ft.Text(model_icon, size=16),
                    width=36,
                    height=36,
                    bgcolor=model_cfg["color"] if model_cfg else ft.colors.GREEN_700,
                    border_radius=18,
                    alignment=ft.alignment.center
                ),
                ft.Column([
                    ft.Text(
                        model_cfg["short_name"] if model_cfg else model_name.split()[0],
                        weight=ft.FontWeight.W_700,
                        color=model_cfg["color"] if model_cfg else ft.colors.GREEN_400,
                        size=16
                    ),
                    ft.Text(
                        f"{message.query_type or 'general'} • {message.complexity or 'auto'}",
                        size=11,
                        color=ft.colors.GREY_400
                    )
                ], spacing=2),
                ft.Container(expand=True),
                ft.Text(message.timestamp.strftime("%H:%M"), size=12, color=ft.colors.GREY_500)
            ], spacing=12)
            
            content_container = ft.Container(
                content=create_professional_content(message.content, page),
                margin=ft.margin.only(left=48, top=8)
            )
            
            controls = [header, content_container]
            
            # Show documentation search indicator
            if message.routing_info and message.routing_info.get("doc_search_triggered"):
                doc_results = message.routing_info.get('doc_results', [])
                doc_indicator = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.LIBRARY_BOOKS, size=16, color=ft.colors.PURPLE_300),
                        ft.Text(f"Used {len(doc_results)} official docs", size=12, weight=ft.FontWeight.W_500, color=ft.colors.PURPLE_300)
                    ], spacing=8),
                    bgcolor=ft.colors.with_opacity(0.1, ft.colors.PURPLE_400),
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    border_radius=8,
                    margin=ft.margin.only(left=48, top=12)
                )
                controls.append(doc_indicator)
            
            # Show web search indicator
            elif message.routing_info and message.routing_info.get("web_search_triggered"):
                web_results = message.routing_info.get('web_results', [])
                web_indicator = ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.SEARCH, size=16, color=ft.colors.BLUE_300),
                        ft.Text(f"Used {len(web_results)} web sources", size=12, weight=ft.FontWeight.W_500, color=ft.colors.BLUE_300)
                    ], spacing=8),
                    bgcolor=ft.colors.with_opacity(0.1, ft.colors.BLUE_400),
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    border_radius=8,
                    margin=ft.margin.only(left=48, top=12)
                )
                controls.append(web_indicator)
            
            # REMOVED: No fallback indicators shown to user
            
            if message.routing_info and message.routing_info.get("reasoning"):
                controls.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.icons.AUTO_AWESOME, size=14, color=ft.colors.PURPLE_300),
                            ft.Text(
                                message.routing_info.get("reasoning", ""),
                                size=11,
                                color=ft.colors.PURPLE_300
                            )
                        ], spacing=8),
                        bgcolor=ft.colors.with_opacity(0.08, ft.colors.PURPLE_400),
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        border_radius=8,
                        margin=ft.margin.only(left=48, top=8)
                    )
                )
            
            if message.processing_time:
                perf = f"⏱️ {message.processing_time:.1f}s"
                controls.append(
                    ft.Container(
                        content=ft.Text(perf, size=10, color=ft.colors.GREY_400),
                        margin=ft.margin.only(left=48, top=4)
                    )
                )

            return ft.Container(
                content=ft.Container(
                    content=ft.Column(controls, spacing=0),
                    padding=ft.padding.all(20),
                    bgcolor=ft.colors.with_opacity(0.03, ft.colors.GREEN_600),
                    border_radius=16,
                    border=ft.border.all(1, ft.colors.with_opacity(0.08, ft.colors.GREEN_400))
                ),
                margin=ft.margin.only(right=60, bottom=20)
            )

    def update_model_dropdown():
        opts = [ft.dropdown.Option("auto", "🤖 Auto-Select (Smart Routing)")]
        for key, cfg in MODEL_CAPABILITIES.items():
            if cfg.get("available"):
                opts.append(ft.dropdown.Option(key, f"{cfg.get('icon','')} {cfg.get('short_name','')}"))
        model_dropdown.options = opts
        model_dropdown.value = current_model_selection
        page.update()

    def set_model_selection(value: str):
        nonlocal current_model_selection
        current_model_selection = value
        if value == "auto":
            update_status("Smart routing enabled", ft.colors.PURPLE_400)
        else:
            if value in MODEL_CAPABILITIES:
                update_status(f"Using {MODEL_CAPABILITIES[value]['short_name']}", MODEL_CAPABILITIES[value]["color"])

    def toggle_routing_mode(is_enabled: bool):
        model_dropdown.disabled = is_enabled
        if is_enabled:
            model_dropdown.value = "auto"
            set_model_selection("auto")
        else:
            update_status("Manual model selection", ft.colors.ORANGE_400)
        page.update()

    def update_status(msg: str, color=ft.colors.GREEN_400):
        status_indicator.content.controls[0].color = color
        status_indicator.content.controls[1].value = msg
        status_indicator.content.controls[1].color = color
        status_indicator.bgcolor = ft.colors.with_opacity(0.08, color)
        page.update()

    def generate_chat_title(first_message: str) -> str:
        words = first_message.split()
        return " ".join(words[:TITLE_FROM_FIRST_USER_MSG]) + ("..." if len(words) > TITLE_FROM_FIRST_USER_MSG else "")

    def load_chats():
        nonlocal chats
        chats = db.get_all_chats()
        if not chats:
            default_chat = db.create_chat("Welcome to BLACKHOLE AI")
            if default_chat:
                chats = [default_chat]
        update_chat_list()

    def update_chat_list():
        chat_list.controls.clear()
        
        new_chat_btn = ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Icon(ft.icons.ADD_ROUNDED, size=18, color=ft.colors.WHITE),
                    bgcolor=ft.colors.BLUE_600,
                    width=36,
                    height=36,
                    alignment=ft.alignment.center,
                    border_radius=10
                ),
                ft.Text("New Chat", weight=ft.FontWeight.W_600, size=15)
            ], spacing=12),
            padding=ft.padding.all(16),
            bgcolor=ft.colors.BLUE_800,
            border_radius=12,
            on_click=lambda e: create_new_chat(),
        )
        chat_list.controls.append(new_chat_btn)

        for c in chats:
            msgs = db.get_chat_messages(c.id)
            is_selected = (c.id == current_chat_id)
            
            item = ft.Container(
                content=ft.Row([
                    ft.Column([
                        ft.Text(
                            c.title,
                            weight=ft.FontWeight.W_600 if is_selected else ft.FontWeight.W_500,
                            size=14,
                            color=ft.colors.WHITE if is_selected else ft.colors.GREY_300,
                            max_lines=2,
                            overflow=ft.TextOverflow.ELLIPSIS
                        ),
                        ft.Row([
                            ft.Text(c.updated_at.strftime("%b %d, %H:%M"), size=11, color=ft.colors.GREY_400),
                            ft.Text("•", size=11, color=ft.colors.GREY_500),
                            ft.Text(f"{len(msgs)} msgs", size=11, color=ft.colors.GREY_500)
                        ], spacing=6)
                    ], expand=True, spacing=6),
                    ft.PopupMenuButton(
                        icon=ft.icons.MORE_VERT_ROUNDED,
                        icon_size=18,
                        items=[
                            ft.PopupMenuItem(text="✏️ Rename", on_click=lambda e, chat_id=c.id: rename_chat(chat_id)),
                            ft.PopupMenuItem(text="🗑️ Delete", on_click=lambda e, chat_id=c.id: delete_chat(chat_id))
                        ]
                    )
                ], vertical_alignment=ft.CrossAxisAlignment.START),
                padding=ft.padding.all(12),
                border_radius=10,
                bgcolor=ft.colors.with_opacity(0.1, ft.colors.BLUE_600) if is_selected else None,
                border=ft.border.all(1, ft.colors.with_opacity(0.3, ft.colors.BLUE_400)) if is_selected else None,
                on_click=lambda e, chat_id=c.id: select_chat(chat_id),
                margin=ft.margin.only(bottom=8)
            )
            chat_list.controls.append(item)
        page.update()

    def create_new_chat():
        nonlocal chats
        new_c = db.create_chat()
        if new_c:
            chats.insert(0, new_c)
            select_chat(new_c.id)
            update_chat_list()
            update_status("New chat created", ft.colors.GREEN_400)

    def select_chat(chat_id: str):
        nonlocal current_chat_id, current_messages
        current_chat_id = chat_id
        sel = next((x for x in chats if x.id == chat_id), None)
        if sel:
            current_chat_title.value = sel.title
        current_messages = db.get_chat_messages(chat_id)
        update_chat_display()
        update_chat_list()
        update_status(f"Chat selected: {len(current_messages)} messages", ft.colors.BLUE_400)

    def rename_chat(chat_id: str):
        sel = next((x for x in chats if x.id == chat_id), None)
        title_field = ft.TextField(
            value=sel.title if sel else "", 
            expand=True, 
            border_radius=12, 
            filled=True,
            text_style=ft.TextStyle(size=14)
        )
        
        def save(e):
            nt = title_field.value.strip()
            if nt:
                db.update_chat_title(chat_id, nt)
                for ch in chats:
                    if ch.id == chat_id:
                        ch.title = nt
                        break
                if chat_id == current_chat_id:
                    current_chat_title.value = nt
                update_chat_list()
                update_status("Chat renamed", ft.colors.GREEN_400)
                page.close(dlg)
        
        dlg = ft.AlertDialog(
            title=ft.Text("Rename Chat", size=18, weight=ft.FontWeight.BOLD),
            content=ft.Container(content=title_field, width=400, padding=ft.padding.symmetric(vertical=10)),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: page.close(dlg)),
                ft.ElevatedButton("Save", bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE, on_click=save)
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        page.open(dlg)
        title_field.focus()

    def delete_chat(chat_id: str):
        nonlocal chats, current_chat_id
        sel = next((x for x in chats if x.id == chat_id), None)
        
        def confirm_delete(e):
            nonlocal chats, current_chat_id
            db.delete_chat(chat_id)
            chats = [c for c in chats if c.id != chat_id]
            if current_chat_id == chat_id:
                if chats:
                    select_chat(chats[0].id)
                else:
                    new_c = db.create_chat()
                    if new_c:
                        chats = [new_c]
                        select_chat(new_c.id)
            update_chat_list()
            update_status("🗑️ Chat deleted", ft.colors.ORANGE_400)
            page.close(confirm_dialog)
        
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("Delete Chat", weight=ft.FontWeight.BOLD, size=18),
            content=ft.Column([
                ft.Text("Are you sure you want to delete this chat?", size=14),
                ft.Container(
                    content=ft.Text(sel.title if sel else "", style=ft.TextStyle(italic=True), color=ft.colors.GREY_400, size=13),
                    padding=ft.padding.symmetric(vertical=8)
                ),
            ], tight=True, spacing=8),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: page.close(confirm_dialog)),
                ft.ElevatedButton("Delete", bgcolor=ft.colors.RED_600, color=ft.colors.WHITE, on_click=confirm_delete),
            ],
            actions_alignment=ft.MainAxisAlignment.END
        )
        page.open(confirm_dialog)

    def update_chat_display():
        chat_display.controls.clear()
        
        if not current_chat_id:
            welcome = ft.Container(
                content=ft.Column([
                    ft.Text("🕳️", size=64, text_align=ft.TextAlign.CENTER),
                    ft.Text("BLACKHOLE AI", size=28, weight=ft.FontWeight.BOLD, 
                           text_align=ft.TextAlign.CENTER, color=ft.colors.BLUE_400),
                    ft.Text("Seamless AI Assistant", size=16, weight=ft.FontWeight.W_500,
                           text_align=ft.TextAlign.CENTER, color=ft.colors.GREY_400),
                    ft.Text("🔹 Smart documentation search for code queries", 
                           size=14, text_align=ft.TextAlign.CENTER, color=ft.colors.PURPLE_400),
                    ft.Text("🔹 Seamless fallback - no interruptions", 
                           size=14, text_align=ft.TextAlign.CENTER, color=ft.colors.ORANGE_400),
                    ft.Text("Select a chat to start or create a new one", 
                           size=14, text_align=ft.TextAlign.CENTER, color=ft.colors.GREY_500)
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                padding=ft.padding.all(60)
            )
            chat_display.controls.append(welcome)
            page.update()
            return

        for m in current_messages:
            chat_display.controls.append(create_message_bubble(m))
        page.update()

    def send_message(e=None):
        nonlocal current_messages
        if not current_chat_id:
            create_new_chat()
            return

        txt = input_field.value.strip()
        if not txt:
            return

        input_field.value = ""
        send_btn.content = ft.ProgressRing(width=20, height=20, stroke_width=2, color=ft.colors.WHITE)
        send_btn.bgcolor = ft.colors.GREY_600
        page.update()

        user_msg = Message(
            id=str(uuid.uuid4()),
            chat_id=current_chat_id,
            role="user",
            content=txt,
            timestamp=datetime.now()
        )
        db.save_message(user_msg)
        current_messages.append(user_msg)

        current_chat = next((c for c in chats if c.id == current_chat_id), None)
        if current_chat and (current_chat.title == "New Chat" or current_chat.title.startswith("New Chat")):
            new_title = generate_chat_title(txt)
            db.update_chat_title(current_chat_id, new_title)
            current_chat.title = new_title
            current_chat_title.value = new_title

        update_chat_display()
        
        # Enhanced status messages
        if ai.router.should_trigger_doc_search(txt):
            update_status("Searching official docs & thinking...", ft.colors.PURPLE_400)
        elif ai.router.should_trigger_web_search(txt):
            update_status("Searching web & thinking...", ft.colors.ORANGE_400)
        else:
            update_status("Thinking...", ft.colors.ORANGE_400)

        def process_ai_response():
            try:
                use_routing = routing_switch_toggle.value
                manual_model = current_model_selection if current_model_selection != "auto" else None

                response, routing_info = ai.generate_context_aware_response(txt, current_chat_id, use_routing, manual_model)

                ai_msg = Message(
                    id=str(uuid.uuid4()),
                    chat_id=current_chat_id,
                    role="assistant",
                    content=response,
                    timestamp=datetime.now(),
                    model_used=routing_info.get("model_name"),
                    query_type=routing_info.get("query_type"),
                    complexity=routing_info.get("complexity"),
                    processing_time=routing_info.get("processing_time"),
                    routing_info=routing_info
                )
                db.save_message(ai_msg)
                current_messages.append(ai_msg)

                def update_ui():
                    update_chat_display()
                    processing_time = routing_info.get("processing_time", 0)
                    model_name = routing_info.get("model_name", "AI").split()[0]
                    
                    status_msg = f"✅ {model_name} responded in {processing_time:.1f}s"
                    
                    # REMOVED: No fallback status messages - user never sees fallback
                    
                    if routing_info.get("doc_search_triggered"):
                        status_msg += " (with docs)"
                        status_color = ft.colors.PURPLE_400
                    elif routing_info.get("web_search_triggered"):
                        status_msg += " (with web search)"
                        status_color = ft.colors.BLUE_400
                    else:
                        status_color = ft.colors.GREEN_400
                    
                    update_status(status_msg, status_color)
                    
                    send_btn.content = ft.Icon(ft.icons.SEND_ROUNDED, color=ft.colors.WHITE, size=20)
                    send_btn.bgcolor = ft.colors.BLUE_600
                    update_chat_list()

                update_ui()

            except Exception as ex:
                err_msg = Message(
                    id=str(uuid.uuid4()),
                    chat_id=current_chat_id,
                    role="assistant",
                    content=f"I encountered an error: {ex}",
                    timestamp=datetime.now(),
                    model_used="Error"
                )
                db.save_message(err_msg)
                current_messages.append(err_msg)
                
                def update_error_ui():
                    update_chat_display()
                    update_status("❌ Error occurred", ft.colors.RED_400)
                    send_btn.content = ft.Icon(ft.icons.SEND_ROUNDED, color=ft.colors.WHITE, size=20)
                    send_btn.bgcolor = ft.colors.BLUE_600
                    update_chat_list()
                
                update_error_ui()

        threading.Thread(target=process_ai_response, daemon=True).start()

    # Wire up events
    send_btn.on_click = lambda e: send_message()
    input_field.on_submit = lambda e: send_message()
    model_dropdown.on_change = lambda e: set_model_selection(e.control.value)
    routing_switch_toggle.on_change = lambda e: toggle_routing_mode(e.control.value)

    # Layout
    sidebar = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Column([
                    ft.Text("🕳️ BLACKHOLE", size=20, weight=ft.FontWeight.BOLD, color=ft.colors.WHITE),
                    ft.Text("AI Seamless", size=13, color=ft.colors.BLUE_400, weight=ft.FontWeight.W_500)
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=ft.padding.symmetric(vertical=20)
            ),
            ft.Divider(color=ft.colors.with_opacity(0.1, ft.colors.WHITE)),
            chat_list
        ]),
        width=340,
        bgcolor=ft.colors.GREY_900,
        border=ft.border.only(right=ft.border.BorderSide(1, ft.colors.with_opacity(0.08, ft.colors.WHITE)))
    )

    # FINAL HEADER FIX with proper vertical spacing
    header = ft.Container(
        content=ft.Row([
            # Left column - Title and status (with MORE vertical spacing)
            ft.Column([
                current_chat_title,
                ft.Container(height=12),  # ADD VERTICAL SPACER - KEY FIX!
                ft.Row([
                    ft.Text("📚 Doc-Aware", size=11, color=ft.colors.PURPLE_400, weight=ft.FontWeight.W_500),
                    ft.Text("•", size=11, color=ft.colors.GREY_500),
                    ft.Text("🔄 Seamless-Fallback", size=11, color=ft.colors.ORANGE_400, weight=ft.FontWeight.W_500),
                    ft.Text("•", size=11, color=ft.colors.GREY_500),
                    ft.Text(f"Today: {CURRENT_DATE}", size=11, color=ft.colors.GREEN_400)
                ], spacing=6)
            ], expand=True),
            # Right column - Controls 
            ft.Row([
                model_dropdown,
                ft.Container(width=16),
                routing_switch,
                ft.Container(width=16),
                status_indicator
            ], spacing=12)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.START),  # KEY: Added vertical alignment
        padding=ft.padding.symmetric(horizontal=24, vertical=18),  # Slightly more vertical padding
        bgcolor=ft.colors.GREY_800,
        border=ft.border.only(bottom=ft.border.BorderSide(1, ft.colors.with_opacity(0.08, ft.colors.WHITE)))
    )

    input_area = ft.Container(
        content=ft.Row([input_field, send_btn], spacing=16),
        padding=ft.padding.symmetric(horizontal=24, vertical=16),
        bgcolor=ft.colors.GREY_800,
        border=ft.border.only(top=ft.border.BorderSide(1, ft.colors.with_opacity(0.08, ft.colors.WHITE)))
    )

    main_content = ft.Column([
        header,
        ft.Container(
            content=chat_display,
            expand=True,
            bgcolor=ft.colors.GREY_900
        ),
        input_area
    ], expand=True)

    page.add(ft.Row([sidebar, ft.Container(content=main_content, expand=True)], expand=True))

    # Keyboard shortcuts
    def on_key(e: ft.KeyboardEvent):
        if e.key == "Enter" and e.ctrl:
            send_message()
        elif e.key == "F1":
            create_new_chat()
        elif e.key == "Escape":
            input_field.focus()
    
    page.on_keyboard_event = on_key

    # Initialize
    update_model_dropdown()
    load_chats()
    if chats:
        select_chat(chats[0].id)
    else:
        create_new_chat()
    
    input_field.focus()
    update_status("Seamless AI ready - Smart docs with invisible fallback", ft.colors.GREEN_400)
    print("✅ BLACKHOLE AI Seamless ready!")

def run_self_tests():
    print("🧪 Running seamless fallback self-tests...\n")
    
    db = DatabaseManager("test_" + DATABASE_FILE)
    ai = MultiAI(db)
    
    test_cases = [
        "write python code with latest syntax for file handling",
        "show me current OpenAI model names for API",
        "write a java code for hello world", 
        "Who is the current Prime Minister of India?",
        "What's the weather in New Delhi today?"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"Test {i}: {query}")
        
        router = IntelligentRouter(db)
        query_type, confidence = router.classify_query_type(query)
        complexity, complexity_conf = router.analyze_query_complexity(query)
        
        print(f"  📝 Classification: {query_type} (confidence: {confidence:.2f})")
        print(f"  🔧 Complexity: {complexity} (score: {complexity_conf:.2f})")
        
        # Test documentation search trigger
        doc_triggered = router.should_trigger_doc_search(query)
        print(f"  📚 Documentation search triggered: {doc_triggered}")
        
        if doc_triggered:
            try:
                doc_results = search_official_docs(query, limit=2)
                if doc_results:
                    print(f"  📖 Documentation search returned: {len(doc_results)} results")
                    for j, result in enumerate(doc_results):
                        print(f"    [DOC-{j+1}] {result['title'][:60]}...")
                else:
                    print("  📖 Documentation search returned no results")
            except Exception as e:
                print(f"  ❌ Documentation search failed: {e}")
        
        # Test web search trigger
        web_triggered = router.should_trigger_web_search(query)
        print(f"  🔍 Web search triggered: {web_triggered}")
        
        if web_triggered:
            try:
                web_results = perform_web_search(query, limit=2)
                if web_results:
                    print(f"  🌐 Web search returned: {len(web_results)} results")
                    for j, result in enumerate(web_results):
                        print(f"    [{j+1}] {result['title'][:60]}...")
                else:
                    print("  🌐 Web search returned no results")
            except Exception as e:
                print(f"  ❌ Web search failed: {e}")
        
        selected_model, routing_info = router.select_optimal_model(query, "test_chat")
        print(f"  🤖 Selected model: {routing_info.get('model_name', 'Unknown')}")
        print(f"  💭 Reasoning: {routing_info.get('reasoning', 'No reasoning')}")
        print()
    
    # Cleanup
    import os
    if os.path.exists("test_" + DATABASE_FILE):
        os.remove("test_" + DATABASE_FILE)
    
    print("✅ Seamless fallback tests completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        run_self_tests()
    else:
        try:
            ft.app(main)
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            print("\nTry running with --selftest to check functionality")