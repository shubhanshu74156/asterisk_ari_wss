
import socket
import threading
import wave
import struct
import numpy as np
import asyncio
import os
import tempfile
import time
from openai import OpenAI
from dotenv import load_dotenv
import random
import json
import requests
import sqlite3
from datetime import datetime, timedelta
import aiohttp
import aiofiles


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🤖 OpenAI Client Initialized")

DEBUG = True

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class CustomFunctionLibrary:
    """Extended library with custom functions for API calls and external operations"""

    def __init__(self):
        print("🔧 Custom Function Library initialized")
        # Initialize any connections or configurations
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "your_api_key_here")
        self.news_api_key = os.getenv("NEWS_API_KEY", "your_api_key_here")
        self.db_path = "call_data.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing call data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_message TEXT,
                    ai_response TEXT,
                    functions_called TEXT
                )
            """)
            conn.commit()
            conn.close()
            print("📊 Database initialized successfully")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")

    # ============= BASIC FUNCTIONS (from previous example) =============
    def end_call(self, reason="User requested"):
        """End the current call"""
        print(f"📞 [FUNCTION] End call function initiated - Reason: {reason}")
        result = {"action": "end_call", "reason": reason, "message": f"Call ended: {reason}"}
        print("✅ [FUNCTION] End call function executed")
        return result

    def roll_dice(self, num_dice=1, sides=6):
        """Roll one or more dice"""
        print(f"🎲 [FUNCTION] Rolling {num_dice} dice with {sides} sides each")
        results = [random.randint(1, sides) for _ in range(num_dice)]
        total = sum(results)
        result = {
            "action": "roll_dice",
            "num_dice": num_dice,
            "sides": sides,
            "results": results,
            "total": total,
            "message": f"Rolled {results} (total: {total})" if num_dice > 1 else f"Rolled {results[0]}"
        }
        print(f"🎯 [FUNCTION] Dice results: {result['message']}")
        return result

    # ============= CUSTOM API CALL FUNCTIONS =============
    def get_weather(self, city="London", country_code="UK"):
        """Get current weather for a city using OpenWeatherMap API

        Args:
            city (str): City name
            country_code (str): Country code (e.g., "UK", "US")
        """
        print(f"🌤️ [FUNCTION] Getting weather for {city}, {country_code}")

        try:
            # Using OpenWeatherMap API (free tier available)
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": f"{city},{country_code}",
                "appid": self.weather_api_key,
                "units": "metric"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    "action": "get_weather",
                    "city": city,
                    "country": country_code,
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "message": f"Weather in {city}: {data['main']['temp']}°C, {data['weather'][0]['description']}"
                }
                print(f"✅ [FUNCTION] Weather retrieved: {weather_info['message']}")
                return weather_info
            else:
                error_msg = f"Could not get weather for {city}. Please check the city name."
                print(f"❌ [FUNCTION] Weather API error: {response.status_code}")
                return {"action": "get_weather", "error": error_msg, "message": error_msg}

        except Exception as e:
            error_msg = f"Weather service unavailable: {str(e)}"
            print(f"❌ [FUNCTION] Weather error: {e}")
            return {"action": "get_weather", "error": error_msg, "message": error_msg}

    def get_latest_news(self, category="general", country="us", num_articles=3):
        """Get latest news headlines using NewsAPI

        Args:
            category (str): News category (general, business, technology, etc.)
            country (str): Country code for news
            num_articles (int): Number of articles to retrieve
        """
        print(f"📰 [FUNCTION] Getting {num_articles} latest {category} news from {country}")

        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": category,
                "country": country,
                "pageSize": num_articles,
                "apiKey": self.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                articles = []

                for article in data.get("articles", [])[:num_articles]:
                    articles.append({
                        "title": article["title"],
                        "description": article.get("description", "No description"),
                        "source": article["source"]["name"]
                    })

                result = {
                    "action": "get_latest_news",
                    "category": category,
                    "country": country,
                    "articles": articles,
                    "message": f"Found {len(articles)} latest {category} news articles"
                }

                print(f"✅ [FUNCTION] News retrieved: {len(articles)} articles")
                return result
            else:
                error_msg = "News service unavailable at the moment"
                print(f"❌ [FUNCTION] News API error: {response.status_code}")
                return {"action": "get_latest_news", "error": error_msg, "message": error_msg}

        except Exception as e:
            error_msg = f"Could not retrieve news: {str(e)}"
            print(f"❌ [FUNCTION] News error: {e}")
            return {"action": "get_latest_news", "error": error_msg, "message": error_msg}

    def search_web(self, query, num_results=3):
        """Search the web using a search API (example with DuckDuckGo)

        Args:
            query (str): Search query
            num_results (int): Number of results to return
        """
        print(f"🔍 [FUNCTION] Searching web for: '{query}'")

        try:
            # Using DuckDuckGo Instant Answer API (free, no API key needed)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Get abstract or definition
                abstract = data.get("Abstract", "")
                definition = data.get("Definition", "")
                answer = data.get("Answer", "")

                search_result = abstract or definition or answer or "No direct answer found"

                result = {
                    "action": "search_web",
                    "query": query,
                    "result": search_result,
                    "source": data.get("AbstractSource", "DuckDuckGo"),
                    "message": f"Search result for '{query}': {search_result[:200]}..."
                }

                print(f"✅ [FUNCTION] Web search completed")
                return result
            else:
                error_msg = f"Could not search for '{query}'"
                return {"action": "search_web", "error": error_msg, "message": error_msg}

        except Exception as e:
            error_msg = f"Search unavailable: {str(e)}"
            print(f"❌ [FUNCTION] Search error: {e}")
            return {"action": "search_web", "error": error_msg, "message": error_msg}

    # ============= DATABASE FUNCTIONS =============
    def save_conversation_log(self, user_message, ai_response, functions_called=""):
        """Save conversation to database

        Args:
            user_message (str): What the user said
            ai_response (str): AI response
            functions_called (str): Functions that were called
        """
        print("💾 [FUNCTION] Saving conversation to database")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO call_logs (timestamp, user_message, ai_response, functions_called)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                user_message,
                ai_response,
                functions_called
            ))

            conn.commit()
            log_id = cursor.lastrowid
            conn.close()

            result = {
                "action": "save_conversation_log",
                "log_id": log_id,
                "message": f"Conversation saved with ID {log_id}"
            }

            print(f"✅ [FUNCTION] Conversation saved with ID {log_id}")
            return result

        except Exception as e:
            error_msg = f"Could not save conversation: {str(e)}"
            print(f"❌ [FUNCTION] Database save error: {e}")
            return {"action": "save_conversation_log", "error": error_msg, "message": error_msg}

    def get_conversation_history(self, limit=5):
        """Get recent conversation history from database

        Args:
            limit (int): Number of recent conversations to retrieve
        """
        print(f"📜 [FUNCTION] Getting last {limit} conversations from database")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, user_message, ai_response, functions_called
                FROM call_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            conversations = []
            for row in rows:
                conversations.append({
                    "timestamp": row[0],
                    "user_message": row[1],
                    "ai_response": row[2],
                    "functions_called": row[3]
                })

            result = {
                "action": "get_conversation_history",
                "conversations": conversations,
                "count": len(conversations),
                "message": f"Retrieved {len(conversations)} recent conversations"
            }

            print(f"✅ [FUNCTION] Retrieved {len(conversations)} conversations")
            return result

        except Exception as e:
            error_msg = f"Could not retrieve history: {str(e)}"
            print(f"❌ [FUNCTION] Database read error: {e}")
            return {"action": "get_conversation_history", "error": error_msg, "message": error_msg}

    # ============= FILE OPERATIONS =============
    def save_note(self, note_content, filename=""):
        """Save a note to file

        Args:
            note_content (str): Content of the note
            filename (str): Optional filename, auto-generated if empty
        """
        print(f"📝 [FUNCTION] Saving note to file")

        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"note_{timestamp}.txt"

            # Ensure notes directory exists
            notes_dir = "notes"
            os.makedirs(notes_dir, exist_ok=True)

            filepath = os.path.join(notes_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Note created: {datetime.now().isoformat()}\n")
                f.write(f"Content:\n{note_content}")

            result = {
                "action": "save_note",
                "filename": filename,
                "filepath": filepath,
                "content_length": len(note_content),
                "message": f"Note saved as {filename}"
            }

            print(f"✅ [FUNCTION] Note saved: {filename}")
            return result

        except Exception as e:
            error_msg = f"Could not save note: {str(e)}"
            print(f"❌ [FUNCTION] File save error: {e}")
            return {"action": "save_note", "error": error_msg, "message": error_msg}

    def read_note(self, filename):
        """Read a note from file

        Args:
            filename (str): Name of the file to read
        """
        print(f"📖 [FUNCTION] Reading note from file: {filename}")

        try:
            notes_dir = "notes"
            filepath = os.path.join(notes_dir, filename)

            if not os.path.exists(filepath):
                error_msg = f"Note '{filename}' not found"
                return {"action": "read_note", "error": error_msg, "message": error_msg}

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            result = {
                "action": "read_note",
                "filename": filename,
                "content": content,
                "content_length": len(content),
                "message": f"Read note {filename} ({len(content)} characters)"
            }

            print(f"✅ [FUNCTION] Note read: {filename}")
            return result

        except Exception as e:
            error_msg = f"Could not read note: {str(e)}"
            print(f"❌ [FUNCTION] File read error: {e}")
            return {"action": "read_note", "error": error_msg, "message": error_msg}

    # ============= SYSTEM OPERATIONS =============
    def get_system_info(self):
        """Get system information"""
        print("💻 [FUNCTION] Getting system information")

        try:
            import platform
            import psutil

            result = {
                "action": "get_system_info",
                "system": platform.system(),
                "release": platform.release(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
                "disk_usage": round(psutil.disk_usage('/').percent, 2),
                "message": f"System: {platform.system()} {platform.release()}, RAM: {round(psutil.virtual_memory().total / (1024**3), 2)}GB"
            }

            print(f"✅ [FUNCTION] System info retrieved")
            return result

        except Exception as e:
            error_msg = f"Could not get system info: {str(e)}"
            print(f"❌ [FUNCTION] System info error: {e}")
            return {"action": "get_system_info", "error": error_msg, "message": error_msg}

    def calculate(self, expression):
        """Safely calculate mathematical expressions

        Args:
            expression (str): Mathematical expression to calculate
        """
        print(f"🧮 [FUNCTION] Calculating: {expression}")

        try:
            # Safe evaluation - only allow basic math operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})

            # Remove any potentially dangerous operations
            dangerous = ["import", "exec", "eval", "open", "__"]
            if any(danger in expression.lower() for danger in dangerous):
                error_msg = "Expression contains potentially unsafe operations"
                return {"action": "calculate", "error": error_msg, "message": error_msg}

            result_value = eval(expression, {"__builtins__": {}}, allowed_names)

            result = {
                "action": "calculate",
                "expression": expression,
                "result": result_value,
                "message": f"{expression} = {result_value}"
            }

            print(f"✅ [FUNCTION] Calculation result: {result_value}")
            return result

        except Exception as e:
            error_msg = f"Could not calculate '{expression}': {str(e)}"
            print(f"❌ [FUNCTION] Calculation error: {e}")
            return {"action": "calculate", "error": error_msg, "message": error_msg}

    # ============= FUNCTION DEFINITIONS FOR AI =============
    def get_available_functions(self):
        """Get list of all available functions for AI"""
        return [
            # Basic functions
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the current phone call. Use when user says goodbye or wants to hang up.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for ending the call"}
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "roll_dice",
                    "description": "Roll dice for games or random decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "num_dice": {"type": "integer", "description": "Number of dice", "default": 1},
                            "sides": {"type": "integer", "description": "Sides per die", "default": 6}
                        }
                    }
                }
            },
            # API call functions
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for any city. Use when user asks about weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "country_code": {"type": "string", "description": "Country code like US, UK", "default": "US"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_latest_news",
                    "description": "Get latest news headlines. Use when user asks about news or current events.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": ["general", "business", "technology", "sports", "entertainment"], "default": "general"},
                            "country": {"type": "string", "description": "Country code", "default": "us"},
                            "num_articles": {"type": "integer", "description": "Number of articles", "default": 3}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information. Use when user asks questions that need web search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "description": "Number of results", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            # Database functions
            {
                "type": "function",
                "function": {
                    "name": "save_conversation_log",
                    "description": "Save important conversations to database. Use when user asks to remember something.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_message": {"type": "string", "description": "User's message"},
                            "ai_response": {"type": "string", "description": "AI response"},
                            "functions_called": {"type": "string", "description": "Functions used", "default": ""}
                        },
                        "required": ["user_message", "ai_response"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_conversation_history",
                    "description": "Get previous conversation history. Use when user asks about past conversations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of conversations", "default": 5}
                        }
                    }
                }
            },
            # File operations
            {
                "type": "function",
                "function": {
                    "name": "save_note",
                    "description": "Save a note to file. Use when user wants to save information or create notes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note_content": {"type": "string", "description": "Content of the note"},
                            "filename": {"type": "string", "description": "Optional filename", "default": ""}
                        },
                        "required": ["note_content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_note",
                    "description": "Read a previously saved note. Use when user asks to read or retrieve saved notes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the file to read"}
                        },
                        "required": ["filename"]
                    }
                }
            },
            # System functions
            {
                "type": "function",
                "function": {
                    "name": "get_system_info",
                    "description": "Get system information. Use when user asks about server or system status.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate mathematical expressions. Use when user needs math calculations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to calculate"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def execute_function(self, function_name, arguments):
        """Execute a function by name with given arguments"""
        try:
            function_map = {
                # Basic functions
                "end_call": self.end_call,
                "roll_dice": self.roll_dice,
                # API functions
                "get_weather": self.get_weather,
                "get_latest_news": self.get_latest_news,
                "search_web": self.search_web,
                # Database functions
                "save_conversation_log": self.save_conversation_log,
                "get_conversation_history": self.get_conversation_history,
                # File functions
                "save_note": self.save_note,
                "read_note": self.read_note,
                # System functions
                "get_system_info": self.get_system_info,
                "calculate": self.calculate,
            }

            if function_name in function_map:
                return function_map[function_name](**arguments)
            else:
                return {"error": f"Function {function_name} not found"}

        except Exception as e:
            print(f"❌ [FUNCTION] Error executing {function_name}: {e}")
            return {"error": f"Error executing {function_name}: {str(e)}"}


# Add the math import at the top
import math

print("✅ Custom Function Library with API calls created!")
print("\n🌟 Enhanced Functions Available:")
print("\n📡 API FUNCTIONS:")
print("- get_weather(city, country_code): Get weather from OpenWeatherMap")
print("- get_latest_news(category, country, num_articles): Get news from NewsAPI")
print("- search_web(query, num_results): Web search with DuckDuckGo")
print("\n💾 DATABASE FUNCTIONS:")
print("- save_conversation_log(user_msg, ai_response, functions): Save to SQLite")
print("- get_conversation_history(limit): Retrieve conversation history")
print("\n📁 FILE FUNCTIONS:")
print("- save_note(content, filename): Save notes to files")
print("- read_note(filename): Read saved notes")
print("\n💻 SYSTEM FUNCTIONS:")
print("- get_system_info(): Get server system information")
print("- calculate(expression): Safe mathematical calculations")
print("\n🔧 SETUP REQUIRED:")
print("1. Get API keys:")
print("   - Weather: https://openweathermap.org/api")
print("   - News: https://newsapi.org/")
print("2. Add to .env file:")
print("   WEATHER_API_KEY=your_weather_api_key")
print("   NEWS_API_KEY=your_news_api_key")
print("3. Install dependencies:")
print("   pip install requests psutil aiohttp aiofiles")
