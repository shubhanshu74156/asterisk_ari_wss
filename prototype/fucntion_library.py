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
import statistics
from collections import deque


# Load environment
load_dotenv()
print("🌍 Environment Variables Loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🤖 OpenAI Client Initialized")

DEBUG = False  # FIXED: Set to False for production

HOST = "0.0.0.0"
PORT = 9092

# AudioSocket message types
MSG_HANGUP = 0x00
MSG_UUID = 0x01
MSG_DTMF = 0x03
MSG_AUDIO = 0x10
MSG_ERROR = 0xff


class FunctionLibrary:
    """Library of available functions that AI can call"""

    def __init__(self):
        print("🔧 Function Library initialized")

    def end_call(self, reason="User requested"):
        """End the current call"""
        print(f"📞 [FUNCTION] End call: {reason}")
        return {"action": "end_call", "reason": reason, "message": f"Call ended: {reason}"}

    def roll_dice(self, num_dice=1, sides=6):
        """Roll one or more dice"""
        print(f"🎲 [FUNCTION] Rolling {num_dice}d{sides}")
        results = [random.randint(1, sides) for _ in range(num_dice)]
        total = sum(results)
        return {
            "action": "roll_dice",
            "num_dice": num_dice,
            "sides": sides,
            "results": results,
            "total": total,
            "message": f"Rolled {results} (total: {total})" if num_dice > 1 else f"Rolled {results[0]}"
        }

    def pick_random_animal(self, category="all"):
        """Pick a random animal from a specified category"""
        animals = {
            "all": ["Lion", "Elephant", "Cheetah", "Dog", "Cat", "Pig", "Sparrow", "Eagle", "Snake", "Turtle", "Shark", "Dolphin"],
            "mammals": ["Lion", "Elephant", "Cheetah", "Dog", "Cat", "Pig", "Dolphin", "Bear", "Tiger"],
            "birds": ["Sparrow", "Eagle", "Parrot", "Penguin", "Owl", "Flamingo"],
            "reptiles": ["Snake", "Turtle", "Lizard", "Crocodile", "Iguana"],
            "sea": ["Shark", "Dolphin", "Whale", "Octopus", "Starfish"]
        }

        animal_list = animals.get(category.lower(), animals["all"])
        selected_animal = random.choice(animal_list)

        return {
            "action": "pick_random_animal",
            "category": category,
            "selected_animal": selected_animal,
            "message": f"Selected {selected_animal} from {category} animals"
        }

    def get_random_number(self, min_value=1, max_value=100):
        """Generate a random number within a specified range"""
        number = random.randint(min_value, max_value)
        return {
            "action": "get_random_number",
            "min_value": min_value,
            "max_value": max_value,
            "number": number,
            "message": f"Generated random number: {number}"
        }

    def flip_coin(self, num_flips=1):
        """Flip one or more coins"""
        results = [random.choice(["Heads", "Tails"]) for _ in range(num_flips)]
        return {
            "action": "flip_coin",
            "num_flips": num_flips,
            "results": results,
            "message": f"Coin flip results: {', '.join(results)}" if num_flips > 1 else f"Coin flip: {results[0]}"
        }

    def get_available_functions(self):
        """Get list of all available functions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "End the current phone call. Use when user says goodbye, wants to hang up, or indicates they're done.",
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
                    "description": "Roll one or more dice. Use when user wants to roll dice, play dice games, or needs random numbers via dice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "num_dice": {"type": "integer", "description": "Number of dice to roll", "default": 1},
                            "sides": {"type": "integer", "description": "Number of sides on each die", "default": 6}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "pick_random_animal",
                    "description": "Pick a random animal from a category. Use when user wants to know about animals, play animal games, or needs a random animal name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Category of animal to pick from",
                                "enum": ["all", "mammals", "birds", "reptiles", "sea"],
                                "default": "all"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_random_number",
                    "description": "Generate a random number within a range. Use when user needs random numbers, lottery numbers, or number games.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_value": {"type": "integer", "description": "Minimum value", "default": 1},
                            "max_value": {"type": "integer", "description": "Maximum value", "default": 100}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "flip_coin",
                    "description": "Flip one or more coins to get heads or tails. Use when user wants to flip coins, make decisions, or play coin games.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "num_flips": {"type": "integer", "description": "Number of coins to flip", "default": 1}
                        }
                    }
                }
            }
        ]

    def execute_function(self, function_name, arguments):
        """Execute a function by name with given arguments"""
        try:
            function_map = {
                "end_call": self.end_call,
                "roll_dice": self.roll_dice,
                "pick_random_animal": self.pick_random_animal,
                "get_random_number": self.get_random_number,
                "flip_coin": self.flip_coin
            }

            if function_name in function_map:
                return function_map[function_name](**arguments)
            else:
                return {"error": f"Function {function_name} not found"}

        except Exception as e:
            print(f"❌ [FUNCTION] Error executing {function_name}: {e}")
            return {"error": f"Error executing {function_name}: {str(e)}"}