#!/usr/bin/env python3
"""
RaspAI Advanced - Raspberry Pi Voice Assistant with always-on wake word (polling, no threads)

This version:
- Polls microphone input in a loop for wake word
- After wake word, listens for query
- Speaks responses asynchronously (non-blocking speak)
- No threading, no concurrency issues
"""

import os
import time
import json
import datetime
import numpy as np
import pyaudio
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import requests
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import random

load_dotenv()

WAKE_WORD = "python"
AUDIO_TIMEOUT = 5
MAX_HISTORY_LENGTH = 10
HISTORY_FILE = "conversation_history.json"

BUILT_IN_COMMANDS = {
    "stop": ["stop", "exit", "quit", "bye", "goodbye"],
    "time": ["what time is it", "tell me the time", "current time"],
    "date": ["what day is it", "tell me the date", "current date", "what's today's date"],
    "weather": ["weather", "forecast", "climate"],
    "set timer": ["set timer of", "set timer for"],
    "send email": ["send email", "send mail", "send gmail", "sent mail", "sent email", "sent gmail"],
    "clear history": ["clear history", "forget conversation", "new conversation"]
}

OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

try:
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    exit(1)


class ConversationHistory:
    def __init__(self, history_file=HISTORY_FILE, max_length=MAX_HISTORY_LENGTH):
        self.history_file = history_file
        self.max_length = max_length
        self.history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                if len(self.history) > self.max_length:
                    self.history = self.history[-self.max_length:]
                print(f"Loaded {len(self.history)} conversation turns from history.")
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            self.history = []
    
    def save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def add_interaction(self, user_query, assistant_response):
        timestamp = datetime.datetime.now().isoformat()
        interaction = {"timestamp": timestamp,"user_query":user_query,"assistant_response":assistant_response}
        self.history.append(interaction)
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]
        self.save_history()
    
    def clear(self):
        self.history = []
        self.save_history()
        print("Conversation history cleared.")
    
    def get_recent_history(self, num_turns=3):
        return self.history[-num_turns:] if len(self.history)>0 else []
    
    def format_for_context(self, num_turns=3):
        recent = self.get_recent_history(num_turns)
        if not recent:
            return ""
        context = "Previous conversation:\n"
        for item in recent:
            context += f":User    {item['user_query']}\nAssistant: {item['assistant_response']}\n"
        return context


class AdvancedVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        self.conversation = ConversationHistory()
        self.running = True
        self.speaking = False
        
        with self.microphone as source:
            print("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
            print("Calibration done, ready to listen.")
    
    def speak(self, text):
        if not text:
            return
        print(f"Assistant: {text}")
        self.speaking = True
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        self.speaking = False
    
    def listen(self, timeout=AUDIO_TIMEOUT, phrase_time_limit=5):
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = self.recognizer.recognize_google(audio)
                return text.lower()
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except Exception as e:
                print(f"Listen error: {e}")
                return None

    def check_for_commands(self, query):
        query_lower = query.lower()
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["stop"]):
            self.running = False
            return "Goodbye! Shutting down."
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["time"]):
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["date"]):
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}."
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["weather"]):
            return self.get_weather()
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["clear history"]):
            self.conversation.clear()
            return "I've cleared our conversation history."
        # Extend commands as needed
        return None

    def get_weather(self):
        if not OPENWEATHERMAP_API_KEY:
            return "OpenWeatherMap API key not configured."
        city = "London"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        try:
            response = requests.get(url)
            data = response.json()
            if data.get("cod") != 200:
                return "I couldn't fetch the weather information right now."
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C."
        except Exception as e:
            print(f"Weather API error: {e}")
            return "Sorry, I couldn't get the weather information."

    def process_with_gemini(self, query):
        if not query:
            return "I didn't catch that. Can you try again?"
        cmd_response = self.check_for_commands(query)
        if cmd_response:
            return cmd_response
        try:
            context = self.conversation.format_for_context()
            prompt = f"{context}\nUser's question: {query}\nRespond only to the last question." if context else query
            response = model.generate_content(prompt)
            response_text = response.text
            self.conversation.add_interaction(query, response_text)
            return response_text
        except Exception as e:
            print(f"Gemini error: {e}")
            return "Sorry, I encountered an error processing your request."

    def run(self):
        print("Assistant started. Say the wake word to begin.")
        while self.running:
            if self.speaking:
                # While assistant is speaking, just wait shortly and keep looping (no blocking)
                time.sleep(0.1)
                continue

            print("Listening for wake word...")
            text = self.listen(timeout=3, phrase_time_limit=3)
            if text and WAKE_WORD in text:
                print("Wake word detected!")
                self.speak("Yes?")
                print("Listening for your query...")
                query = self.listen(timeout=7, phrase_time_limit=15)
                if query:
                    print(f"User query: {query}")
                    response = self.process_with_gemini(query)
                    self.speak(response)
                else:
                    self.speak("Sorry, I didn't hear anything.")
            else:
                # no wake word, just loop and listen again
                pass
            time.sleep(0.1)  # small delay to reduce CPU usage

        print("Assistant shutting down.")


def main():
    try:
        assistant = AdvancedVoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
