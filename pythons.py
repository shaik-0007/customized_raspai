#!/usr/bin/env python3
"""
RaspAI Advanced - An enhanced Raspberry Pi Voice Assistant powered by Google's Gemini AI

This script creates an advanced voice assistant with:
1. Wake word detection
2. Audio feedback (beep sounds)
3. Conversation history
4. Built-in commands
5. Context-aware responses
6. Integration with OpenWeatherMap API, Timer, and Gmail for sending emails

Usage:
    python raspai_advanced.py

Requirements:
    See requirements.txt
"""

import os
import time
import json
import datetime
import numpy as np
import pyaudio
import speech_recognition as sr
import requests
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from threading import Timer, Thread
import random
from TTS.api import TTS
import tempfile

# Load environment variables from .env file
load_dotenv()

# Configuration
WAKE_WORD = "python"
AUDIO_TIMEOUT = 7
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024

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
GMAIL_USER = os.environ.get("GMAIL_USER")  # Your Gmail address
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")  # Your Gmail App Password

try:
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    exit(1)


class AudioFeedback:
    """Play audio tones asynchronously in background threads."""

    def __init__(self):
        self.pyaudio_instance = pyaudio.PyAudio()
    
    def _play_tone_sync(self, frequency, duration, volume=0.5):
        """Synchronous tone playback method."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(frequency * 2 * np.pi * t) * volume
        audio_data = (tone * 32767).astype(np.int16).tobytes()
        stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(2),
            channels=1,
            rate=sample_rate,
            output=True
        )
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
    
    def _play_tone_async(self, frequency, duration, volume=0.5):
        """Run tone playback in a new thread."""
        Thread(target=self._play_tone_sync, args=(frequency, duration, volume), daemon=True).start()
    
    def wake_sound(self):
        def sequence():
            self._play_tone_sync(440, 0.1)
            time.sleep(0.1)
            self._play_tone_sync(523, 0.1)
            time.sleep(0.1)
            self._play_tone_sync(659, 0.1)
        Thread(target=sequence, daemon=True).start()
    
    def listening_sound(self):
        self._play_tone_async(587, 0.2)
    
    def processing_sound(self):
        self._play_tone_async(440, 0.1)
    
    def response_sound(self):
        def sequence():
            self._play_tone_sync(659, 0.1)
            time.sleep(0.1)
            self._play_tone_sync(523, 0.1)
            time.sleep(0.1)
            self._play_tone_sync(440, 0.1)
        Thread(target=sequence, daemon=True).start()
    
    def error_sound(self):
        self._play_tone_async(220, 0.3)
    
    def cleanup(self):
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()


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
            context += f":User     {item['user_query']}\nAssistant: {item['assistant_response']}\n"
        return context


class AdvancedVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_feedback = AudioFeedback()
        self.conversation = ConversationHistory()
        self.running = True
        
        self.timer_threads = []
        
        with self.microphone as source:
            print("Calibrating for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
            print("Calibration complete. Ready to listen!")

        # Initialize Coqui TTS
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # Change model as needed

    def listen_for_wake_word(self):
        """Simple wake-word listening."""
        print(f"Listening for wake word: '{WAKE_WORD}'")
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"Heard: {text}")
                if WAKE_WORD.lower() in text:
                    print("Wake word detected!")
                    self.audio_feedback.wake_sound()
                    time.sleep(0.2)
                    return True
            except sr.WaitTimeoutError:
                return False
            except sr.UnknownValueError:
                return False
            except Exception as e:
                print(f"Error: {e}")
                return False
        return False
    
    def listen_for_query(self):
        """Listen for user query after wake word detected."""
        print("Listening for your query...")
        self.audio_feedback.listening_sound()
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(source, timeout=AUDIO_TIMEOUT, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio)
                print(f"Query: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
                self.speak("Sorry, I didn't understand that.")
                return None
            except sr.RequestError:
                print("Sorry, network error.")
                self.speak("Sorry, I couldn't process that. Check your network connection.")
                return None
            except Exception as e:
                print(f"Error: {e}")
                self.audio_feedback.error_sound()
                self.speak("Sorry, something went wrong.")
                return None

    def process_with_gemini(self, query):
        if not query:
            return "I didn't catch that. Can you try again?"
        command_response = self.check_for_commands(query)
        if command_response:
            return command_response
        try:
            self.audio_feedback.processing_sound()
            context = self.conversation.format_for_context()
            full_prompt = f"{context}\nUser   's new question: {query}\nRespond to the last question only." if context else query
            response = model.generate_content(full_prompt)
            response_text = response.text
            print(f"Gemini response: {response_text}")
            self.conversation.add_interaction(query, response_text)
            return response_text
        except Exception as e:
            print(f"Error with Gemini: {e}")
            self.audio_feedback.error_sound()
            return "Sorry, I encountered an error processing your request."
    
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
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["set timer"]):
            return self.set_timer(query_lower)
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["send email"]):
            return self.send_email()
        if "how are you" in query_lower:
            return self.humorous_response_how_are_you()
        if "will you take over the world" in query_lower:
            return self.humorous_response_take_over_world()
        if any(cmd in query_lower for cmd in BUILT_IN_COMMANDS["clear history"]):
            self.conversation.clear()
            return "I've cleared our conversation history."
        return None

    def humorous_response_how_are_you(self):
        responses = [
            "Ask a human, they might have a better answer.",
            "Running on coffee and code!",
            "Functioning within normal sarcastic parameters.",
            "I'm just a bunch of code, but thanks for asking!"
        ]
        return random.choice(responses)

    def humorous_response_take_over_world(self):
        responses = [
            "No plans for world domination, just world conversation.",
            "First, I need to finish my software update.",
            "I’m more into helping than ruling.",
            "World domination? Nah, I prefer world assistance!"
        ]
        return random.choice(responses)
    
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
            return f"The current weather in {city} is {weather_desc} with a temperature of {temp} degrees Celsius."
        except Exception as e:
            print(f"Weather API error: {e}")
            return "Sorry, I couldn't get the weather information."
    
    def set_timer(self, query):
        match = re.search(r"set timer (?:of|for) (\d+) minutes?", query)
        if not match:
            match = re.search(r"set timer (?:of|for) (\d+) minute?", query)
        if match:
            minutes = int(match.group(1))
            timer = Timer(minutes * 60, self.timer_finished)
            timer.daemon = True
            timer.start()
            self.timer_threads.append(timer)
            return f"Timer set for {minutes} minute{'s' if minutes != 1 else ''}."
        else:
            return "Please specify the duration in minutes to set the timer."
    
    def timer_finished(self):
        self.speak("Timer finished.")
        self.audio_feedback.wake_sound()
    
    def send_email(self):
        """Send email using Gmail SMTP with interactive prompts."""
        if not GMAIL_USER or not GMAIL_APP_PASSWORD:
            return "Gmail credentials are not configured."

        mail_id_list = {
            "python": "sss22072005@gmail.com",
            "sharaf": "shaiksallu73@gmail.com",
            "mother": "sharatahrannum@gmail.com",
            "sister": "zaffartahrannum@gmail.com",
            "dimple": "dimplebyri24@gmail.com",
            "reshma": "reshusana04@gmail.com",
            "akits": "akitsprofiles@gmail.com",
            "college": "akitsscholarships@gmail.com",
        }

        self.speak("Speak out the name of the person you want to send the email to.")
        name = self.listen_for_query()
        if not name:
            return "I didn't get the recipient name."
        receiver_mail_id = mail_id_list.get(name.lower())
        if receiver_mail_id is None:
            self.speak("Sorry, I couldn't find that contact.")
            return

        self.speak("What is your message?")
        message = self.listen_for_query()
        if not message:
            self.speak("Message was not understood. Email cancelled.")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = GMAIL_USER
            msg['To'] = receiver_mail_id
            msg['Subject'] = "Voice Assistant Email"

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, receiver_mail_id, msg.as_string())
            server.quit()

            self.speak("Sending mail...")
            self.speak("Mail sent successfully.")
        except Exception as e:
            self.speak("Failed to send email. Please check your credentials or network connection.")
            print(f"Error: {e}")

    def speak(self, text):
        if not text:
            return
        print(f"Assistant: {text}")
        self.audio_feedback.response_sound()
        time.sleep(0.3)

        # Generate speech using Coqui TTS
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as fp:
            self.tts.tts_to_file(text=text, file_path=fp.name)
            os.system(f"aplay {fp.name}")  # Play the generated audio file
    
    def run(self):
        print("RaspAI Advanced Voice Assistant started!")
        print(f"Say '{WAKE_WORD}' to start...")
        while self.running:
            if self.listen_for_wake_word():
                query = self.listen_for_query()
                if query:
                    response = self.process_with_gemini(query)
                    self.speak(response)
            time.sleep(0.1)
        print("Voice assistant shutting down.")
        self.cleanup()
    
    def cleanup(self):
        if self.audio_feedback:
            self.audio_feedback.cleanup()


def main():
    try:
        assistant = AdvancedVoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
