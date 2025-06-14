import re
import random
import csv
import os
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import get_close_matches
import nltk
import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pytz

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class Chatbot:
    def __init__(self):
        self.user_name = None
        self.conversation_history = []
        self.load_qa_dataset()
        self.initialize_intents()

        # Separate vectorizers for intents and QA
        self.intent_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer='char_wb',
            max_features=5000,
            lowercase=True,
            strip_accents='unicode'
        )

        # Specific vectorizer for QA matching
        self.qa_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer='word',
            max_features=10000,
            lowercase=True,
            stop_words='english'
        )

        self.intent_vectors = None
        self.vectorize_intents()
        self.vectorize_qa()

        # Small talk responses
        self.small_talk_responses = [
            "I'm doing well, thank you!",
            "I'm fine, thanks for asking!",
            "All good here!",
            "I'm just a bot, but I'm doing great!"
        ]

    def vectorize_qa(self):
        """Create TF-IDF vectors for QA dataset."""
        if self.qa_pairs:
            self.questions = list(self.qa_pairs.keys())
            try:
                self.qa_vectors = self.qa_vectorizer.fit_transform(self.questions)
            except Exception as e:
                print(f"Error in QA vectorization: {str(e)}")

    def get_welcome_message(self):
        """Return the welcome message."""
        return "Hello! I'm your ChatBot. What's your name?"

    def load_qa_dataset(self):
        """Load the QA dataset."""
        self.qa_pairs = {}
        try:
            with open('COMP3074-CW1-Dataset.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.qa_pairs[row['Question'].lower()] = row['Answer']
        except Exception as e:
            print(f"Error loading QA dataset: {str(e)}")
            self.qa_pairs = {}

    def initialize_intents(self):
        """Initialize intents with example phrases."""
        self.intents = {
            'greeting': [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon',
                'good evening', 'howdy', 'hi there', 'greetings', "what's up"
            ],
            'farewell': [
                'bye', 'goodbye!', 'see you', 'farewell', 'quit', 'exit',
                'see you later', 'have a good day', 'take care', 'bye bye'
            ],
            'capabilities': [
                'what can you do', 'help',
                'what do you do', 'show me what you can do'
            ],
            'time_query': [
                'what time is it', 'what is the time', 'current time',
                'time', 'tell me the time', 'time now', 'time in'
            ],
            'date_query': [
                'what date is it', 'what is the date', 'current date',
                'date', 'tell me the date', 'what day is it'
            ],
            'name_query': [
                'what is my name', 'who am i', 'my name', 'tell me my name',
                'do you know my name',
            ],
            'weather_query': [
                'what is the weather', 'weather', 'temperature',
                'how is the weather', 'what is the temperature', 'weather in'
            ],
            'small_talk': [
                'how are you', "what's up", 'what is up', "how's it going",
                'how are you doing', 'how do you do', "what's new"
            ]
        }

        self.capabilities_response = [
            "I can help with answering questions from QA dataset, providing the current time and date, checking the weather for any city, and remembering your name."
        ]

    def vectorize_intents(self):
        """Create TF-IDF vectors for intent phrases."""
        all_phrases = []
        self.intent_mapping = []

        for intent, phrases in self.intents.items():
            intent_doc = ' '.join(phrases)
            all_phrases.append(intent_doc)
            self.intent_mapping.append(intent)

        self.intent_vectors = self.intent_vectorizer.fit_transform(all_phrases)

    def get_coordinates(self, city):
        """Get coordinates for a given city using Open-Meteo Geocoding API."""
        try:
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    return result['latitude'], result['longitude'], result['name'], result.get('country', '')
            return None
        except Exception:
            return None

    def get_weather(self, location=None):
        """Fetch current weather data for a specific location."""
        if not location:
            return "Please specify a location. For example: 'What's the weather in London?'"

        coordinates = self.get_coordinates(location)
        if not coordinates:
            return f"I couldn't find the location '{location}'. Please try another city."

        lat, lon, city, country = coordinates
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&temperature_unit=celsius"
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                current_weather = data.get('current_weather', {})
                temperature = current_weather.get('temperature')
                windspeed = current_weather.get('windspeed')

                if temperature is not None:
                    weather_description = "warm" if temperature > 20 else "mild" if temperature > 10 else "cold"
                    location_name = f"{city}, {country}" if country else city
                    return f"The current temperature in {location_name} is {temperature}°C ({weather_description}) with a wind speed of {windspeed} km/h."

            return "I'm sorry, I couldn't fetch the weather information at the moment."
        except Exception:
            return "Sorry, there was an error getting the weather data."

    def extract_location(self, user_input):
        """Extract location from weather-related queries."""
        patterns = [
            r"weather (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"weather of (.+?)(?:\?|$| please| now| today)",
            r"temperature (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"how'?s the weather (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"what'?s the weather (?:in|at|for) (.+?)(?:\?|$| please| now| today)"
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def extract_time_location(self, user_input):
        """Extract location from time-related queries."""
        patterns = [
            r"time (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"what'?s the time (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"current time (?:in|at|for) (.+?)(?:\?|$| please| now| today)",
            r"time of (.+?)(?:\?|$| please| now| today)",
            r"tell me the time (?:in|at|for) (.+?)(?:\?|$| please| now| today)"
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def get_time_in_location(self, location):
        """Get current time in the specified location."""
        try:
            geolocator = Nominatim(user_agent="chatbot")
            geocode_result = geolocator.geocode(location)
            if geocode_result:
                timezone_finder = TimezoneFinder()
                timezone = timezone_finder.timezone_at(
                    lng=geocode_result.longitude, lat=geocode_result.latitude)
                if timezone:
                    tz = pytz.timezone(timezone)
                    current_time = datetime.now(tz).strftime("%I:%M %p")
                    return f"The current time in {location} is {current_time}"
            return f"Sorry, I couldn't find the time for {location}."
        except Exception as e:
            print(f"Error in get_time_in_location: {str(e)}")
            return "Sorry, there was an error getting the time."

    def extract_name(self, user_input):
        """Extract name from user input."""
        patterns = [
            r"(?:my name is|i am|i'm|call me) ([A-Za-z\s]+)",
            r"([A-Za-z\s]+) is my name",
            r"^([A-Za-z]+)$",
            r"name: ([A-Za-z\s]+)",
            r"name is ([A-Za-z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                return name.title()
        return None

    def handle_time_query(self):
        """Handle time-related queries."""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def handle_date_query(self):
        """Handle date-related queries."""
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"

    def get_intent(self, user_input, threshold=0.4):
        """Determine intent using similarity matching."""
        user_input = user_input.lower().strip()

        try:
            user_vector = self.intent_vectorizer.transform([user_input])
            similarities = cosine_similarity(user_vector, self.intent_vectors)[0]

            best_match_index = np.argmax(similarities)
            best_match_score = similarities[best_match_index]

            if best_match_score >= threshold:
                return self.intent_mapping[best_match_index], best_match_score

        except Exception as e:
            print(f"Error in intent matching: {str(e)}")

        return None, 0.0

    def find_best_qa_match(self, user_input):
        """Find best matching question in QA dataset using similarity."""
        if not self.qa_pairs:
            return None

        try:
            input_vector = self.qa_vectorizer.transform([user_input])

            similarities = cosine_similarity(input_vector, self.qa_vectors)[0]
            best_match_index = np.argmax(similarities)
            best_match_score = similarities[best_match_index]

            if best_match_score >= 0.3:
                return self.questions[best_match_index]

        except Exception as e:
            print(f"Error in QA matching: {str(e)}")

        return None

    def handle_user_input(self, user_input):
        """Process user input and generate response."""
        if not self.user_name:
            extracted_name = self.extract_name(user_input)
            if extracted_name:
                self.user_name = extracted_name
                return f"Nice to meet you, {self.user_name}! How can I help you today?"
            else:
                return "I didn't quite catch your name. Could you tell me again?"

        # Check if input is a question
        is_question = user_input.strip().endswith('?') or user_input.strip().lower().startswith(
            ('what', 'how', 'why', 'where', 'when', 'who'))

        if is_question:
            # Try to find a QA match first
            best_qa_match = self.find_best_qa_match(user_input)
            if best_qa_match:
                return self.qa_pairs[best_qa_match]

        # Proceed with intent recognition
        intent, score = self.get_intent(user_input)

        if intent and score >= 0.7:
            if intent == 'greeting':
                return f"Hello {self.user_name}! How can I help you today?"
            elif intent == 'farewell':
                return f"Goodbye {self.user_name}! Have a great day!"
            elif intent == 'capabilities':
                return self.capabilities_response[0]
            elif intent == 'time_query':
                # Handle time in a specific location
                location = self.extract_time_location(user_input)
                if location:
                    return self.get_time_in_location(location)
                else:
                    return self.handle_time_query()
            elif intent == 'date_query':
                return self.handle_date_query()
            elif intent == 'name_query':
                return f"Your name is {self.user_name}!"
            elif intent == 'weather_query':
                location = self.extract_location(user_input)
                return self.get_weather(location)
            elif intent == 'small_talk':
                return random.choice(self.small_talk_responses)
        else:
            # Try QA matching if intent confidence is low
            best_qa_match = self.find_best_qa_match(user_input)
            if best_qa_match:
                return self.qa_pairs[best_qa_match]

            return "I'm not sure how to answer that. Could you rephrase your question?"


def main():
    chatbot = Chatbot()
    print(chatbot.get_welcome_message())

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break

        response = chatbot.handle_user_input(user_input)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()
