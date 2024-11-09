# chatbot.py

import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_qa_dataset
from utils.intent_processor import (
    initialize_intents,
    get_intent,
    extract_name,
    extract_location,
    extract_time_location,
)
from utils.weather_service import get_weather, get_time_in_location
from datetime import datetime

class Chatbot:
    def __init__(self):
        self.user_name = None
        self.conversation_history = []
        self.qa_pairs, self.questions, self.qa_vectors, self.qa_vectorizer = load_qa_dataset()
        (
            self.intents,
            self.intent_mapping,
            self.intent_vectors,
            self.intent_vectorizer,
            self.capabilities_response,
            self.small_talk_responses,
        ) = initialize_intents()

    def get_welcome_message(self):
        """Return the welcome message."""
        return "Hello! I'm your ChatBot. What's your name?"

    def handle_time_query(self):
        """Handle time-related queries."""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"

    def handle_date_query(self):
        """Handle date-related queries."""
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"

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
            extracted_name = extract_name(user_input)
            if extracted_name:
                self.user_name = extracted_name
                return f"Nice to meet you, {self.user_name}! How can I help you today?"
            else:
                return "I didn't quite catch your name. Could you tell me again?"

        # Check if input is a question
        is_question = user_input.strip().endswith('?') or user_input.strip().lower().startswith(
            ('what', 'how', 'why', 'where', 'when', 'who')
        )

        if is_question:
            # Try to find a QA match first
            best_qa_match = self.find_best_qa_match(user_input)
            if best_qa_match:
                return self.qa_pairs[best_qa_match]

        # Proceed with intent recognition
        intent, score = get_intent(user_input, self.intent_vectorizer, self.intent_vectors, self.intent_mapping)

        if intent and score >= 0.7:
            if intent == 'greeting':
                return f"Hello {self.user_name}! How can I help you today?"
            elif intent == 'farewell':
                return f"Goodbye {self.user_name}! Have a great day!"
            elif intent == 'capabilities':
                return self.capabilities_response[0]
            elif intent == 'time_query':
                # Handle time in a specific location
                location = extract_time_location(user_input)
                if location:
                    return get_time_in_location(location)
                else:
                    return self.handle_time_query()
            elif intent == 'date_query':
                return self.handle_date_query()
            elif intent == 'name_query':
                return f"Your name is {self.user_name}!"
            elif intent == 'weather_query':
                location = extract_location(user_input)
                return get_weather(location)
            elif intent == 'small_talk':
                return random.choice(self.small_talk_responses)
        else:
            # Try QA matching if intent confidence is low
            best_qa_match = self.find_best_qa_match(user_input)
            if best_qa_match:
                return self.qa_pairs[best_qa_match]

            return "I'm not sure how to answer that. Could you rephrase your question?"
