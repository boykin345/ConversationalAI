import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
import json
import re
import random
import csv
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import get_close_matches
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Assistant")
        self.root.geometry("600x800")
        
        # Configure style
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#f0f0f0")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, style="Custom.TFrame", padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=("Georgia", 12),
            background="#ffffff",
            foreground="#000000"
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create input frame
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X, pady=5)
        
        # Create input field
        self.input_field = ttk.Entry(
            self.input_frame,
            font=("Georgia", 12)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Create send button
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Bind enter key to send message
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
        # Initialize chatbot backend
        self.chatbot = Chatbot()
        
        # Display welcome message
        self.display_message("Chatbot: " + self.chatbot.get_welcome_message())

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            # Display user message
            self.display_message("You: " + message)
            
            # Get chatbot response
            response = self.chatbot.handle_user_input(message)
            
            # Display chatbot response
            self.display_message("Chatbot: " + response)
            
            # Clear input field
            self.input_field.delete(0, tk.END)

    def display_message(self, message):
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)

class Chatbot:
    def __init__(self):
        self.user_name = None
        self.conversation_history = []
        self.load_qa_dataset()
        self.initialize_intents()

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
            self.qa_pairs = {}  # Initialize empty if file not found

    def initialize_intents(self):
        """Initialize basic intents and responses."""
        self.intents = {
            'greeting': [
                'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                'good evening', 'howdy'
            ],
            'farewell': [
                'bye', 'goodbye', 'see you', 'farewell', 'quit', 'exit'
            ],
            'capabilities': [
                'what can you do', 'help', 'what are your features', 
                'what do you do', 'what are you capable of', 'what are your functions',
                'how can you help me', 'what help do you offer'
            ],
            'time_query': [
                'what time is it', 'what is the time', 'current time', 
                'tell me the time', "what's the time", 'time', 'current time'
            ],
            'date_query': [
                'what date is it', 'what is the date', 'current date', 
                'tell me the date', "what's today's date", 'today\'s date', 'date'
            ],
            'name_query': [
                'what is my name', 'who am i', 'my name', 'my name is',
            ],
            'small_talk': {
                'how_are_you': [
                    'how are you', 'how are you doing', 'how is it going',
                    'how do you do', 'whats up', "what's up"
                ],
                'feeling': [
                    'are you happy', 'are you sad', 'how do you feel',
                    'what is your mood'
                ],
                'age': [
                    'how old are you', 'what is your age', 'when were you created',
                    'what is your birth date'
                ],
                'hobby': [
                    'what do you like', 'what are your hobbies', 
                    'what do you do for fun'
                ],
                'weather': [
                    'how is the weather', 'is it nice outside',
                    'what is the weather like'
                ]
            }
        }

        self.capabilities_response = [
            f"""I can help you with several things:
1. Answer questions from my knowledge base
2. Have friendly conversations and small talk
3. Tell you the current time and date
4. Remember your name and personalize our chat

Just ask me anything, and I'll do my best to help!""",
            
            f"""Here's what I can do for you:
- Chat with you about various topics
- Answer questions from my database
- Keep track of time and date
- Remember who you are
- Engage in friendly conversation
- Provide information and assistance

Feel free to ask me anything!"""
        ]

        self.small_talk_responses = {
            'how_are_you': [
                "I'm doing great, thanks for asking! How are you?",
                "I'm wonderful! Thanks for asking. How's your day going?",
                "All systems running smoothly! How about you?"
            ],
            'feeling': [
                "I'm designed to be helpful and positive!",
                "I'm feeling productive and ready to help!",
                "I'm always happy to chat with you!"
            ],
            'age': [
                "I'm a relatively young AI, but I'm learning new things every day!",
                "I exist in the eternal now of computation!",
                "Age is just a number for AIs like me!"
            ],
            'hobby': [
                "I love learning new things and chatting with people like you!",
                "My favorite activity is helping people and having interesting conversations!",
                "I enjoy processing information and solving problems!"
            ],
            'weather': [
                "I don't experience weather myself, but I hope it's nice where you are!",
                "You'll have to tell me - I don't have a window to look out of!",
                "I'm always in a climate-controlled server, but how's the weather on your end?"
            ]
        }

    def get_welcome_message(self):
        """Return the welcome message."""
        return "Hello! I'm your ChatBot. What's your name?"

    def extract_name(self, user_input):
        """Extract name from user input."""
        patterns = [
            r"(?:my name is|i am|i'm|call me) ([A-Za-z\s]+)",
            r"([A-Za-z\s]+) is my name",
            r"^([A-Za-z]+)$"
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

    def find_best_match(self, user_input):
        """Find the best matching question in the dataset."""
        if not self.qa_pairs:
            return None
            
        user_input = user_input.lower()
        matches = get_close_matches(user_input, self.qa_pairs.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    def handle_small_talk(self, user_input):
        """Handle small talk interactions."""
        user_input_lower = user_input.lower()
        
        for category, phrases in self.intents['small_talk'].items():
            if isinstance(phrases, list) and any(phrase in user_input_lower for phrase in phrases):
                return random.choice(self.small_talk_responses[category])
            elif isinstance(phrases, dict):
                for subcategory, subphrases in phrases.items():
                    if any(phrase in user_input_lower for phrase in subphrases):
                        return random.choice(self.small_talk_responses[subcategory])
        
        return None

    def handle_user_input(self, user_input):
        """Process user input and generate appropriate response."""
        try:
            # Handle empty input
            if not user_input.strip():
                return "Please say something!"

            # Handle name capture if no name is set
            if not self.user_name:
                extracted_name = self.extract_name(user_input)
                if extracted_name:
                    self.user_name = extracted_name
                    return f"Nice to meet you, {self.user_name}! How can I help you today?"
                else:
                    return "I didn't quite catch your name. Could you tell me again?"

            # Convert input to lowercase for intent matching
            user_input_lower = user_input.lower()

            # Check for capabilities query
            if any(phrase in user_input_lower for phrase in self.intents['capabilities']):
                return random.choice(self.capabilities_response)

            # Check for small talk
            small_talk_response = self.handle_small_talk(user_input)
            if small_talk_response:
                return small_talk_response

            # Check for time query
            if any(phrase in user_input_lower for phrase in self.intents['time_query']):
                return self.handle_time_query()

            # Check for date query
            if any(phrase in user_input_lower for phrase in self.intents['date_query']):
                return self.handle_date_query()

            # Check for name query
            if any(phrase in user_input_lower for phrase in self.intents['name_query']):
                return f"Your name is {self.user_name}!"

            # Check for greetings
            if any(phrase in user_input_lower for phrase in self.intents['greeting']):
                return f"Hello again, {self.user_name}! How can I help you?"

            # Check for farewell
            if any(phrase in user_input_lower for phrase in self.intents['farewell']):
                return f"Goodbye, {self.user_name}! Have a great day!"

            # Try to find matching question in QA dataset
            best_match = self.find_best_match(user_input)
            if best_match:
                return self.qa_pairs[best_match]

            return "I'm not sure how to answer that. Could you rephrase your question?"

        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return "I encountered an error. Please try again."

def main():
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()