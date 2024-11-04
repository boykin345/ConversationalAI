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

            # Try to find matching question
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