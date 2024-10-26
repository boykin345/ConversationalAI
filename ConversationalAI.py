import re
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class Chatbot:
    def __init__(self, qa_file='COMP3074-CW1-Dataset.csv'):
        self.user_name = None
        self.qa_dataset = self.load_qa_dataset(qa_file)
        self.intents = {
            "greet": ["hi", "hello", "hey"],
            "name_query": ["what is my name"],
            "small_talk": ["how are you"],
            "capability_query": ["what can you do"],
            "qa": list(self.qa_dataset.keys())
        }
        self.stop_words = set(stopwords.words("english"))

    def load_qa_dataset(self, qa_file):
        """Loads the Q&A dataset from a CSV file."""
        dataset = {}
        with open(qa_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                question = row['Question'].lower()
                answer = row['Answer']
                dataset[question] = answer
        return dataset

    def greet_user(self):
        if self.user_name:
            return f"Hello {self.user_name}, how can I help you today?"
        return "Hello, welcome! What is your name?"

    def handle_name(self, user_input):
        """Attempts to capture the user's name."""
        if not self.user_name:
            # First, look for phrases like "name is," "I am," or "I'm" followed by a name
            name_match = re.search(r"(?:name is|i am|i'm)\s+(\w+)", user_input, re.IGNORECASE)
            
            if name_match:
                # Capture the name from the matched group
                self.user_name = name_match.group(1)
                return f"Nice to meet you, {self.user_name}!"
            
            # If no phrase is found, assume the entire input is the name if it's a single word
            elif user_input.strip().isalpha():
                self.user_name = user_input.strip()
                return f"Nice to meet you, {self.user_name}!"
            
            return "I didn't catch your name. Please tell me your name."
        return None



    def preprocess_text(self, text):
        """Tokenizes and removes stop words from text."""
        tokens = word_tokenize(text.lower())
        filtered_words = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return ' '.join(filtered_words)

    def handle_intent(self, user_input):
        """Routes conversation based on detected intent."""
        # Preprocess user input for better matching
        user_input_lower = self.preprocess_text(user_input)

        # Detect greeting
        if any(greet in user_input_lower for greet in self.intents["greet"]):
            return self.greet_user()
        
        # Detect name query
        elif any(name_query in user_input_lower for name_query in self.intents["name_query"]):
            if self.user_name:
                return f"Your name is {self.user_name}."
            return "I don't know your name yet. Please tell me your name."

        # Detect small talk
        elif any(talk in user_input_lower for talk in self.intents["small_talk"]):
            return "I'm just a chatbot, but thank you for asking!"

        # Detect capability query
        elif any(capability in user_input_lower for capability in self.intents["capability_query"]):
            return "I can help you book flights, answer questions, and chat with you!"

        # Detect QA queries
        for question in self.intents["qa"]:
            if question in user_input_lower:
                return self.qa_dataset[question]

        # If no intent is detected, ask for clarification
        return "I'm sorry, I didn't understand that. Could you rephrase?"

    def chat(self, user_input):
        """Main chat function that processes user input."""
        if not self.user_name:
            name_response = self.handle_name(user_input)
            if name_response:
                return name_response
        response = self.handle_intent(user_input)
        return response


# Example of running the chatbot
chatbot = Chatbot()
print(chatbot.greet_user())

# Simulating conversation flow
while True:
    user_input = input()  # User's input during conversation
    response = chatbot.chat(user_input)
    print(response)
