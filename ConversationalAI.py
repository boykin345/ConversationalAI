import re

class Chatbot:
    def __init__(self):
        self.user_name = None
        self.qa_dataset = {
            "what are stocks and bonds": "Stocks represent ownership in a company, while bonds are loans made by an investor to a borrower."
        }
        self.intents = {
            "greet": ["hi", "hello", "hey"],
            "name_query": ["what is my name"],
            "small_talk": ["how are you"],
            "capability_query": ["what can you do"],
            "qa": list(self.qa_dataset.keys())
        }

    def greet_user(self):
        if self.user_name:
            return f"Hello {self.user_name}, how can I help you today?"
        return "Hello, welcome! What is your name?"

    def handle_name(self, user_input):
        """Attempts to capture the user's name."""
        if not self.user_name:
            # Assume the first word is the user's name
            name_guess = user_input.strip().split()[0]
            if name_guess.isalpha():
                self.user_name = name_guess
                return f"Nice to meet you, {self.user_name}!"
            return "I didn't catch your name. Please tell me your name."
        return None

    def handle_intent(self, user_input):
        """Routes conversation based on detected intent."""
        # Lowercase the user input for case-insensitive matching
        user_input_lower = user_input.lower()

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
