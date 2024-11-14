import re
import random
import pytz
import numpy as np
from datetime import datetime
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

def extract_cities(user_input):
    """Extract departure and destination cities from user input."""
    # Pattern to match "from City1 to City2"
    pattern = r'from\s+([A-Za-z\s]+)\s+to\s+([A-Za-z\s]+)'
    match = re.search(pattern, user_input.lower())
    if match:
        departure = match.group(1).strip().title()
        destination = match.group(2).strip().title()
        return departure, destination
    else:
        # Try alternative pattern "to City2 from City1"
        pattern = r'to\s+([A-Za-z\s]+)\s+from\s+([A-Za-z\s]+)'
        match = re.search(pattern, user_input.lower())
        if match:
            departure = match.group(2).strip().title()
            destination = match.group(1).strip().title()
            return departure, destination
    return None, None

def extract_travel_dates(user_input):
    """Extract travel dates and flexibility from user input."""
    # Regex pattern to match dates in formats like DD/MM/YYYY or D/M/YYYY
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
    dates = re.findall(date_pattern, user_input)
    flexible = 'flexible' in user_input.lower()
    if dates:
        departure_date = dates[0]
        return_date = dates[1] if len(dates) > 1 else None
        return departure_date, return_date, flexible
    else:
        # Try to parse dates in formats like '10th to 19th'
        date_range_pattern = r'(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|-)\s*(\d{1,2})(?:st|nd|rd|th)?'
        match = re.search(date_range_pattern, user_input.lower())
        if match:
            current_month = datetime.now().month
            current_year = datetime.now().year
            departure_day = match.group(1)
            return_day = match.group(2)
            departure_date = f"{departure_day}/{current_month}/{current_year}"
            return_date = f"{return_day}/{current_month}/{current_year}"
            return departure_date, return_date, flexible
    return None, None, flexible

def check_flight_availability(date):
    """Simulate checking flight availability."""
    # For demonstration, let's assume '19/11/2021' is unavailable
    unavailable_dates = ['19/11/2021']
    if date in unavailable_dates:
        return False
    return True

def extract_new_return_date(user_input):
    """Extract a new return date from user input."""
    # Try to find a date in DD/MM/YYYY format
    date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
    match = re.search(date_pattern, user_input)
    if match:
        return match.group(1)
    else:
        # Try to extract a day number (e.g., "Change the date to the 20th")
        day_pattern = r'change the date to (?:the )?(\d{1,2})(?:st|nd|rd|th)?'
        match = re.search(day_pattern, user_input.lower())
        if match:
            day = match.group(1)
            current_month = datetime.now().month
            current_year = datetime.now().year
            return f"{day}/{current_month}/{current_year}"
    return None


class Chatbot:
    def __init__(self):
        self.in_transaction = False
        self.transaction_data = {}
        self.transaction_data = {
            'departure_city': None,
            'destination_city': None,
            'trip_type': None,
            'departure_date': None,
            'return_date': None,
            'date_flexible': None,
        }
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
            self.greeting_responses,
            self.farewell_responses
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

    
    def handle_transaction(self, user_input):
        if 'Quit transaction' in user_input.lower():
            self.in_transaction = False
            self.transaction_data = {}
            return "Transaction cancelled. If there's anything else I can assist you with, please let me know."
        if not self.transaction_data['departure_city']:
            # Extract departure and destination cities
            # Assuming you have a function to extract cities
            departure, destination = extract_cities(user_input)
            if departure and destination:
                self.transaction_data['departure_city'] = departure
                self.transaction_data['destination_city'] = destination
                return "Would that be a return trip or a single flight?"
            else:
                return "I'm sorry, could you please specify the departure and destination cities?"

        elif not self.transaction_data['trip_type']:
            # Determine trip type
            if 'return' in user_input.lower():
                self.transaction_data['trip_type'] = 'return'
            elif 'single' in user_input.lower() or 'one-way' in user_input.lower():
                self.transaction_data['trip_type'] = 'single'
            else:
                return "Please specify if it's a return trip or a single flight."
            return "When were you thinking of going?"

        elif not self.transaction_data['departure_date']:
            # Extract travel dates
            departure_date, return_date, date_flexible = extract_travel_dates(user_input)
            if departure_date:
                self.transaction_data['departure_date'] = departure_date
                self.transaction_data['return_date'] = return_date if self.transaction_data['trip_type'] == 'return' else None
                self.transaction_data['date_flexible'] = date_flexible
                return "Thank you, I will search my database for available tickets. Please wait."
            else:
                return "Please provide the precise dates of your trip and whether you are flexible with your dates."

        else:
            # Process booking or handle further steps
            # For simplicity, let's assume we found flights
            if self.transaction_data['trip_type'] == 'return' and not self.transaction_data.get('flight_options_provided'):
                # Simulate checking availability
                available_return = check_flight_availability(self.transaction_data['return_date'])
                if available_return:
                    self.transaction_data['flight_options_provided'] = True
                    return f"I have found flights on your requested dates. Would you like to proceed with booking?(type 'proceed' to continue/ 'change' to change the return date)"
                else:
                    return "I'm sorry, there are no return flights available on that date. Would you like to change the return date or book a single flight?"
            else:
                # Handle user's decision
                if 'change' in user_input.lower() and 'date' in user_input.lower():
                    new_return_date = extract_new_return_date(user_input)
                    if new_return_date:
                        self.transaction_data['return_date'] = new_return_date
                        # Re-check availability
                        return "Let me check the availability for the new return date. Please wait."
                    else:
                        return "Please specify the new return date you'd like."
                elif 'proceed' in user_input.lower():
                    # Complete booking
                    self.in_transaction = False
                    self.transaction_data = {}
                    return "Your booking is confirmed. Thank you for choosing Skynet Travel Agency!"
                elif 'quit' in user_input.lower():
                    self.in_transaction = False
                    self.transaction_data = {}
                    return "Transaction cancelled. If there's anything else I can assist you with, please let me know."
                else:
                    return "I'm not sure how to proceed. Would you like to change the return date or proceed with booking?"
                
    def extract_cities(user_input):
        # Simple extraction using regular expressions
        # Implement proper NLP parsing for production code
        import re
        pattern = r'from (\w+) to (\w+)'
        match = re.search(pattern, user_input.lower())
        if match:
            return match.group(1).title(), match.group(2).title()
        return None, None

    def extract_travel_dates(user_input):
        # Extract dates and flexibility
        # Implement proper date parsing
        # For now, let's assume dates are in DD/MM/YYYY format
        import re
        date_pattern = r'(\d{2}/\d{2}/\d{4})'
        dates = re.findall(date_pattern, user_input)
        flexible = 'flexible' in user_input.lower()
        if dates:
            departure_date = dates[0]
            return_date = dates[1] if len(dates) > 1 else None
            return departure_date, return_date, flexible
        return None, None, flexible

    def extract_new_return_date(user_input):
        # Extract new return date
        import re
        date_pattern = r'(\d{2}/\d{2}/\d{4})'
        match = re.search(date_pattern, user_input)
        if match:
            return match.group(1)
        else:
            # Check for day specification like '20th'
            day_pattern = r'(\d{1,2})(?:st|nd|rd|th)?'
            match = re.search(day_pattern, user_input)
            if match:
                day = match.group(1)
                # You would need more context to construct full date
                # For simplicity, assume same month and year
                return f"{day}/11/2021"
        return None
    
    def get_current_time_in_nottingham(self):
        """Get the current time in Nottingham."""
        nottingham_timezone = pytz.timezone('Europe/London')
        now_nottingham = datetime.now(nottingham_timezone)
        return now_nottingham

    def get_time_of_day(self, now):
        """Determine the time of day based on the hour."""
        hour = now.hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    def check_flight_availability(date):
        # Simulate checking flight availability
        # In production, this would query a database or API
        unavailable_dates = ['19/11/2021']
        return date not in unavailable_dates



    def handle_user_input(self, user_input):
        """Process user input and generate response."""
        if not self.user_name:
            extracted_name = extract_name(user_input)
            if extracted_name:
                self.user_name = extracted_name
                return f"Nice to meet you, {self.user_name}! How can I help you today?"
            else:
                return "I didn't quite catch your name. Could you tell me again?"

        if self.in_transaction:
            # Handle transaction flow
            return self.handle_transaction(user_input)
        else:
            # Check for transaction initiation intent only if not already in a transaction
            if 'book a flight' in user_input.lower() or 'flight from' in user_input.lower():
                self.in_transaction = True
                self.transaction_data = {
                    'departure_city': None,
                    'destination_city': None,
                    'trip_type': None,
                    'departure_date': None,
                    'return_date': None,
                    'date_flexible': None,
                    'flight_options_provided': False,
                }
                return "Hello, welcome to the Skynet Travel Agency. How may I assist you?"

            # Proceed with intent recognition
            intent, score = get_intent(
                user_input,
                self.intent_vectorizer,
                self.intent_vectors,
                self.intent_mapping
            )

            if intent and score >= 0.7:
                # Handle recognized intents
                if intent == 'greeting':
                    now_nottingam = self.get_current_time_in_nottingham()
                    time_of_day = self.get_time_of_day(now_nottingam)
                    response = random.choice(self.greeting_responses)
                    return response.format(name=self.user_name, time_of_day=time_of_day)
                elif intent == 'farewell':
                    response = random.choice(self.farewell_responses)
                    return response.format(name=self.user_name)
                elif intent == 'capabilities':
                    return random.choice(self.capabilities_response)
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
                    return "I'm not sure how to answer that. Could you rephrase your question?"
            else:
                # Check if input is a question
                is_question = (
                    user_input.strip().endswith('?') or
                    user_input.strip().lower().startswith(
                        ('what', 'how', 'why', 'where', 'when', 'who')
                    )
                )

                if is_question:
                    # Try to find a QA match
                    best_qa_match = self.find_best_qa_match(user_input)
                    if best_qa_match:
                        return self.qa_pairs[best_qa_match]

            # If all else fails
            return "I'm not sure how to answer that. Could you rephrase your question?"
