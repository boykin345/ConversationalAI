import re
import dateparser
import csv
import random
import pytz
import numpy as np
from dateparser.search import search_dates
from datetime import datetime, timedelta
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
    user_input = user_input.lower()
    patterns = [
        # Pattern 1: from City1 to City2
        r'from\s+([A-Za-z\s\-]+?)\s+to\s+([A-Za-z\s\-]+)',
        # Pattern 2: to City2 from City1
        r'to\s+([A-Za-z\s\-]+?)\s+from\s+([A-Za-z\s\-]+)',
        # Pattern 3: City1 - City2
        r'([A-Za-z\s\-]+?)\s*-\s*([A-Za-z\s\-]+)',
        # Pattern 4: City1 to City2
        r'([A-Za-z\s\-]+?)\s+to\s+([A-Za-z\s\-]+)',
        # Pattern 5: from City
        r'from\s+([A-Za-z\s\-]+)',
        # Pattern 6: to City
        r'to\s+([A-Za-z\s\-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                departure = groups[0].strip().title()
                destination = groups[1].strip().title()
                return departure, destination
            elif len(groups) == 1:
                if 'from' in pattern:
                    departure = groups[0].strip().title()
                    return departure, None
                elif 'to' in pattern:
                    destination = groups[0].strip().title()
                    return None, destination
    return None, None

def extract_single_date(user_input):
    """Extract a single date from user input."""
    dates = search_dates(user_input, settings={'PREFER_DATES_FROM': 'future'})
    if dates:
        date = dates[0][1]
        return date.strftime('%d/%m/%Y')
    else:
        return None

def extract_travel_dates(user_input):
    """Extract travel dates and flexibility from user input."""
    user_input = user_input.lower()
    flexible = 'flexible' in user_input

    # Use dateparser to parse natural language dates
    dates = search_dates(user_input, settings={'PREFER_DATES_FROM': 'future'})

    if dates:
        parsed_dates = [date[1] for date in dates]
        if parsed_dates:
            departure_date = parsed_dates[0]
            departure_date_str = departure_date.strftime('%d/%m/%Y')

            if len(parsed_dates) > 1:
                return_date = parsed_dates[1]
                return_date_str = return_date.strftime('%d/%m/%Y')
            else:
                return_date_str = None

            return departure_date_str, return_date_str, flexible

    # No dates found
    return None, None, flexible

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

def get_current_time_in_nottingham():
    """Get the current time in Nottingham."""
    nottingham_timezone = pytz.timezone('Europe/London')
    now_nottingham = datetime.now(nottingham_timezone)
    return now_nottingham

def get_time_of_day(now):
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
        self.tickets = self.load_ticket_dataset()
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

    def load_ticket_dataset(self):
        """Load the ticket dataset from a CSV file."""
        tickets = []
        try:
            with open('tickets.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert numerical fields
                    row['flight_id'] = int(row['flight_id'])
                    row['available_seats'] = int(row['available_seats'])
                    row['price'] = float(row['price'])
                    tickets.append(row)
            return tickets
        except Exception as e:
            print(f"Error loading ticket dataset: {e}")
            return []
        
    def check_flight_availability(self, departure_city, destination_city, date):
        """Check flight availability in the dataset."""
        available_flights = []
        for ticket in self.tickets:
            if (ticket['departure_city'].lower() == departure_city.lower() and
                ticket['destination_city'].lower() == destination_city.lower() and
                ticket['departure_date'] == date and
                ticket['available_seats'] > 0):
                available_flights.append(ticket)
        return available_flights

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
        """Handle the flight booking transaction."""
        if 'quit transaction' in user_input.lower():
            self.in_transaction = False
            self.transaction_data = {}
            return "Transaction cancelled."

        # Step 1: Get departure and destination cities
        if not self.transaction_data.get('departure_city') or not self.transaction_data.get('destination_city'):
            departure, destination = extract_cities(user_input)
            if departure and destination:
                self.transaction_data['departure_city'] = departure
                self.transaction_data['destination_city'] = destination
                return "Would that be a return trip or a single flight?"
            elif departure:
                self.transaction_data['departure_city'] = departure
                return "Where would you like to fly to?"
            elif destination:
                self.transaction_data['destination_city'] = destination
                return "Where would you be flying from?"
            else:
                return "Please specify the departure and destination cities."

        # Step 2: Get trip type
        if not self.transaction_data.get('trip_type'):
            if 'return' in user_input.lower():
                self.transaction_data['trip_type'] = 'return'
            elif 'single' in user_input.lower() or 'one-way' in user_input.lower():
                self.transaction_data['trip_type'] = 'single'
            else:
                return "Please specify if it's a return trip or a single flight."
            return "When would you like to travel?"

        # Step 3: Get travel dates
        if not self.transaction_data.get('departure_date'):
            departure_date, return_date, flexible = extract_travel_dates(user_input)
            if departure_date:
                self.transaction_data['departure_date'] = departure_date
                if self.transaction_data['trip_type'] == 'return':
                    if return_date:
                        self.transaction_data['return_date'] = return_date
                    else:
                        return "Please specify your return date."
                return self.search_and_present_flights()
            return "Please provide your travel dates and mention if you're flexible."

        if self.transaction_data['trip_type'] == 'return' and not self.transaction_data.get('return_date'):
            return_date = extract_single_date(user_input)
            if return_date:
                self.transaction_data['return_date'] = return_date
                return self.search_and_present_flights()
            return "Please provide a valid return date."

        # Step 4: Handle flight selection
        if 'awaiting_flight_selection' in self.transaction_data:
            return self.process_flight_selection(user_input)

        # Step 5: Handle booking finalization
        if 'proceed' in user_input.lower():
            return self.finalize_booking()
        elif 'cancel' in user_input.lower():
            self.in_transaction = False
            self.transaction_data = {}
            return "Booking cancelled."
        return "Would you like to proceed with the booking?"

    def search_and_present_flights(self):
        """Search for flights and present options to the user."""
        # Check availability for departure flight
        available_flights = self.check_flight_availability(
            self.transaction_data['departure_city'],
            self.transaction_data['destination_city'],
            self.transaction_data['departure_date']
        )
        self.transaction_data['available_flights'] = available_flights

        # Check availability for return flight if it's a return trip
        if self.transaction_data['trip_type'] == 'return':
            return_flights = self.check_flight_availability(
                self.transaction_data['destination_city'],
                self.transaction_data['departure_city'],
                self.transaction_data['return_date']
            )
            self.transaction_data['return_flights'] = return_flights
        else:
            self.transaction_data['return_flights'] = []

        # Check if flights are available
        if available_flights and (self.transaction_data['trip_type'] == 'single' or self.transaction_data['return_flights']):
            # Present flight options to the user
            self.transaction_data['awaiting_flight_selection'] = True
            return self.present_flight_options()
        else:
            if self.transaction_data['trip_type'] == 'return':
                return "I'm sorry, there are no return flights available on that date. Would you like to change the return date or book a single flight?"
            else:
                return "I'm sorry, there are no flights available on that date. Would you like to change the date or cancel the booking?"

    def present_flight_options(self):
        """Present available flight options to the user."""
        message = "Here are the available flights:\n"

        # List departure flights
        message += "\nDeparture flights:\n"
        for idx, flight in enumerate(self.transaction_data['available_flights'], start=1):
            message += f"{idx}. Flight {flight['flight_id']} from {flight['departure_city']} to {flight['destination_city']} on {flight['departure_date']} at ${flight['price']}\n"

        # List return flights if applicable
        if self.transaction_data['trip_type'] == 'return':
            message += "\nReturn flights:\n"
            for idx, flight in enumerate(self.transaction_data['return_flights'], start=1):
                message += f"{idx}. Flight {flight['flight_id']} from {flight['departure_city']} to {flight['destination_city']} on {flight['departure_date']} at ${flight['price']}\n"

        message += "\nPlease select your departure flight by entering the number."
        return message

    def process_flight_selection(self, user_input):
        """Process the user's flight selection."""
        # Handle proceed/cancel commands first
        if user_input.lower() == 'proceed':
            return self.finalize_booking()
        if user_input.lower() == 'cancel':
            self.in_transaction = False
            self.transaction_data = {}
            return "Booking cancelled."

        try:
            selection = int(user_input)
            
            # Handle departure flight selection
            if not self.transaction_data.get('selected_departure_flight'):
                if 0 < selection <= len(self.transaction_data['available_flights']):
                    self.transaction_data['selected_departure_flight'] = self.transaction_data['available_flights'][selection - 1]
                    if self.transaction_data['trip_type'] == 'return':
                        return_flights = self.transaction_data.get('return_flights', [])
                        if return_flights:
                            return "Now please select your return flight by entering its number."
                    return self.present_confirmation_prompt()
                return "Invalid flight number. Please try again."
                
            # Handle return flight selection
            if (self.transaction_data['trip_type'] == 'return' and 
                not self.transaction_data.get('selected_return_flight')):
                if 0 < selection <= len(self.transaction_data['return_flights']):
                    self.transaction_data['selected_return_flight'] = self.transaction_data['return_flights'][selection - 1]
                    return self.present_confirmation_prompt()
                return "Invalid flight number. Please try again."
                
        except ValueError:
            return "Please enter a valid flight number."

    def present_confirmation_prompt(self):
        """Present booking confirmation prompt."""
        if self.transaction_data['trip_type'] == 'single' or self.transaction_data.get('selected_return_flight'):
            return "Would you like to proceed with booking? (Type 'proceed' or 'cancel')"
        return "Please select your flight number."

    def finalize_booking(self):
        """Complete the booking process."""
        if not self.transaction_data.get('selected_departure_flight'):
            return "No flights selected. Please start your booking again."
            
        total_price = self.transaction_data['selected_departure_flight']['price']
        if self.transaction_data['trip_type'] == 'return':
            if not self.transaction_data.get('selected_return_flight'):
                return "Return flight not selected. Please select a return flight."
            total_price += self.transaction_data['selected_return_flight']['price']
            
        # Update available seats
        self.transaction_data['selected_departure_flight']['available_seats'] -= 1
        if self.transaction_data.get('selected_return_flight'):
            self.transaction_data['selected_return_flight']['available_seats'] -= 1
        
        self.in_transaction = False
        self.transaction_data = {}
        return f"Your booking is confirmed. The total price is ${total_price:.2f}. Thank you for choosing Skynet Travel Agency!"


    def extract_cities(user_input):
        # Simple extraction using regular expressions
        # Implement proper NLP parsing for production code
        import re
        pattern = r'from (\w+) to (\w+)'
        match = re.search(pattern, user_input.lower())
        if match:
            return match.group(1).title(), match.group(2).title()
        return None, None


    def handle_user_input(self, user_input):
        """Process user input and generate response."""
        user_input_lower = user_input.lower()

        if not self.user_name:
            extracted_name = extract_name(user_input)
            if extracted_name:
                self.user_name = extracted_name
                return f"Nice to meet you, {self.user_name}! How can I help you today?\n(to end the conversation write 'quit' or 'exit')"
            else:
                return "I didn't quite catch your name. Could you tell me again?"

        if self.in_transaction:
            # Handle transaction flow
            return self.handle_transaction(user_input)
        else:
            booking_patterns = [
            r'\bbook (?:a )?(?:flight|ticket)\b',
            r'\bneed (?:a )?(?:flight|ticket)\b',
            r'\bget (?:me )?(?:a )?(?:flight|ticket)\b',
            r'\bfind (?:me )?(?:a )?(?:flight|ticket)\b',
            r'\blooking for (?:a )?(?:flight|ticket)\b',
            r'\bi want to (?:fly|travel)(?: to| from)?\b',
            r'\bi want (?:a )?(?:flight|ticket)\b',
            r'\b(?:flight|ticket) from\b',
            r'\b(?:flight|ticket) to\b',
            r'\bfly (?:to|from)\b',
            r'\btravel (?:to|from)\b',
            r'\bneed to go to\b',
        ]
            booking_intent_detected = False
            for pattern in booking_patterns:
                if re.search(pattern, user_input_lower):
                    booking_intent_detected = True
                    break
            if booking_intent_detected:
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
                return "Welcome to Skynet Travel Agency! How can I assist you today?\n(write city to city to book a flight(if city's name is more than one word, please use '-' to separate the words.))\nTo quit write 'quit transaction'"

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
