
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def initialize_intents():
    """Initialize intents with example phrases."""
    intents = {
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

    capabilities_response = [
        "I can help with answering questions from the QA dataset, providing the current time and date, checking the weather for any city, and remembering your name."
    ]

    small_talk_responses = [
        "I'm doing well, thank you!",
        "I'm fine, thanks for asking!",
        "All good here!",
        "I'm just a bot, but I'm doing great!"
    ]

    intent_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer='char_wb',
        max_features=5000,
        lowercase=True,
        strip_accents='unicode'
    )

    all_phrases = []
    intent_mapping = []

    for intent, phrases in intents.items():
        intent_doc = ' '.join(phrases)
        all_phrases.append(intent_doc)
        intent_mapping.append(intent)

    intent_vectors = intent_vectorizer.fit_transform(all_phrases)

    return intents, intent_mapping, intent_vectors, intent_vectorizer, capabilities_response, small_talk_responses

def get_intent(user_input, intent_vectorizer, intent_vectors, intent_mapping, threshold=0.4):
    """Determine intent using similarity matching."""
    user_input = user_input.lower().strip()

    try:
        user_vector = intent_vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, intent_vectors)[0]

        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]

        if best_match_score >= threshold:
            return intent_mapping[best_match_index], best_match_score

    except Exception as e:
        print(f"Error in intent matching: {str(e)}")

    return None, 0.0

def extract_name(user_input):
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

def extract_location(user_input):
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

def extract_time_location(user_input):
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
