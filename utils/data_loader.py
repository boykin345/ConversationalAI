# utils/data_loader.py

import csv
from sklearn.feature_extraction.text import TfidfVectorizer

def load_qa_dataset():
    """Load the QA dataset and vectorize questions."""
    qa_pairs = {}
    questions = []
    qa_vectors = None
    qa_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer='word',
        max_features=10000,
        lowercase=True,
        stop_words='english'
    )

    try:
        with open('COMP3074-CW1-Dataset.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row['Question'].lower()
                qa_pairs[question] = row['Answer']
        questions = list(qa_pairs.keys())
        qa_vectors = qa_vectorizer.fit_transform(questions)
    except Exception as e:
        print(f"Error loading QA dataset: {str(e)}")
        qa_pairs = {}

    return qa_pairs, questions, qa_vectors, qa_vectorizer
