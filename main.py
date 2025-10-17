import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import random
import string

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
def load_intents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['intents']

# Preprocess text: tokenize and lemmatize
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Find matching intent based on user input
def get_intent(user_input, intents):
    user_tokens = preprocess_text(user_input)
    
    for intent in intents:
        for pattern in intent['patterns']:
            pattern_tokens = preprocess_text(pattern)
            # Check if any pattern tokens match user input tokens
            if any(token in user_tokens for token in pattern_tokens):
                return random.choice(intent['responses'])
    # Default response if no intent matches
    return "Sorry, I didn't understand that. Try something else!"

# Main chatbot function
def chatbot():
    intents = load_intents('intents.json')
    print("🤖 Hello! How can I help you today? (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("🤖 Goodbye!")
            break
        response = get_intent(user_input, intents)
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
