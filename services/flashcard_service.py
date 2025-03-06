from google import genai
from google.genai import types
from typing import Dict, List

from models import FlashcardSet
from config import Config
from utils import clean_flashcard_text

class FlashcardGenerator:
    """Generate and process flashcards"""
    def __init__(self, client):
        self.client = client
        self.unique_cards = FlashcardSet()

def generate_flashcards_batch(client, topic):
    """Generate flashcards from topic in batch mode"""
    generator = FlashcardGenerator(client)
    
    try:
        # Formulate prompt
        prompt = Config.FLASHCARD_GENERATION_PROMPT + f"\nTopic: {topic}"
        
        # Generate flashcards
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=types.Part.from_text(text=prompt),
            config=Config.FLASHCARD_CONFIG
        )
        
        # Extract flashcards
        all_flashcards = []
        raw_cards = response.text.split('\n')
        
        for card in raw_cards:
            cleaned = clean_flashcard_text(card)
            if cleaned and generator.unique_cards.add(cleaned):
                all_flashcards.append(cleaned)
        
        return {
            'flashcards': all_flashcards,
            'count': len(generator.unique_cards)
        }
                
    except Exception as e:
        print(f"Error in generate_flashcards_batch: {str(e)}")
        raise
