from google import genai
from google.genai import types
from typing import Dict, List, Any
import json

from models import FlashcardSet
from config import Config
from utils import clean_flashcard_text

def generate_flashcards_batch(client, topic):
    """Generate flashcards from topic in batch mode"""
    try:
        # Use the new prompt template for multiple-choice questions
        prompt = Config.generate_prompt_template(topic)
        
        # Generate flashcards with JSON response
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=types.Part.from_text(text=prompt),
            config=Config.FLASHCARD_CONFIG
        )
        
        # Parse JSON response
        try:
            # Extract response as JSON
            flashcards_data = json.loads(response.text)
            
            # Format flashcards for compatibility with existing UI
            formatted_flashcards = []
            
            # Convert the multiple-choice format to the legacy format for backward compatibility
            for card in flashcards_data:
                # Format: Q: [question] | A: [correct_answer] (+ incorrect answers as options)
                options = [card['ca']] + card['ia']
                options_formatted = ", ".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
                formatted_card = f"Q: {card['q']} | A: {card['ca']}"
                formatted_flashcards.append(formatted_card)
                
                # Also store the raw card for future use with a new UI
                card['_raw'] = card.copy()
            
            return {
                'flashcards': formatted_flashcards,
                'count': len(formatted_flashcards),
                'mc_data': flashcards_data  # Include the raw multiple-choice data
            }
            
        except json.JSONDecodeError:
            # Fallback to legacy processing if JSON parsing fails
            print("Warning: Failed to parse JSON response, falling back to legacy format")
            raw_cards = response.text.split('\n')
            all_flashcards = []
            
            generator = FlashcardGenerator(client)
            for card in raw_cards:
                cleaned = clean_flashcard_text(card)
                if cleaned and generator.unique_cards.add(cleaned):
                    all_flashcards.append(cleaned)
            
            return {
                'flashcards': all_flashcards,
                'count': len(all_flashcards)
            }
                
    except Exception as e:
        print(f"Error in generate_flashcards_batch: {str(e)}")
        raise
