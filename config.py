import os
import tempfile
from google.genai import types

class Config:
    """Application configuration"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    # Use temporary directory for Vercel
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'flashcard_uploads')
    CHUNK_SIZE = 15000
    DEFAULT_BATCH_SIZE = 100  # Number of cards to generate per request
    
    # JSON Schema for multiple-choice flashcards
    FLASHCARD_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},  # short for question
                "ca": {"type": "string"}, # short for correct_answer
                "ia": {                   # short for incorrect_answers
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3
                }
            },
            "required": ["q", "ca", "ia"]
        }
    }
    
    FLASHCARD_CONFIG = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        max_output_tokens=8192,
        response_mime_type="application/json",  # Ensure JSON response
        stop_sequences=['###'],
        system_instruction=(
            "You are a professional educator creating multiple-choice flashcards. "
            "Each flashcard must have a question, one correct answer, and three incorrect answers. "
            "Your responses must strictly follow the JSON schema provided."
        )
    )
    
    @staticmethod
    def generate_prompt_template(topic, batch_size=None):
        """Generate a prompt template for AI flashcard generation"""
        if batch_size is None:
            batch_size = Config.DEFAULT_BATCH_SIZE
            
        return f"""
        <data>{topic}</data>
        <instructions>
        You are an expert educator creating flashcards about <data>.
        Generate comprehensive, accurate, and engaging flashcards following these strict guidelines:

        1. Each flashcard must have:
           - A clear, concise question that tests understanding
           - One definitively correct answer
           - Three plausible but incorrect answers
           - CRITICAL: All answers (correct and incorrect) MUST:
             * Be similar in length (within 10-15 characters of each other)
             * Use the same level of detail and complexity
             * Follow the same grammatical structure
             * Be equally specific/general
        
        2. Question types must be evenly distributed:
           - Factual recall (25% of cards)
           - Concept application (25% of cards)
           - Problem-solving (25% of cards)
           - Relationships between concepts (25% of cards)
        
        3. Ensure quality control:
           - No duplicate questions or answers
           - All content is factually accurate
           - Clear, unambiguous wording
           - Progressive difficulty (easy -> medium -> hard)
           - Avoid answers that are obviously wrong
           - Don't make the correct answer stand out by length or detail
        
        Format your response as a JSON array of objects, each with:
        - 'q': the flashcard question (short for question)
        - 'ca': the correct answer (short for correct_answer)
        - 'ia': array of exactly three incorrect answers (short for incorrect_answers)

        Generate {batch_size} unique flashcards covering different aspects of the topic.
        Ensure comprehensive coverage by:
        1. Breaking down the topic into key subtopics
        2. Creating equal numbers of cards for each subtopic
        3. Varying question types within each subtopic
        4. Including both fundamental and advanced concepts
        5. Maintaining consistent answer length and style throughout
        </instructions>
        """