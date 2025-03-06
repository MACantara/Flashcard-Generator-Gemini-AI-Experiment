import os
import tempfile
from google.genai import types

class Config:
    """Application configuration"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    # Use temporary directory for Vercel
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'flashcard_uploads')
    CHUNK_SIZE = 15000
    MAX_CARDS = 100
    
    FLASHCARD_CONFIG = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        max_output_tokens=8192,
        stop_sequences=['###'],
        system_instruction=(
            "You are a professional educator creating clear, concise flashcards. "
            "Each flashcard should be formatted exactly as 'Q: question | A: answer'. "
            "Questions should be specific and answers should be comprehensive but brief."
        )
    )
    
    # Centralized Prompts
    FLASHCARD_GENERATION_PROMPT = f"""
    Create {MAX_CARDS} flashcards from this content.
    Format EXACTLY as shown:
    Q: [question here] | A: [answer here]

    Each flashcard must be on its own line.
    Do not include any other text.
    """
