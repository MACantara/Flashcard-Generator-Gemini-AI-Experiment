from flask import Flask, render_template, request, jsonify, Response, session
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
import re
from werkzeug.utils import secure_filename
import tempfile
import uuid
from io import StringIO
import PyPDF2
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass

app = Flask(__name__)
load_dotenv()

# Configure Gemini AI
client = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

# Types and Models
@dataclass
class FlashcardEvent:
    """Event data structure for SSE responses"""
    type: str
    content: str
    
    def to_sse(self) -> str:
        """Convert to SSE format"""
        return f"data: {json.dumps({'type': self.type, 'content': self.content})}\n\n"

class FlashcardSet:
    """Manage unique flashcards with deduplication"""
    def __init__(self):
        self._cards: set = set()
        self._count: int = 0
    
    def add(self, card: str) -> bool:
        """Add card if unique, return True if added"""
        card_normalized = self._normalize_card(card)
        if card_normalized not in self._cards:
            self._cards.add(card_normalized)
            self._count += 1
            return True
        return False
    
    @staticmethod
    def _normalize_card(card: str) -> str:
        """Normalize card text for comparison"""
        q, a = card.split('|')
        return f"{' '.join(q.split()).lower()}|{' '.join(a.split()).lower()}"
    
    def __len__(self) -> int:
        return self._count

# Configuration
class Config:
    """Application configuration"""
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    UPLOAD_FOLDER = 'temp_uploads'
    CHUNK_SIZE = 50000
    MAX_CARDS = 300
    
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
    
    COVERAGE_CONFIG = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=50,
        system_instruction=(
            "You are an educational content evaluator. "
            "Analyze the flashcards and respond ONLY with 'yes' or 'no' based on topic coverage. "
            "Answer 'yes' if the flashcards cover 90-100% of the content, 'no' otherwise."
        )
    )
    
    # Centralized Prompts
    FLASHCARD_GENERATION_PROMPT = """
    Create flashcards from this content.
    Format EXACTLY as shown:
    Q: [question here] | A: [answer here]

    Each flashcard must be on its own line.
    Do not include any other text.
    """

    COVERAGE_CHECK_PROMPT = """
    Evaluate if these flashcards provide comprehensive coverage:

    {content}

    Answer ONLY with 'yes' or 'no'.
    """

# Utilities
def chunk_text(text: str, size: int = Config.CHUNK_SIZE) -> List[str]:
    """Split text into chunks of approximately given size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1
        if current_size + word_size > size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

class FileProcessor:
    """Handle file processing operations"""
    @staticmethod
    def read_content(filepath: str) -> str:
        """Read and extract content from file"""
        if filepath.lower().endswith('.pdf'):
            return FileProcessor._read_pdf(filepath)
        return FileProcessor._read_text(filepath)
    
    @staticmethod
    def _read_pdf(filepath: str) -> str:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = StringIO()
            for page in pdf_reader.pages:
                text.write(page.extract_text())
            return text.getvalue()
    
    @staticmethod
    def _read_text(filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()

class FlashcardGenerator:
    """Generate and process flashcards"""
    def __init__(self, client):
        self.client = client
        self.unique_cards = FlashcardSet()
    
    def process_chunk(self, chunk: str) -> Generator[FlashcardEvent, None, None]:
        """Process a single chunk and yield flashcard events"""
        try:
            response = self.client.models.generate_content_stream(
                model='gemini-2.0-flash-lite-preview-02-05',
                contents=[{'text': Config.FLASHCARD_GENERATION_PROMPT}, {'text': chunk}],
                config=Config.FLASHCARD_CONFIG
            )
            
            buffer = ""
            for chunk in response:
                if chunk.text:
                    buffer += chunk.text
                    cards = buffer.split('\n')
                    buffer = cards[-1]  # Keep incomplete card
                    
                    for card in cards[:-1]:
                        if 'Q:' in card and '|' in card and 'A:' in card:
                            if self.unique_cards.add(card):  # Only add and emit if unique
                                yield FlashcardEvent('flashcard', card.strip())
            
            # Process any remaining complete card in buffer
            if buffer and 'Q:' in buffer and '|' in buffer and 'A:' in buffer:
                yield FlashcardEvent('flashcard', buffer.strip())
                
        except Exception as e:
            yield FlashcardEvent('error', str(e))
    
    def check_coverage(self, chunk: str, cards: List[str]) -> bool:
        """Check if cards provide sufficient coverage"""
        coverage_prompt = Config.COVERAGE_CHECK_PROMPT.format(
            content=f"Content chunk:\n{chunk}\n\nFlashcards:\n{chr(10).join(cards)}"
        )
        
        coverage_response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=types.Part.from_text(text=coverage_prompt),
            config=Config.COVERAGE_CONFIG
        )
        
        coverage_result = coverage_response.text.strip().lower()
        return coverage_result == 'yes'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def process_file_content_stream(filepath):
    """Process file content with streaming and coverage checks"""
    # Initialize FlashcardGenerator for this request
    generator = FlashcardGenerator(client)
    
    try:
        content = FileProcessor.read_content(filepath)
        chunks = chunk_text(content)
        print(f"Total chunks to process: {len(chunks)}")
        
        chunk_index = 0
        
        while chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            current_batch = []
            chunk_fully_covered = False
            
            while not chunk_fully_covered:
                for event in generator.process_chunk(chunk):
                    if event.type == 'flashcard':
                        current_batch.append(event.content)
                        yield event.to_sse()
                    elif event.type == 'error':
                        raise Exception(event.content)
                
                # Check coverage for this specific chunk
                if current_batch:
                    chunk_fully_covered = generator.check_coverage(chunk, current_batch)
                    event_data = FlashcardEvent(
                        'progress',
                        f'Chunk {chunk_index + 1}/{len(chunks)} coverage: {"yes" if chunk_fully_covered else "no"}'
                    )
                    yield event_data.to_sse()
                else:
                    # If no cards were generated, consider chunk processed
                    chunk_fully_covered = True
            
            # Update progress message
            event_data = FlashcardEvent(
                'progress',
                f'Completed chunk {chunk_index + 1}/{len(chunks)} with {len(current_batch)} unique cards'
            )
            yield event_data.to_sse()
            
            chunk_index += 1
        
        # Send completion event
        event_data = FlashcardEvent(
            'complete',
            f'File processing complete! Generated {len(generator.unique_cards)} unique flashcards from {len(chunks)} chunks.'
        )
        yield event_data.to_sse()
        
    except Exception as e:
        print(f"Error in process_file_content_stream: {str(e)}")
        raise
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def clean_flashcard_text(text: str) -> Optional[str]:
    """Clean and format a single flashcard text"""
    if not text or '|' not in text:
        return None
    parts = text.strip().split('|')
    if len(parts) != 2:
        return None
        
    q_part = parts[0].strip()
    a_part = parts[1].strip()
    
    if not q_part.startswith('Q:') or not a_part.startswith('A:'):
        return None
        
    question = q_part[2:].strip()
    answer = a_part[2:].strip()
    
    if not question or not answer:
        return None
        
    return f"Q: {question} | A: {answer}"

def generate_flashcards_stream(topic, content=None):
    """Generate flashcards from topic or content"""
    # Initialize generator for this request
    generator = FlashcardGenerator(client)
    
    try:
        if content:
            # Handle content-based generation
            print("Processing file content:", content[:100])
            cards = content.split('\n')
            for card in cards:
                cleaned = clean_flashcard_text(card)
                if cleaned:
                    event_data = FlashcardEvent('flashcard', cleaned)
                    yield event_data.to_sse()
                    
            yield FlashcardEvent('complete', f'Generated {len(cards)} flashcards').to_sse()
            return
            
        # Handle topic-based generation
        prompt = Config.FLASHCARD_GENERATION_PROMPT + f"\nTopic: {topic}"
        current_batch = []
        buffer = ""
        
        # Stream flashcard generation
        for chunk in client.models.generate_content_stream(
            model='gemini-2.0-flash-lite-preview-02-05',
            contents=types.Part.from_text(text=prompt),
            config=Config.FLASHCARD_CONFIG
        ):
            if chunk.text:
                buffer += chunk.text
                cards = buffer.split('\n')
                buffer = cards[-1]  # Keep incomplete card
                
                for card in cards[:-1]:
                    cleaned = clean_flashcard_text(card)
                    if cleaned and generator.unique_cards.add(cleaned):
                        yield FlashcardEvent('flashcard', cleaned).to_sse()
                        current_batch.append(cleaned)
        
        # Process any remaining content
        if buffer:
            cleaned = clean_flashcard_text(buffer)
            if cleaned and generator.unique_cards.add(cleaned):
                yield FlashcardEvent('flashcard', cleaned).to_sse()
                current_batch.append(cleaned)
        
        # Check coverage
        if current_batch:
            coverage_prompt = Config.COVERAGE_CHECK_PROMPT.format(
                content=f"Topic: {topic}\n\nFlashcards:\n{chr(10).join(current_batch)}"
            )
            
            coverage_response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=types.Part.from_text(text=coverage_prompt),
                config=Config.COVERAGE_CONFIG
            )
            
            coverage_result = coverage_response.text.strip().lower()
            yield FlashcardEvent('coverage', coverage_result).to_sse()
            
            if coverage_result == 'yes':
                yield FlashcardEvent('complete', 'Coverage target achieved!').to_sse()
            elif len(current_batch) >= Config.MAX_CARDS:
                yield FlashcardEvent('complete', 'Maximum flashcard limit reached').to_sse()
                
    except Exception as e:
        print(f"Error in generate_flashcards_stream: {str(e)}")
        yield FlashcardEvent('error', str(e)).to_sse()

def init_app():
    """Initialize application requirements"""
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    app.secret_key = os.urandom(24)

# Initialize app
app = Flask(__name__)
init_app()

@app.route('/upload-file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Create a safe filename and ensure upload directory exists
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        
        # Ensure upload directory exists
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save file safely
        try:
            file.save(filepath)
        except Exception as e:
            print(f"File save error: {str(e)}")
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Store filepath in session
        session['uploaded_file'] = filepath
        return jsonify({'success': True, 'filename': filename})
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stream-generate', methods=['GET'])
def stream_generate():
    # Add CORS headers for SSE
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    }
    
    is_file_request = request.args.get('file') == 'true'
    
    if is_file_request and 'uploaded_file' in session:
        filepath = session.pop('uploaded_file', None)
        if filepath and os.path.exists(filepath):
            try:
                return Response(
                    process_file_content_stream(filepath),
                    mimetype='text/event-stream',
                    headers=headers
                )
            except Exception as e:
                print(f"Error in stream_generate: {str(e)}")
                return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'No valid file found'}), 400
    
    # Handle regular topic-based generation
    topic = request.args.get('topic')
    if not topic:
        return jsonify({'error': 'Topic is required'}), 400
        
    return Response(
        generate_flashcards_stream(topic),
        mimetype='text/event-stream',
        headers=headers
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    topic = request.form['topic']
    try:
        flashcards = generate_flashcards(topic)
        return jsonify({'flashcards': flashcards})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)