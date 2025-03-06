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

# For Vercel deployment
import tempfile
import hashlib
import time

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
    # Use temporary directory for Vercel
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'flashcard_uploads')
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

def process_file_content_batch(filepath):
    """Process file content in batch mode instead of streaming"""
    # Initialize FlashcardGenerator for this request
    generator = FlashcardGenerator(client)
    
    try:
        content = FileProcessor.read_content(filepath)
        chunks = chunk_text(content)
        print(f"Total chunks to process: {len(chunks)}")
        
        all_flashcards = []
        
        for i, chunk in enumerate(chunks):
            # Process chunk
            cards_for_chunk = []
            
            # Generate flashcards for this chunk
            response = client.models.generate_content(
                model='gemini-2.0-flash-lite',
                contents=[{'text': Config.FLASHCARD_GENERATION_PROMPT}, {'text': chunk}],
                config=Config.FLASHCARD_CONFIG
            )
            
            # Extract flashcards
            raw_cards = response.text.split('\n')
            for card in raw_cards:
                if 'Q:' in card and '|' in card and 'A:' in card:
                    cleaned = clean_flashcard_text(card)
                    if cleaned and generator.unique_cards.add(cleaned):
                        cards_for_chunk.append(cleaned)
            
            all_flashcards.extend(cards_for_chunk)
            print(f"Processed chunk {i+1}/{len(chunks)}: {len(cards_for_chunk)} cards")
        
        return {
            'flashcards': all_flashcards,
            'count': len(generator.unique_cards),
            'chunks': len(chunks)
        }
        
    except Exception as e:
        print(f"Error in process_file_content_batch: {str(e)}")
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

def generate_flashcards_batch(topic):
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
        
        # Check coverage
        has_coverage = False
        if all_flashcards:
            coverage_prompt = Config.COVERAGE_CHECK_PROMPT.format(
                content=f"Topic: {topic}\n\nFlashcards:\n{chr(10).join(all_flashcards)}"
            )
            
            coverage_response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=types.Part.from_text(text=coverage_prompt),
                config=Config.COVERAGE_CONFIG
            )
            
            has_coverage = coverage_response.text.strip().lower() == 'yes'
        
        return {
            'flashcards': all_flashcards,
            'count': len(generator.unique_cards),
            'coverage': has_coverage
        }
                
    except Exception as e:
        print(f"Error in generate_flashcards_batch: {str(e)}")
        raise

def init_app():
    """Initialize application requirements"""
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Initialize app
app = Flask(__name__)
init_app()

class ProcessingState:
    """Store processing state between requests"""
    @staticmethod
    def get_file_key(filepath):
        """Generate a unique key for a file"""
        return hashlib.md5(filepath.encode('utf-8')).hexdigest()
    
    @staticmethod
    def init_file_state(filepath):
        """Initialize processing state for a file"""
        content = FileProcessor.read_content(filepath)
        chunks = chunk_text(content)
        file_key = ProcessingState.get_file_key(filepath)
        
        state = {
            'filepath': filepath,
            'chunks': chunks,
            'total_chunks': len(chunks),
            'processed_chunks': [],
            'current_index': 0,
            'all_flashcards': [],
            'is_complete': False,
            'has_coverage': False,
            'last_updated': time.time()
        }
        
        session[file_key] = state
        return file_key
    
    @staticmethod
    def get_state(file_key):
        """Get processing state for a file"""
        return session.get(file_key)
    
    @staticmethod
    def update_state(file_key, updates):
        """Update processing state for a file"""
        if file_key in session:
            state = session[file_key]
            for key, value in updates.items():
                state[key] = value
            state['last_updated'] = time.time()
            session[file_key] = state
            return True
        return False
    
    @staticmethod
    def cleanup_old_states(max_age=3600):  # 1 hour
        """Remove old processing states"""
        now = time.time()
        keys_to_remove = []
        
        for key in list(session.keys()):
            if isinstance(session[key], dict) and 'last_updated' in session[key]:
                if now - session[key]['last_updated'] > max_age:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if session.get(key, {}).get('filepath') and os.path.exists(session[key]['filepath']):
                try:
                    os.remove(session[key]['filepath'])
                except:
                    pass
            del session[key]

def process_file_chunk_batch(file_key, chunk_index):
    """Process a single chunk of a file in batch mode"""
    state = ProcessingState.get_state(file_key)
    if not state or state['is_complete'] or chunk_index >= len(state['chunks']):
        return {'error': 'Invalid state or chunk index'}
    
    # Initialize generator for this chunk
    generator = FlashcardGenerator(client)
    chunk = state['chunks'][chunk_index]
    
    try:
        # Generate flashcards for this chunk
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=[{'text': Config.FLASHCARD_GENERATION_PROMPT}, {'text': chunk}],
            config=Config.FLASHCARD_CONFIG
        )
        
        # Extract flashcards
        chunk_flashcards = []
        raw_cards = response.text.split('\n')
        
        for card in raw_cards:
            if 'Q:' in card and '|' in card and 'A:' in card:
                cleaned = clean_flashcard_text(card)
                if cleaned:
                    chunk_flashcards.append(cleaned)
        
        # Update state
        all_cards = state['all_flashcards'] + chunk_flashcards
        state['processed_chunks'].append(chunk_index)
        state['all_flashcards'] = all_cards
        state['current_index'] = chunk_index + 1
        
        if len(state['processed_chunks']) == state['total_chunks']:
            state['is_complete'] = True
        
        ProcessingState.update_state(file_key, state)
        
        return {
            'flashcards': chunk_flashcards,
            'all_flashcards_count': len(all_cards),
            'chunk_index': chunk_index,
            'total_chunks': state['total_chunks'],
            'is_complete': state['is_complete']
        }
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return {'error': str(e)}

def check_file_coverage(file_key):
    """Check if the current flashcards provide sufficient coverage"""
    state = ProcessingState.get_state(file_key)
    if not state:
        return {'error': 'Invalid file key'}
    
    if len(state['all_flashcards']) == 0:
        return {'coverage': False, 'message': 'No flashcards generated yet'}
    
    try:
        # Use a sample of content to check coverage
        content_sample = "\n".join([state['chunks'][i] for i in state['processed_chunks'][:5]])
        
        coverage_prompt = Config.COVERAGE_CHECK_PROMPT.format(
            content=f"Content sample:\n{content_sample}\n\nFlashcards:\n{chr(10).join(state['all_flashcards'][:100])}"
        )
        
        coverage_response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=types.Part.from_text(text=coverage_prompt),
            config=Config.COVERAGE_CONFIG
        )
        
        has_coverage = coverage_response.text.strip().lower() == 'yes'
        
        # Update state
        state['has_coverage'] = has_coverage
        ProcessingState.update_state(file_key, state)
        
        return {
            'coverage': has_coverage,
            'message': 'Coverage is sufficient' if has_coverage else 'Additional flashcards needed'
        }
        
    except Exception as e:
        print(f"Error checking coverage: {str(e)}")
        return {'error': str(e)}

def get_file_state(file_key):
    """Get current processing state for a file"""
    state = ProcessingState.get_state(file_key)
    if not state:
        return {'error': 'Invalid file key'}
    
    return {
        'total_chunks': state['total_chunks'],
        'processed_chunks': len(state['processed_chunks']),
        'current_index': state['current_index'],
        'flashcard_count': len(state['all_flashcards']),
        'is_complete': state['is_complete'],
        'has_coverage': state['has_coverage'],
        'next_chunk': state['current_index'] if state['current_index'] < state['total_chunks'] else None
    }

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
        
        # Initialize processing state
        file_key = ProcessingState.init_file_state(filepath)
        
        # Clean up old processing states
        ProcessingState.cleanup_old_states()
        
        return jsonify({
            'success': True, 
            'file_key': file_key, 
            'filename': filename
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate flashcards from a topic"""
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
            
        result = generate_flashcards_batch(topic)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-from-file', methods=['POST'])
def generate_from_file():
    """Generate flashcards from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)
        
        # Process file
        result = process_file_content_batch(filepath)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate_from_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate-chunk', methods=['POST'])
def generate_chunk():
    """Generate flashcards from a specific chunk of a file"""
    try:
        data = request.get_json()
        file_key = data.get('file_key')
        chunk_index = data.get('chunk_index')
        
        if not file_key:
            return jsonify({'error': 'File key is required'}), 400
        
        state = ProcessingState.get_state(file_key)
        if not state:
            return jsonify({'error': 'Invalid file key'}), 400
        
        if chunk_index is None:
            chunk_index = state['current_index']
        
        if chunk_index >= state['total_chunks']:
            return jsonify({
                'message': 'All chunks processed',
                'is_complete': True
            })
        
        result = process_file_chunk_batch(file_key, chunk_index)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate_chunk: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-coverage', methods=['POST'])
def check_coverage():
    """Check if the current flashcards provide sufficient coverage"""
    try:
        data = request.get_json()
        file_key = data.get('file_key')
        
        if not file_key:
            return jsonify({'error': 'File key is required'}), 400
        
        result = check_file_coverage(file_key)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in check_coverage: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/file-state', methods=['GET'])
def file_state():
    """Get current processing state for a file"""
    file_key = request.args.get('file_key')
    
    if not file_key:
        return jsonify({'error': 'File key is required'}), 400
    
    result = get_file_state(file_key)
    return jsonify(result)

@app.route('/')
def home():
    return render_template('index.html')

# Vercel requires the app to be named 'app'
if __name__ == '__main__':
    app.run(debug=False)