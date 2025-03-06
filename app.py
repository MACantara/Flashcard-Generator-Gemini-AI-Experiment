from flask import Flask, render_template, request, jsonify, Response, session
from google import genai
import os
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename

# Import from project modules
from config import Config
from utils import allowed_file
from services.file_service import FileProcessor
from services.flashcard_service import generate_flashcards_batch
from services.storage_service import ProcessingState
from services.chunk_service import process_file_chunk_batch, get_file_state

app = Flask(__name__)
load_dotenv()

# Configure Gemini AI
client = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))

def init_app():
    """Initialize application requirements"""
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Initialize app
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
        
        # Initialize processing state
        file_key = ProcessingState.init_file_state(filepath)
        state = ProcessingState.get_state(file_key)
        
        # Clean up old processing states
        ProcessingState.cleanup_old_states()
        
        return jsonify({
            'success': True, 
            'file_key': file_key, 
            'filename': filename,
            'total_chunks': state['total_chunks']
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
            
        result = generate_flashcards_batch(client, topic)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate: {str(e)}")
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
        
        result = process_file_chunk_batch(client, file_key, chunk_index)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate_chunk: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/file-state', methods=['GET'])
def file_state():
    """Get current processing state for a file"""
    file_key = request.args.get('file_key')
    
    if not file_key:
        return jsonify({'error': 'File key is required'}), 400
    
    result = get_file_state(file_key)
    return jsonify(result)

@app.route('/all-file-flashcards', methods=['GET'])
def all_file_flashcards():
    """Get all flashcards for a file"""
    file_key = request.args.get('file_key')
    
    if not file_key:
        return jsonify({'error': 'File key is required'}), 400
    
    all_flashcards = ProcessingState.get_all_flashcards(file_key)
    
    return jsonify({
        'flashcards': all_flashcards,
        'count': len(all_flashcards)
    })

@app.route('/')
def home():
    return render_template('index.html')

# Vercel requires the app to be named 'app'
if __name__ == '__main__':
    app.run(debug=False)