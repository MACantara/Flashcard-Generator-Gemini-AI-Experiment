import os
import hashlib
import time
import pickle
import shutil
from flask import session

from config import Config
from services.file_service import FileProcessor
from utils import chunk_text

class ProcessingState:
    """Store processing state between requests"""
    @staticmethod
    def get_file_key(filepath):
        """Generate a unique key for a file"""
        return hashlib.md5(filepath.encode('utf-8')).hexdigest()
    
    @staticmethod
    def init_file_state(filepath):
        """Initialize processing state for a file"""
        try:
            # Create a unique directory for this file processing
            file_key = ProcessingState.get_file_key(filepath)
            state_dir = os.path.join(Config.UPLOAD_FOLDER, file_key)
            os.makedirs(state_dir, exist_ok=True)
            
            # Read content and divide into chunks
            content = FileProcessor.read_content(filepath)
            chunks = chunk_text(content)
            
            # Store chunks in separate files
            chunks_dir = os.path.join(state_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            for i, chunk in enumerate(chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{i}.txt")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk)
            
            # Create a lightweight state object for the session
            state = {
                'file_key': file_key,
                'state_dir': state_dir,
                'total_chunks': len(chunks),
                'processed_chunks': [],
                'current_index': 0,
                'is_complete': False,
                'last_updated': time.time()
            }
            
            # Store lightweight state in session
            session[file_key] = state
            
            return file_key
        except Exception as e:
            print(f"Error in init_file_state: {str(e)}")
            # Clean up any partial files
            if 'state_dir' in locals() and os.path.exists(state_dir):
                shutil.rmtree(state_dir, ignore_errors=True)
            raise
    
    @staticmethod
    def get_state(file_key):
        """Get processing state for a file"""
        return session.get(file_key)
    
    @staticmethod
    def get_chunk(file_key, chunk_index):
        """Get a specific chunk content"""
        state = ProcessingState.get_state(file_key)
        if not state:
            return None
        
        chunk_file = os.path.join(state['state_dir'], "chunks", f"chunk_{chunk_index}.txt")
        if not os.path.exists(chunk_file):
            return None
            
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def update_state(file_key, updates):
        """Update processing state for a file"""
        if file_key in session:
            state = session[file_key]
            
            # Handle flashcards separately to prevent session bloat
            if 'all_flashcards' in updates:
                flashcards = updates.pop('all_flashcards')
                # Save flashcards to file
                flashcards_file = os.path.join(state['state_dir'], "flashcards.pkl")
                with open(flashcards_file, 'wb') as f:
                    pickle.dump(flashcards, f)
                
            # Update other state properties
            for key, value in updates.items():
                state[key] = value
                
            state['last_updated'] = time.time()
            session[file_key] = state
            return True
        return False
    
    @staticmethod
    def get_all_flashcards(file_key):
        """Get all flashcards for a file"""
        state = ProcessingState.get_state(file_key)
        if not state:
            return []
            
        flashcards_file = os.path.join(state['state_dir'], "flashcards.pkl")
        if os.path.exists(flashcards_file):
            with open(flashcards_file, 'rb') as f:
                return pickle.load(f)
        return []
    
    @staticmethod
    def append_flashcards(file_key, new_flashcards):
        """Append new flashcards to existing ones"""
        state = ProcessingState.get_state(file_key)
        if not state:
            return False
            
        current = ProcessingState.get_all_flashcards(file_key)
        updated = current + new_flashcards
        
        flashcards_file = os.path.join(state['state_dir'], "flashcards.pkl")
        with open(flashcards_file, 'wb') as f:
            pickle.dump(updated, f)
        
        return True
    
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
            state = session[key]
            if 'state_dir' in state and os.path.exists(state['state_dir']):
                try:
                    shutil.rmtree(state['state_dir'], ignore_errors=True)
                except:
                    pass
            del session[key]
