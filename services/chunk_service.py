from models import FlashcardGenerator
from config import Config
from utils import clean_flashcard_text
from services.storage_service import ProcessingState

def process_file_chunk_batch(client, file_key, chunk_index):
    """Process a single chunk of a file in batch mode"""
    state = ProcessingState.get_state(file_key)
    if not state or state['is_complete'] or chunk_index >= state['total_chunks']:
        return {'error': 'Invalid state or chunk index'}
    
    # Get chunk content
    chunk = ProcessingState.get_chunk(file_key, chunk_index)
    if not chunk:
        return {'error': f'Cannot read chunk {chunk_index}'}
    
    # Initialize generator for this chunk
    generator = FlashcardGenerator(client)
    
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
        ProcessingState.append_flashcards(file_key, chunk_flashcards)
        
        # Update processing state
        state['processed_chunks'].append(chunk_index)
        state['current_index'] = chunk_index + 1
        
        if len(state['processed_chunks']) == state['total_chunks']:
            state['is_complete'] = True
        
        ProcessingState.update_state(file_key, state)
        
        # Get total flashcards count
        all_flashcards = ProcessingState.get_all_flashcards(file_key)
        
        return {
            'flashcards': chunk_flashcards,
            'all_flashcards_count': len(all_flashcards),
            'chunk_index': chunk_index,
            'total_chunks': state['total_chunks'],
            'is_complete': state['is_complete']
        }
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {str(e)}")
        return {'error': str(e)}

def get_file_state(file_key):
    """Get current processing state for a file"""
    state = ProcessingState.get_state(file_key)
    if not state:
        return {'error': 'Invalid file key'}
    
    all_flashcards = ProcessingState.get_all_flashcards(file_key)
    
    return {
        'total_chunks': state['total_chunks'],
        'processed_chunks': len(state['processed_chunks']),
        'current_index': state['current_index'],
        'flashcard_count': len(all_flashcards),
        'is_complete': state['is_complete'],
        'next_chunk': state['current_index'] if state['current_index'] < state['total_chunks'] else None
    }
