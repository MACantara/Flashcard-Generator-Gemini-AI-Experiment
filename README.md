# AI Flashcard Generator

An interactive web application that generates study flashcards using Google's Gemini AI. The application can create flashcards from both topics and uploaded documents (PDF/TXT).

## Features

- Generate flashcards from any topic
- Upload PDF or TXT files for flashcard creation
- Real-time streaming generation
- Duplicate detection and removal
- Coverage analysis for comprehensive learning
- Interactive UI with Bootstrap styling

## Prerequisites

- Python 3.9+
- Google Gemini AI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Flashcard-Generator-Gemini-AI-Experiment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Gemini API key:
```env
GOOGLE_GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Generate flashcards by either:
   - Entering a topic in the "By Topic" tab
   - Uploading a PDF or TXT file in the "From File" tab

## How It Works

1. **Topic-based Generation:**
   - Uses Gemini AI to generate relevant questions and answers
   - Streams results in real-time
   - Checks coverage to ensure comprehensive topic understanding

2. **File-based Generation:**
   - Processes uploaded files in chunks
   - Extracts key information using Gemini AI
   - Generates targeted flashcards from content
   - Ensures complete coverage of the material

## Project Structure

- `app.py` - Main Flask application and AI logic
- `templates/index.html` - Frontend interface
- `requirements.txt` - Python dependencies
- `temp_uploads/` - Temporary storage for uploaded files

## Technical Details

- Built with Flask for the backend
- Uses Server-Sent Events (SSE) for real-time updates
- Implements chunked processing for large files
- Uses Google Gemini AI for content generation
- Bootstrap 5 for responsive design

## Limitations

- Maximum file size depends on available memory
- Processing time varies with content length
- API rate limits may apply
- Currently supports only TXT and PDF files

## Future Improvements

- [ ] Add support for more file formats
- [ ] Implement user accounts and saved flashcards
- [ ] Add export functionality (PDF, CSV)
- [ ] Improve error handling and recovery
- [ ] Add spaced repetition learning features

## Contributing

This is a prototype project. Feel free to fork and experiment with improvements.

## License

MIT License - See LICENSE file for details
