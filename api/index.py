import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app from main application file
from app import app

# This is the entry point for Vercel
if __name__ == "__main__":
    app.run()
