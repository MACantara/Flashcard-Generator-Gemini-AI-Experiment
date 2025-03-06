from io import StringIO
import PyPDF2
from typing import Dict

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
