import os
import logging
from typing import List, Dict
from pathlib import Path
import PyPDF2
from docx import Document
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'token_count': len(chunk_tokens)
            })
        
        return chunks
    
    def process_documents(self, document_paths: List[str]) -> List[Dict]:
        """Process multiple documents and return chunks"""
        all_chunks = []
        
        for doc_path in document_paths:
            logger.info(f"Processing document: {doc_path}")
            text = self.extract_text(doc_path)
            
            if text:
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    chunk['source'] = doc_path
                    chunk['document_id'] = len(all_chunks)
                all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(document_paths)} documents into {len(all_chunks)} chunks")
        return all_chunks
