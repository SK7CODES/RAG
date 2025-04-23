"""
Simple document retrieval module to test basic functionality.
This module doesn't rely on CrewAI and provides a simplified way to:
1. Extract text from PDF, DOCX, and TXT files
2. Create embeddings 
3. Allow for retrieval of similar content
"""

import os
import sys
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
import json
from datetime import datetime
import google.generativeai as genai
import PyPDF2
import pdfplumber
import docx2txt
import docx
import time

# For embeddings and similarity
from sentence_transformers import SentenceTransformer

class SimpleDocRetrieval:
    """A simplified document retrieval system using Gemini API"""
    
    def __init__(self, api_key: str):
        """Initialize the document retrieval system with the Gemini API key."""
        self.api_key = api_key
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create a temporary directory for storing documents
        self.temp_dir = os.path.join(tempfile.gettempdir(), "simple_doc_retrieval")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Storage for document content
        self.documents = {}
        
        # Text chunking parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from a file based on its extension."""
        try:
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()
            
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            else:
                print(f"Unsupported file extension: {file_extension}")
                return None
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error extracting text from TXT {file_path}: {str(e)}")
            return ""
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {str(e)}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text:
            return []
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
        
        return chunks
    
    def add_document(self, file_path: str) -> bool:
        """Add a document to the retrieval system."""
        try:
            # Extract text from the document
            document_text = self.extract_text_from_file(file_path)
            
            if not document_text:
                print(f"No text could be extracted from {file_path}")
                return False
            
            # Chunk the document text
            chunks = self.chunk_text(document_text)
            
            # Add to documents dictionary
            file_name = os.path.basename(file_path)
            self.documents[file_name] = {
                "path": file_path,
                "chunks": chunks,
                "full_text": document_text
            }
            
            print(f"Successfully added document: {file_name} with {len(chunks)} chunks")
            return True
        
        except Exception as e:
            print(f"Error adding document {file_path}: {str(e)}")
            return False
    
    def query(self, question: str) -> str:
        """Query the documents with a question using Gemini API."""
        if not self.documents:
            return "No documents have been added to the system."
        
        try:
            # Prepare a context with document content
            context = "Here are excerpts from relevant documents:\n\n"
            
            # Include content from all documents (this is a simplified approach)
            for doc_name, doc_data in self.documents.items():
                # Include first chunk and last chunk to provide a sample of content
                if doc_data["chunks"]:
                    context += f"Document: {doc_name}\n"
                    
                    # Add an excerpt from the beginning
                    context += "Excerpt from beginning:\n"
                    context += doc_data["chunks"][0][:500] + "...\n\n"
                    
                    # Add an excerpt from the end if there are multiple chunks
                    if len(doc_data["chunks"]) > 1:
                        context += "Excerpt from end:\n"
                        context += "..." + doc_data["chunks"][-1][-500:] + "\n\n"
            
            # Create a prompt for Gemini
            prompt = f"""
            I'll provide you with some context from documents, and then ask a question.
            Please answer the question based on the context provided.
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer. If the information is not found in the context, 
            state that clearly and give a best guess based on your general knowledge.
            """
            
            # Query Gemini
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error querying documents: {str(e)}"
    
    def clear(self):
        """Clear all documents from the system."""
        self.documents = {}
        return True 