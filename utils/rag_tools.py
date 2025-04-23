import os
import tempfile
from typing import List, Dict, Any, Optional, Union
import urllib.parse
from datetime import datetime
import json
from config.config import GEMINI_API_KEY

# CrewAI Tools imports
try:
    from crewai_tools import RagTool, PDFSearchTool, YoutubeVideoSearchTool, WebsiteSearchTool
except ImportError:
    pass

class RagToolManager:
    """
    Manages RAG tools from CrewAI for various data types.
    """
    
    def __init__(self):
        """Initialize the RAG tool manager."""
        # Configure RAG with custom embeddings using Gemini
        rag_config = {
            "app": {
                "name": "multimodal_rag",
            },
            "llm": {
                "provider": "google",
                "config": {
                    "model": "gemini-1.5-pro",
                    "api_key": GEMINI_API_KEY
                }
            },
            "embedding_model": {
                "provider": "google",
                "config": {
                    "model": "models/embedding-001",
                    "task_type": "retrieval_document",
                    "api_key": GEMINI_API_KEY
                }
            }
        }
        
        # Main RAG tool for general content
        self.rag_tool = RagTool(config=rag_config)
        
        # Specialized tools for different content types
        self.pdf_tool = PDFSearchTool(config=rag_config)
        self.youtube_tool = YoutubeVideoSearchTool(config=rag_config)
        self.website_tool = WebsiteSearchTool(config=rag_config)
        
        # Keep track of added content
        self.knowledge_base = {
            'documents': [],
            'web_pages': [],
            'youtube_videos': [],
            'images': [],
            'audio': [],
            'video': []
        }
    
    def add_to_knowledge_base(self, file_path: str) -> bool:
        """
        Add a file to the knowledge base.
        
        Args:
            file_path (str): Path to the file to add
            
        Returns:
            bool: Success status
        """
        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip('.').lower()
            
            # Process based on file type
            if ext == 'pdf':
                self._add_pdf(file_path)
            elif ext in ['txt', 'docx', 'pptx']:
                self._add_document(file_path)
            elif ext in ['png', 'jpg', 'jpeg']:
                self._add_image(file_path)
            elif ext in ['mp3', 'wav']:
                self._add_audio(file_path)
            elif ext in ['mp4']:
                self._add_video(file_path)
            else:
                # Add as generic file
                self._add_document(file_path)
            
            return True
        except Exception as e:
            print(f"Error adding file to knowledge base: {str(e)}")
            return False
    
    def add_web_content(self, url: str) -> bool:
        """
        Add web content to the knowledge base.
        
        Args:
            url (str): URL of the web content to add
            
        Returns:
            bool: Success status
        """
        try:
            # Determine if it's a YouTube video
            is_youtube = 'youtube.com' in url or 'youtu.be' in url
            
            if is_youtube:
                self._add_youtube_video(url)
            else:
                self._add_website(url)
            
            return True
        except Exception as e:
            print(f"Error adding web content to knowledge base: {str(e)}")
            return False
    
    def _add_pdf(self, file_path: str):
        """Add a PDF to the knowledge base."""
        # Add to the general RAG tool
        self.rag_tool.add(data_type="file", path=file_path)
        
        # Also add to the specialized PDF tool
        self.pdf_tool = PDFSearchTool(pdf=file_path)
        
        # Log the addition
        self.knowledge_base['documents'].append({
            'type': 'pdf',
            'path': file_path,
            'filename': os.path.basename(file_path),
            'added_at': datetime.now().isoformat()
        })
    
    def _add_document(self, file_path: str):
        """Add a document to the knowledge base."""
        # Add to the general RAG tool
        self.rag_tool.add(data_type="file", path=file_path)
        
        # Log the addition
        self.knowledge_base['documents'].append({
            'type': os.path.splitext(file_path)[1].lstrip('.').lower(),
            'path': file_path,
            'filename': os.path.basename(file_path),
            'added_at': datetime.now().isoformat()
        })
    
    def _add_image(self, file_path: str):
        """Add an image to the knowledge base."""
        # For images, we don't directly add them to RAG
        # but keep track of them for multimodal queries
        self.knowledge_base['images'].append({
            'path': file_path,
            'filename': os.path.basename(file_path),
            'added_at': datetime.now().isoformat()
        })
    
    def _add_audio(self, file_path: str):
        """Add audio to the knowledge base."""
        # For audio, we don't directly add them to RAG
        # but keep track of them for multimodal queries
        self.knowledge_base['audio'].append({
            'path': file_path,
            'filename': os.path.basename(file_path),
            'added_at': datetime.now().isoformat()
        })
    
    def _add_video(self, file_path: str):
        """Add video to the knowledge base."""
        # For videos, we don't directly add them to RAG
        # but keep track of them for multimodal queries
        self.knowledge_base['video'].append({
            'path': file_path,
            'filename': os.path.basename(file_path),
            'added_at': datetime.now().isoformat()
        })
    
    def _add_youtube_video(self, url: str):
        """Add a YouTube video to the knowledge base."""
        # Add to the specialized YouTube tool
        self.youtube_tool = YoutubeVideoSearchTool(youtube_video_url=url)
        
        # Log the addition
        self.knowledge_base['youtube_videos'].append({
            'url': url,
            'added_at': datetime.now().isoformat()
        })
    
    def _add_website(self, url: str):
        """Add a website to the knowledge base."""
        # Add to the general RAG tool
        self.rag_tool.add(data_type="web_page", url=url)
        
        # Add to the specialized Website tool
        self.website_tool = WebsiteSearchTool(website=url)
        
        # Log the addition
        self.knowledge_base['web_pages'].append({
            'url': url,
            'added_at': datetime.now().isoformat()
        })
    
    def query(self, query_text: str) -> str:
        """
        Query the knowledge base using appropriate RAG tools.
        
        Args:
            query_text (str): The query text
            
        Returns:
            str: Combined results from all relevant RAG tools
        """
        results = []
        
        # Use general RAG tool if we have added content
        if (self.knowledge_base['documents'] or 
            self.knowledge_base['web_pages']):
            try:
                rag_result = self.rag_tool.run(query=query_text)
                if rag_result and rag_result.strip():
                    results.append(f"### General Knowledge Base Results\n{rag_result}")
            except Exception as e:
                print(f"Error querying general RAG tool: {str(e)}")
        
        # Use PDF tool if we have PDFs
        if self.knowledge_base['documents']:
            pdf_docs = [doc for doc in self.knowledge_base['documents'] if doc['type'] == 'pdf']
            if pdf_docs:
                try:
                    for pdf_doc in pdf_docs[:3]:  # Limit to 3 PDFs for performance
                        pdf_result = self.pdf_tool.run(query=query_text, pdf=pdf_doc['path'])
                        if pdf_result and pdf_result.strip():
                            results.append(f"### PDF Results ({pdf_doc['filename']})\n{pdf_result}")
                except Exception as e:
                    print(f"Error querying PDF tool: {str(e)}")
        
        # Use YouTube tool if we have YouTube videos
        if self.knowledge_base['youtube_videos']:
            try:
                yt_result = self.youtube_tool.run(search_query=query_text)
                if yt_result and yt_result.strip():
                    results.append(f"### YouTube Video Results\n{yt_result}")
            except Exception as e:
                print(f"Error querying YouTube tool: {str(e)}")
        
        # Use Website tool if we have websites
        if self.knowledge_base['web_pages']:
            try:
                web_result = self.website_tool.run(search_query=query_text)
                if web_result and web_result.strip():
                    results.append(f"### Website Results\n{web_result}")
            except Exception as e:
                print(f"Error querying Website tool: {str(e)}")
        
        # Combine results
        if results:
            return "\n\n".join(results)
        else:
            return "No relevant information found in the knowledge base." 