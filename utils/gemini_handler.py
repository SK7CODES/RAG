import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import mimetypes

class GeminiHandler:
    """
    Handles interactions with the Gemini API for text and multimodal processing.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemini handler with API key.
        
        Args:
            api_key (str): Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Available models
        self.text_model = "gemini-1.5-pro"
        self.multimodal_model = "gemini-1.5-pro-vision"
        
        # Initialize models
        self.text_generation_model = genai.GenerativeModel(self.text_model)
        self.multimodal_generation_model = genai.GenerativeModel(self.multimodal_model)
    
    def generate_text_response(self, prompt: str, rag_manager=None) -> str:
        """
        Generate a text response using Gemini API with optional RAG context.
        
        Args:
            prompt (str): User query text
            rag_manager: RAG tool manager for retrieving context
            
        Returns:
            str: Generated response text
        """
        try:
            # Get RAG context if available
            context = ""
            if rag_manager and hasattr(rag_manager, 'query'):
                context = rag_manager.query(prompt)
            
            # Build the complete prompt with RAG context
            if context:
                complete_prompt = f"""
                Here is some relevant context information: 
                {context}
                
                Based on this context and your knowledge, please answer the following question:
                {prompt}
                """
            else:
                complete_prompt = prompt
            
            # Generate text response
            response = self.text_generation_model.generate_content(complete_prompt)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_multimodal_response(self, text_prompt: str, media_path: str, rag_manager=None) -> str:
        """
        Generate a multimodal response using Gemini API with optional RAG context.
        
        Args:
            text_prompt (str): User query text
            media_path (str): Path to the media file
            rag_manager: RAG tool manager for retrieving context
            
        Returns:
            str: Generated response text
        """
        try:
            # Get content type
            content_type = mimetypes.guess_type(media_path)[0]
            if not content_type:
                # Fallback based on extension
                ext = os.path.splitext(media_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    content_type = f'image/{ext[1:]}'
                elif ext in ['.mp3', '.wav']:
                    content_type = f'audio/{ext[1:]}'
                elif ext in ['.mp4']:
                    content_type = 'video/mp4'
                else:
                    content_type = 'application/octet-stream'
            
            # Read the binary file
            with open(media_path, 'rb') as f:
                media_bytes = f.read()
            
            # Get RAG context if available
            context = ""
            if rag_manager and hasattr(rag_manager, 'query'):
                # If text prompt is empty, use a default one
                query_text = text_prompt if text_prompt else "Analyze this media"
                context = rag_manager.query(query_text)
            
            # Build the complete prompt with RAG context
            if context:
                prompt_text = f"""
                Here is some relevant context information: 
                {context}
                
                Based on this context and the provided media, please answer the following:
                {text_prompt if text_prompt else "Analyze and describe what you see in this media."}
                """
            else:
                prompt_text = text_prompt if text_prompt else "Analyze and describe what you see in this media."
            
            # Create multimodal prompt
            if content_type.startswith('image/'):
                multimodal_prompt = [
                    prompt_text,
                    {
                        "mime_type": content_type,
                        "data": media_bytes
                    }
                ]
            elif content_type.startswith(('audio/', 'video/')):
                multimodal_prompt = [
                    prompt_text,
                    {
                        "mime_type": content_type,
                        "data": media_bytes
                    }
                ]
            else:
                return "Unsupported media type."
            
            # Generate multimodal response
            response = self.multimodal_generation_model.generate_content(multimodal_prompt)
            
            return response.text
        except Exception as e:
            return f"Error processing multimodal query: {str(e)}"
    
    def create_tool_calling_model(self, tools: List[Dict[str, Any]]):
        """
        Create a model that can use tool calling capabilities.
        
        Args:
            tools (List[Dict[str, Any]]): List of tool definitions
            
        Returns:
            GenerativeModel: Model with tool calling capabilities
        """
        return genai.GenerativeModel(
            self.text_model,
            tools=tools
        ) 