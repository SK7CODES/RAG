import streamlit as st
import os
import sys
import tempfile
import base64
import json
import numpy as np
from PIL import Image
import google.generativeai as genai
from datetime import datetime
from simple_doc_retrieval import SimpleDocRetrieval

# Page Configuration
st.set_page_config(
    page_title="AI Assistant with Document Processing",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme settings
st.markdown("""
<style>
    /* Global text color */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, div {
        color: #333333 !important;
    }
    
    /* Make sure text inputs have dark text on light background */
    .stTextInput input, .stTextArea textarea {
        color: #333333 !important;
        background-color: #FFFFFF !important;
    }
    
    /* Fix button text */
    .stButton button {
        color: #FFFFFF !important;
        background-color: #0066CC !important;
        width: 100%;
    }
    
    /* Chat message styling with improved contrast */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        color: #333333 !important;
    }
    .chat-message.assistant {
        background-color: #e6f7ff;
        color: #333333 !important;
    }
    .chat-message .message-content {
        display: flex;
        flex-direction: column;
        color: #333333 !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .file-uploader {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {
        "documents": [],
        "images": [],
        "audio": [],
        "video": [],
        "web_content": []
    }
    
if "doc_retrieval" not in st.session_state:
    st.session_state.doc_retrieval = None
    
if "gemini_handler" not in st.session_state:
    st.session_state.gemini_handler = None

# Gemini Handler Class
class GeminiHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def generate_response(self, prompt, image_data=None):
        try:
            if image_data:
                response = self.model.generate_content([prompt, image_data])
            else:
                response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Sidebar for API Key and File Upload
with st.sidebar:
    st.title("AI Assistant Settings")
    
    # API Key Input
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    if api_key:
        if st.session_state.gemini_handler is None or st.session_state.gemini_handler.api_key != api_key:
            st.session_state.gemini_handler = GeminiHandler(api_key)
            st.session_state.doc_retrieval = SimpleDocRetrieval(api_key)
            st.success("API Key set successfully!")
    
    st.divider()
    
    # Document Processing Section
    st.subheader("Upload Documents")
    uploaded_docs = st.file_uploader(
        "Upload PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True,
        key="document_uploader"
    )
    
    if uploaded_docs:
        for uploaded_file in uploaded_docs:
            if uploaded_file.name not in [doc["name"] for doc in st.session_state.processed_files["documents"]]:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the document
                if st.session_state.doc_retrieval:
                    success = st.session_state.doc_retrieval.add_document(temp_file_path)
                    if success:
                        st.session_state.processed_files["documents"].append({
                            "name": uploaded_file.name,
                            "path": temp_file_path,
                            "type": uploaded_file.type,
                            "added_at": datetime.now().isoformat()
                        })
                        st.success(f"Document processed: {uploaded_file.name}")
                    else:
                        st.error(f"Failed to process document: {uploaded_file.name}")
                else:
                    st.warning("Please set Gemini API Key first")
    
    st.divider()
    
    # Media Files Section
    st.subheader("Upload Media Files")
    uploaded_media = st.file_uploader(
        "Upload Images, Audio, or Video", 
        type=["jpg", "jpeg", "png", "mp3", "wav", "mp4", "avi"], 
        accept_multiple_files=True,
        key="media_uploader"
    )
    
    if uploaded_media:
        for uploaded_file in uploaded_media:
            file_type = uploaded_file.type.split('/')[0]  # image, audio, video
            
            # Check if file already processed
            already_processed = False
            if file_type == "image":
                already_processed = uploaded_file.name in [img["name"] for img in st.session_state.processed_files["images"]]
            elif file_type == "audio":
                already_processed = uploaded_file.name in [audio["name"] for audio in st.session_state.processed_files["audio"]]
            elif file_type == "video":
                already_processed = uploaded_file.name in [video["name"] for video in st.session_state.processed_files["video"]]
            
            if not already_processed:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to processed files
                if file_type == "image":
                    st.session_state.processed_files["images"].append({
                        "name": uploaded_file.name,
                        "path": temp_file_path,
                        "type": uploaded_file.type,
                        "added_at": datetime.now().isoformat()
                    })
                    st.success(f"Image added: {uploaded_file.name}")
                elif file_type == "audio":
                    st.session_state.processed_files["audio"].append({
                        "name": uploaded_file.name,
                        "path": temp_file_path,
                        "type": uploaded_file.type,
                        "added_at": datetime.now().isoformat()
                    })
                    st.success(f"Audio added: {uploaded_file.name}")
                elif file_type == "video":
                    st.session_state.processed_files["video"].append({
                        "name": uploaded_file.name,
                        "path": temp_file_path,
                        "type": uploaded_file.type,
                        "added_at": datetime.now().isoformat()
                    })
                    st.success(f"Video added: {uploaded_file.name}")
    
    st.divider()
    
    # Web Content Section
    st.subheader("Add Web Content")
    web_url = st.text_input("Enter URL", key="web_url_input")
    
    if web_url and st.button("Add URL to Knowledge Base"):
        # Check if URL already in processed files
        if web_url not in [web["url"] for web in st.session_state.processed_files["web_content"]]:
            st.session_state.processed_files["web_content"].append({
                "url": web_url,
                "added_at": datetime.now().isoformat()
            })
            st.success(f"URL added: {web_url}")
        else:
            st.info(f"URL already in knowledge base: {web_url}")
    
    st.divider()
    
    # Knowledge Base Stats
    st.subheader("Knowledge Base Stats")
    st.write(f"Documents: {len(st.session_state.processed_files['documents'])}")
    st.write(f"Images: {len(st.session_state.processed_files['images'])}")
    st.write(f"Audio: {len(st.session_state.processed_files['audio'])}")
    st.write(f"Video: {len(st.session_state.processed_files['video'])}")
    st.write(f"Web Content: {len(st.session_state.processed_files['web_content'])}")
    
    if st.button("Clear Knowledge Base"):
        if st.session_state.doc_retrieval:
            st.session_state.doc_retrieval.clear()
        st.session_state.processed_files = {
            "documents": [],
            "images": [],
            "audio": [],
            "video": [],
            "web_content": []
        }
        st.success("Knowledge base cleared!")

# Main Chat Interface
st.title("AI Assistant with Document Processing")

# Display chat history
for message in st.session_state.chat_history:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message-content">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # For assistant messages, use st.write instead of markdown to handle HTML properly
            with st.chat_message("assistant"):
                st.write(message["content"])

# File upload in chat
chat_file = st.file_uploader(
    "Upload a file to include in your query", 
    type=["jpg", "jpeg", "png", "pdf"],
    key="chat_file_uploader"
)

# Text input for user query
user_query = st.text_input("Ask a question or give me a task", key="user_query")

# Process user query
if user_query and st.button("Send"):
    if st.session_state.gemini_handler is None:
        st.error("Please set Gemini API Key first")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Check if the query is document-related
        doc_related_keywords = ["document", "pdf", "docx", "text", "file", "read", "extract", "content"]
        is_doc_related = any(keyword in user_query.lower() for keyword in doc_related_keywords)
        
        # Generate response
        if is_doc_related and st.session_state.doc_retrieval:
            # Use document retrieval for document-related queries
            response = st.session_state.doc_retrieval.query(user_query)
        else:
            # Use Gemini for general queries or if document retrieval is not available
            image_data = None
            if chat_file and chat_file.type.startswith('image'):
                image = Image.open(chat_file)
                image_data = image
            
            response = st.session_state.gemini_handler.generate_response(user_query, image_data)
        
        # Process response to ensure HTML is properly handled (remove any unwanted tags)
        # Clean up any HTML div tags or other unwanted elements
        response = response.replace("</div>", "").replace("<div>", "")
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Force refresh to display new messages
        st.rerun()

# Run the app
if __name__ == "__main__":
    st.write("App is running!") 