import streamlit as st
import os
import sys
import tempfile
import base64
from datetime import datetime

# Apply patch for typing.Self in Python 3.10
from utils.crewai_patch import *

# Import Google Generative AI after patch
import google.generativeai as genai
from PIL import Image

# Import our custom modules
from utils.rag_tools import RagToolManager
from utils.file_processor import FileProcessor
from utils.gemini_handler import GeminiHandler
from utils.crewai_agents import MultimodalCrewAgents
from config.config import GEMINI_API_KEY, TEMP_DIR

# Set page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

# Try initializing tools with error handling
try:
    if "file_processor" not in st.session_state:
        st.session_state.file_processor = FileProcessor()
    
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = RagToolManager()
except Exception as e:
    st.error(f"Error initializing RAG tools: {str(e)}")
    st.info("Proceeding with limited functionality. Some features may not work.")
    
    if "file_processor" not in st.session_state:
        st.session_state.file_processor = None
    
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = None

if "gemini_handler" not in st.session_state:
    st.session_state.gemini_handler = None
if "crew_agents" not in st.session_state:
    st.session_state.crew_agents = None
if "use_crew" not in st.session_state:
    st.session_state.use_crew = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E8F0FE;
        border-left: 5px solid #4F8BF9;
    }
    .bot-message {
        background-color: #F0F2F6;
        border-left: 5px solid #16A34A;
    }
    .message-content {
        display: flex;
        flex-direction: row;
    }
    .message-text {
        margin-left: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configurations and data uploads
with st.sidebar:
    st.markdown("<h1 class='main-header'>Multimodal RAG System</h1>", unsafe_allow_html=True)
    
    # API Key input
    gemini_api_key = st.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
    
    if gemini_api_key:
        st.session_state.gemini_handler = GeminiHandler(api_key=gemini_api_key)
        
        # Initialize CrewAI agents with error handling
        try:
            st.session_state.crew_agents = MultimodalCrewAgents(gemini_api_key=gemini_api_key)
            st.success("API Key configured successfully!")
        except Exception as e:
            st.warning(f"API Key configured for basic functionality. CrewAI agents initialization failed: {str(e)}")
            st.session_state.crew_agents = None
            st.session_state.use_crew = False
    
    # Toggle between direct Gemini and CrewAI
    if st.session_state.crew_agents:
        st.session_state.use_crew = st.toggle("Use CrewAI Agents", value=False)
        if st.session_state.use_crew:
            st.info("Using CrewAI agents for more sophisticated processing")
    
    st.markdown("---")
    
    # File uploads section
    st.header("Upload Knowledge Base")
    
    if st.session_state.file_processor and st.session_state.rag_manager:
        uploaded_files = st.file_uploader(
            "Upload documents, images, audio, or video files",
            type=["pdf", "txt", "png", "jpg", "jpeg", "mp3", "mp4", "wav", "pptx", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    # Check if file already processed
                    if uploaded_file.name in [f["filename"] for f in st.session_state.processed_files]:
                        continue
                    
                    # Save the file to temp location
                    temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file based on its type
                    result = st.session_state.file_processor.process_file(temp_file_path)
                    
                    # Add to RAG knowledge base
                    try:
                        st.session_state.rag_manager.add_to_knowledge_base(temp_file_path)
                    except Exception as e:
                        st.warning(f"Error adding file to knowledge base: {str(e)}")
                    
                    # Track processed files
                    st.session_state.processed_files.append({
                        "filename": uploaded_file.name,
                        "path": temp_file_path,
                        "type": result.get("content_type", "unknown")
                    })
                
                st.success(f"Successfully processed {len(uploaded_files)} files!")
        
        # URL input for web content and YouTube
        st.header("Add Web Content")
        web_url = st.text_input("Enter a URL (webpage or YouTube video)")
        
        if web_url and st.button("Add to Knowledge Base"):
            with st.spinner("Processing web content..."):
                try:
                    st.session_state.rag_manager.add_web_content(web_url)
                    st.success("Web content added successfully!")
                except Exception as e:
                    st.error(f"Error adding web content: {str(e)}")
        
        # Show knowledge base stats
        if st.session_state.processed_files:
            st.markdown("---")
            st.header("Knowledge Base Stats")
            
            # Count by type
            doc_count = sum(1 for f in st.session_state.processed_files if f["type"] == "text")
            img_count = sum(1 for f in st.session_state.processed_files if f["type"] == "image")
            audio_count = sum(1 for f in st.session_state.processed_files if f["type"] == "audio")
            video_count = sum(1 for f in st.session_state.processed_files if f["type"] == "video")
            
            # Display stats
            st.write(f"📄 Documents: {doc_count}")
            st.write(f"🖼️ Images: {img_count}")
            st.write(f"🔊 Audio: {audio_count}")
            st.write(f"🎬 Video: {video_count}")
            
            # Clear knowledge base button
            if st.button("Clear Knowledge Base"):
                st.session_state.processed_files = []
                
                # Re-initialize the RAG manager
                try:
                    st.session_state.rag_manager = RagToolManager()
                except Exception as e:
                    st.error(f"Error re-initializing RAG manager: {str(e)}")
                    st.session_state.rag_manager = None
                
                st.experimental_rerun()
    else:
        st.warning("RAG tools initialization failed. File upload and processing is disabled.")

# Main chat interface
st.title("Multimodal RAG Assistant")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class='chat-message user-message'>
            <div class='message-content'>
                <div>👤</div>
                <div class='message-text'>{message["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='chat-message bot-message'>
            <div class='message-content'>
                <div>🤖</div>
                <div class='message-text'>{message["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Chat input and media upload for queries
query_text = st.text_input("Ask a question:")
query_media = st.file_uploader(
    "Or upload media for your query:",
    type=["png", "jpg", "jpeg", "mp3", "mp4", "wav"],
    key="query_media"
)

if query_media:
    # Display the uploaded media for the query
    media_type = query_media.type.split('/')[0]
    if media_type == "image":
        st.image(query_media, caption="Uploaded Image for Query", use_column_width=True)
    elif media_type == "audio":
        st.audio(query_media, format="audio/wav")
    elif media_type == "video":
        st.video(query_media)

# Submit button
if st.button("Submit"):
    if not st.session_state.gemini_handler:
        st.error("Please configure the Gemini API key first!")
    elif not (query_text or query_media):
        st.warning("Please enter a question or upload media for your query.")
    else:
        with st.spinner("Generating response..."):
            # Add user message to chat history
            user_content = query_text if query_text else "Multimodal query with uploaded media"
            st.session_state.chat_history.append({"role": "user", "content": user_content})
            
            # Process the query (text and/or media)
            response = None
            try:
                if st.session_state.use_crew and st.session_state.crew_agents:
                    # Use CrewAI for more sophisticated processing
                    files_to_process = [f["path"] for f in st.session_state.processed_files]
                    
                    if query_media:
                        # Save the query media to temp location
                        temp_query_path = os.path.join(TEMP_DIR, query_media.name)
                        with open(temp_query_path, "wb") as f:
                            f.write(query_media.getbuffer())
                        
                        # Add to files to process
                        files_to_process.append(temp_query_path)
                    
                    response = st.session_state.crew_agents.process_multimodal_query(
                        query=query_text if query_text else "Analyze this media",
                        files=files_to_process
                    )
                else:
                    # Use direct Gemini API call
                    if query_media:
                        temp_query_path = os.path.join(TEMP_DIR, query_media.name)
                        with open(temp_query_path, "wb") as f:
                            f.write(query_media.getbuffer())
                        
                        response = st.session_state.gemini_handler.generate_multimodal_response(
                            query_text, 
                            temp_query_path, 
                            st.session_state.rag_manager
                        )
                    else:
                        response = st.session_state.gemini_handler.generate_text_response(
                            query_text,
                            st.session_state.rag_manager
                        )
            except Exception as e:
                response = f"Error generating response: {str(e)}\n\nPlease try again with a different query or check the system configuration."
            
            # Add bot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("Multimodal RAG System powered by CrewAI and Gemini API") 