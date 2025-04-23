import streamlit as st
import os
import tempfile
import base64
from datetime import datetime
import google.generativeai as genai

# Import the simple document retrieval system
from simple_doc_retrieval import SimpleDocRetrieval

# Set up page config
st.set_page_config(
    page_title="Simple Document RAG",
    page_icon="📄",
    layout="wide"
)

# API key input in sidebar
with st.sidebar:
    st.title("Simple Document RAG")
    api_key = st.text_input("Gemini API Key", value="AIzaSyCq_PkkOA-UMmkOREAUmkv0Pg_2t_PZLbQ", type="password")
    st.markdown("---")
    
    if api_key:
        # Initialize the document retrieval system
        if 'doc_retrieval' not in st.session_state:
            st.session_state.doc_retrieval = SimpleDocRetrieval(api_key)
        
        # File upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save file to temp location
                    temp_dir = st.session_state.doc_retrieval.temp_dir
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    # Skip if already processed
                    if 'processed_files' not in st.session_state:
                        st.session_state.processed_files = []
                    
                    if uploaded_file.name in [f['filename'] for f in st.session_state.processed_files]:
                        continue
                    
                    # Save and process
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add to document retrieval system
                    success = st.session_state.doc_retrieval.add_document(temp_file_path)
                    
                    if success:
                        # Track processed files
                        st.session_state.processed_files.append({
                            'filename': uploaded_file.name,
                            'path': temp_file_path,
                            'added_at': datetime.now().isoformat()
                        })
                
                st.success(f"Processed {len(uploaded_files)} documents!")
        
        # Display processed files
        if 'processed_files' in st.session_state and st.session_state.processed_files:
            st.header("Processed Documents")
            for doc in st.session_state.processed_files:
                st.write(f"📄 {doc['filename']}")
            
            # Clear button
            if st.button("Clear All Documents"):
                st.session_state.doc_retrieval = SimpleDocRetrieval(api_key)
                st.session_state.processed_files = []
                st.experimental_rerun()

# Main content area
st.title("Document Question Answering")

# Check if system is initialized
if 'doc_retrieval' not in st.session_state:
    st.warning("Please enter your Gemini API key in the sidebar.")
else:
    # Chat input
    query = st.text_input("Ask a question about your documents:")
    
    if query:
        with st.spinner("Searching documents and generating answer..."):
            # Get answer from document retrieval system
            answer = st.session_state.doc_retrieval.query(query)
            
            # Display answer
            st.subheader("Answer")
            st.write(answer)
            
            # Add to chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Display chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
            
            st.markdown("---") 