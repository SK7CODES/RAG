# Multimodal RAG System

A multimodal retrieval-augmented generation (RAG) system that leverages AI to accurately retrieve, process, and generate context-aware responses from various data types including text, images, videos, and audio.

## Features

- **Multimodal Support**: Process and analyze text documents, images, audio, video, and web content
- **Advanced RAG**: Uses CrewAI tools for efficient retrieval and context generation
- **Agent-based Architecture**: Specialized AI agents for different data types and tasks
- **Gemini Integration**: Powered by Google's Gemini API for advanced language and multimodal capabilities
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## Requirements

- Python 3.10+ (Python 3.11+ recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Full Multimodal App

The full application includes support for multiple modalities:

```
streamlit run app.py
```

### Simplified Document RAG (Recommended for Python 3.10)

If you're having issues with the full multimodal app, use the simplified version that focuses on document processing:

```
streamlit run simple_app.py
```

This simplified version includes:
- PDF, DOCX, and TXT document processing
- Document chunking and embedding
- Semantic search using sentence transformers
- Question answering with Gemini API

## Troubleshooting

### Common Issues with Python 3.10

If using Python 3.10, you might encounter issues with the CrewAI library which requires Python 3.11+. 
We provide two solutions:

1. **Use the simplified app**: `streamlit run simple_app.py` 
2. **Upgrade to Python 3.11+**: This provides full compatibility with all features

### Dependencies Issues

If you encounter dependency issues, try:

```
pip install typing-extensions
```

## Supported File Types

- **Documents**: PDF, TXT, DOCX, PPTX
- **Images**: PNG, JPG, JPEG
- **Audio**: MP3, WAV
- **Video**: MP4
- **Web Content**: Websites and YouTube videos

## Architecture

The system uses a combination of specialized CrewAI agents and tools:

- **RagTool**: General-purpose RAG for various data types
- **PDFSearchTool**: Specialized for searching within PDF documents
- **YoutubeVideoSearchTool**: Specialized for YouTube video content analysis
- **WebsiteSearchTool**: Specialized for website content analysis
- **DallETool**: For image generation and analysis

## API Key

The system uses Google's Gemini API. A test API key is provided by default, but you can use your own by setting the `GEMINI_API_KEY` environment variable.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 