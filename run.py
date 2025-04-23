#!/usr/bin/env python
"""
Entry point script for running the Multimodal RAG System.
"""

import os
import subprocess
import sys

def main():
    """
    Main entry point for the application.
    """
    print("Starting Multimodal RAG System...")
    
    # Check if required directories exist
    for directory in ["data", "data/temp", "config", "utils"]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 