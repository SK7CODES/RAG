# Utils module initialization

# Apply patch for typing.Self in Python 3.10 before imports
from utils.crewai_patch import *

# Import modules after patch is applied
from utils.file_processor import FileProcessor
from utils.gemini_handler import GeminiHandler
from utils.rag_tools import RagToolManager
from utils.crewai_agents import MultimodalCrewAgents 