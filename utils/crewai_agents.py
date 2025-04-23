import os
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    RagTool, 
    PDFSearchTool, 
    YoutubeVideoSearchTool, 
    WebsiteSearchTool,
    DallETool, 
    LlamaIndexTool
)
from config.config import GEMINI_API_KEY, GEMINI_CONFIG

class MultimodalCrewAgents:
    """
    Defines specialized CrewAI agents for multimodal data processing.
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the multimodal crew agents.
        
        Args:
            gemini_api_key (str, optional): Gemini API key for LLM
        """
        self.gemini_api_key = gemini_api_key or GEMINI_API_KEY
        
        # Configure RAG with custom embeddings using Gemini
        self.rag_config = {
            "app": {
                "name": "multimodal_rag",
            },
            "llm": {
                "provider": "google",
                "config": {
                    "model": GEMINI_CONFIG["text_model"],
                    "api_key": self.gemini_api_key
                }
            },
            "embedding_model": {
                "provider": "google",
                "config": {
                    "model": "models/embedding-001",
                    "task_type": "retrieval_document",
                    "api_key": self.gemini_api_key
                }
            }
        }
        
        # Initialize tools with config
        self.rag_tool = RagTool(config=self.rag_config)
        self.pdf_tool = PDFSearchTool(config=self.rag_config)
        self.youtube_tool = YoutubeVideoSearchTool(config=self.rag_config)
        self.website_tool = WebsiteSearchTool(config=self.rag_config)
        self.dalle_tool = DallETool()
    
    def create_text_processing_agent(self) -> Agent:
        """
        Create an agent specialized in processing text documents.
        
        Returns:
            Agent: CrewAI agent for text processing
        """
        return Agent(
            role="Document Analysis Expert",
            goal="Extract key information from text documents and provide comprehensive answers",
            backstory="""You are an AI expert in analyzing text documents.
            With your exceptional skills in natural language processing,
            you can extract key insights and information from various document types.""",
            verbose=True,
            tools=[self.rag_tool, self.pdf_tool]
        )
    
    def create_image_analysis_agent(self) -> Agent:
        """
        Create an agent specialized in analyzing images.
        
        Returns:
            Agent: CrewAI agent for image analysis
        """
        return Agent(
            role="Visual Intelligence Specialist",
            goal="Analyze images and extract key information and insights",
            backstory="""You are an AI expert in computer vision and image analysis.
            You can identify objects, scenes, text, and patterns in images,
            and provide detailed descriptions and insights.""",
            verbose=True,
            tools=[self.dalle_tool]
        )
    
    def create_audio_analysis_agent(self) -> Agent:
        """
        Create an agent specialized in analyzing audio.
        
        Returns:
            Agent: CrewAI agent for audio analysis
        """
        return Agent(
            role="Audio Intelligence Expert",
            goal="Analyze audio content and extract key information",
            backstory="""You are an AI expert in audio processing and analysis.
            You can recognize speech, identify sounds, and extract meaningful
            information from audio recordings.""",
            verbose=True,
            tools=[self.rag_tool]
        )
    
    def create_video_analysis_agent(self) -> Agent:
        """
        Create an agent specialized in analyzing videos.
        
        Returns:
            Agent: CrewAI agent for video analysis
        """
        return Agent(
            role="Video Intelligence Specialist",
            goal="Analyze video content and extract key information and insights",
            backstory="""You are an AI expert in video processing and analysis.
            You can identify scenes, objects, people, and activities in videos,
            and provide detailed descriptions and insights.""",
            verbose=True,
            tools=[self.youtube_tool]
        )
    
    def create_web_research_agent(self) -> Agent:
        """
        Create an agent specialized in researching web content.
        
        Returns:
            Agent: CrewAI agent for web research
        """
        return Agent(
            role="Web Research Specialist",
            goal="Research and analyze web content to provide comprehensive information",
            backstory="""You are an AI expert in web research and information retrieval.
            You can extract relevant information from websites, articles, and other
            online sources to provide comprehensive answers.""",
            verbose=True,
            tools=[self.website_tool, self.rag_tool]
        )
    
    def create_integration_agent(self) -> Agent:
        """
        Create an agent that integrates information from all sources.
        
        Returns:
            Agent: CrewAI agent for information integration
        """
        return Agent(
            role="Information Integration Expert",
            goal="Integrate information from various sources to provide comprehensive responses",
            backstory="""You are an AI expert in knowledge integration and synthesis.
            You can combine information from various sources including documents,
            images, audio, video, and web content to provide comprehensive and coherent responses.""",
            verbose=True,
            tools=[self.rag_tool]
        )
    
    def create_multimodal_crew(self) -> Crew:
        """
        Create a crew of agents for multimodal processing.
        
        Returns:
            Crew: CrewAI crew for multimodal processing
        """
        # Create specialized agents
        text_agent = self.create_text_processing_agent()
        image_agent = self.create_image_analysis_agent()
        video_agent = self.create_video_analysis_agent()
        web_agent = self.create_web_research_agent()
        integration_agent = self.create_integration_agent()
        
        # Create tasks for each agent
        text_task = Task(
            description="""
            Analyze the provided text documents to extract key information relevant to the query.
            Focus on main concepts, entities, relationships, and context.
            """,
            agent=text_agent,
            expected_output="Detailed analysis of text documents with key insights"
        )
        
        image_task = Task(
            description="""
            Analyze the provided images to extract key visual information.
            Identify objects, scenes, text, and any other relevant elements.
            """,
            agent=image_agent,
            expected_output="Detailed analysis of images with key visual insights"
        )
        
        video_task = Task(
            description="""
            Analyze the provided videos to extract key information.
            Identify scenes, objects, people, activities, and temporal patterns.
            """,
            agent=video_agent,
            expected_output="Detailed analysis of videos with key insights"
        )
        
        web_task = Task(
            description="""
            Research relevant information from web sources to answer the query.
            Extract key facts, insights, and context from websites and online content.
            """,
            agent=web_agent,
            expected_output="Comprehensive information from web sources"
        )
        
        integration_task = Task(
            description="""
            Integrate all the information from the text, image, video, and web analyses
            to provide a comprehensive response to the user's query.
            Synthesize the diverse insights into a coherent and informative answer.
            """,
            agent=integration_agent,
            expected_output="Comprehensive and integrated response to the query"
        )
        
        # Create the crew with hierarchical process
        return Crew(
            agents=[text_agent, image_agent, video_agent, web_agent, integration_agent],
            tasks=[text_task, image_task, video_task, web_task, integration_task],
            process=Process.hierarchical,
            verbose=True
        )
    
    def process_multimodal_query(self, query: str, files: List[str] = None) -> str:
        """
        Process a multimodal query using the crew of agents.
        
        Args:
            query (str): The user's query
            files (List[str], optional): List of file paths to process
            
        Returns:
            str: The crew's response to the query
        """
        # Add files to the RAG tool
        if files:
            for file_path in files:
                _, ext = os.path.splitext(file_path)
                ext = ext.lstrip('.').lower()
                
                if ext == 'pdf':
                    self.pdf_tool = PDFSearchTool(pdf=file_path, config=self.rag_config)
                    self.rag_tool.add(data_type="file", path=file_path)
                elif ext in ['png', 'jpg', 'jpeg']:
                    # For images, we don't have direct RAG support in CrewAI yet
                    pass
                elif ext in ['mp4', 'mp3', 'wav']:
                    # For audio/video, we don't have direct RAG support in CrewAI yet
                    pass
                else:
                    self.rag_tool.add(data_type="file", path=file_path)
        
        # Create the crew
        crew = self.create_multimodal_crew()
        
        # Run the crew with the query
        result = crew.kickoff(inputs={"query": query})
        
        return result 