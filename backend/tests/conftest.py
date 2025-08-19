"""
Test fixtures and mocks for the RAG system test suite.

This module provides reusable test fixtures, mock objects, and test data
to support comprehensive testing of the RAG chatbot system.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Add the backend directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_course():
    """Sample course data for testing."""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
            ),
            Lesson(
                lesson_number=1,
                title="Getting Started with Anthropic",
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/b7k1z/getting-started"
            )
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing."""
    return [
        CourseChunk(
            content="Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 0 content: You will learn about large language models and their ability to process images and use tools.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 1 content: This lesson covers getting started with Anthropic's API and basic requests.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=1,
            chunk_index=2
        )
    ]

@pytest.fixture
def sample_search_results():
    """Sample search results from vector store."""
    return SearchResults(
        documents=[
            "Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities.",
            "Course Building Towards Computer Use with Anthropic Lesson 1 content: This lesson covers getting started with Anthropic's API and basic requests."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic", 
                "lesson_number": 1,
                "chunk_index": 2
            }
        ],
        distances=[0.3, 0.5]
    )

@pytest.fixture
def empty_search_results():
    """Empty search results for testing failure cases."""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Search results with error for testing error handling."""
    return SearchResults.empty("Vector store connection failed")


# ============================================================================
# Mock Objects and Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing without database dependency."""
    mock_store = Mock()
    
    # Default successful search behavior
    mock_store.search.return_value = SearchResults(
        documents=[
            "Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 0
            }
        ],
        distances=[0.3]
    )
    
    # Default lesson link behavior
    mock_store.get_lesson_link.return_value = "https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
    
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generation without API calls."""
    mock_client = Mock()
    
    # Mock response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "stop"
    mock_response.content = [Mock(text="This is a test response about course content.")]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_client_with_tools():
    """Mock Anthropic client that simulates tool calling behavior."""
    mock_client = Mock()
    
    # Mock tool use response
    mock_tool_response = Mock()
    mock_tool_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_call_123"
    mock_tool_content.input = {"query": "computer use"}
    
    mock_tool_response.content = [mock_tool_content]
    
    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.stop_reason = "stop"
    mock_final_response.content = [Mock(text="Based on the search results, computer use refers to the ability of AI models to interact with computers.")]
    
    # Configure the mock to return tool response first, then final response
    mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
    
    return mock_client

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


# ============================================================================
# Test Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup."""
    # Set test environment variables
    os.environ["ANTHROPIC_API_KEY"] = "test_api_key_for_testing"
    
    yield
    
    # Cleanup if needed
    pass


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing tool execution."""
    mock_manager = Mock()
    
    # Mock tool definitions
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the course content"},
                    "course_name": {"type": "string", "description": "Course title (partial matches work)"},
                    "lesson_number": {"type": "integer", "description": "Specific lesson number to search within"}
                },
                "required": ["query"]
            }
        }
    ]
    
    # Mock successful tool execution
    mock_manager.execute_tool.return_value = "[Building Towards Computer Use with Anthropic - Lesson 0]\nWelcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities."
    
    # Mock sources
    mock_manager.get_last_sources.return_value = [
        {
            "text": "Building Towards Computer Use with Anthropic - Lesson 0",
            "link": "https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
        }
    ]
    
    mock_manager.reset_sources.return_value = None
    
    return mock_manager


# ============================================================================
# FastAPI Test Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing without dependencies."""
    mock_rag = Mock()
    
    # Mock successful query response
    mock_rag.query.return_value = (
        "This is a test response about computer use capabilities.",
        ["Building Towards Computer Use with Anthropic - Lesson 0"]
    )
    
    # Mock course analytics
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Building Towards Computer Use with Anthropic"]
    }
    
    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test_session_123"
    mock_session_manager.clear_session.return_value = None
    mock_rag.session_manager = mock_session_manager
    
    return mock_rag

@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app without static file mounting to avoid filesystem issues
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models (same as main app)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Add API endpoints (same as main app but with mocked dependencies)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        from fastapi import HTTPException
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        from fastapi import HTTPException
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        from fastapi import HTTPException
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Simple root endpoint for testing static file serving concept
    @app.get("/")
    async def read_root():
        return {"message": "RAG System API is running"}
    
    return app

@pytest.fixture
def test_client(test_app):
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    return TestClient(test_app)


# ============================================================================
# Parameterized Test Data
# ============================================================================

@pytest.fixture
def test_queries():
    """Various test queries for comprehensive testing."""
    return {
        "content_queries": [
            "What is computer use?",
            "How do I get started with Anthropic?",
            "Tell me about lesson 0",
            "What topics are covered in this course?"
        ],
        "course_queries": [
            "What courses are available?",
            "Show me the course outline",
            "List all lessons"
        ],
        "error_queries": [
            "Tell me about nonexistent course",
            "What is lesson 999?",
            ""  # empty query
        ]
    }