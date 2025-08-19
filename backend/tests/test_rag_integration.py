"""
Comprehensive end-to-end tests for RAG system integration.

This module tests the complete RAG system flow from user queries
through tool execution to final responses, identifying where
"query failed" errors originate.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from unittest.mock import patch

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


class TestRAGSystemBasics:
    """Test basic RAG system functionality and initialization."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_rag_system_initialization(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test RAG system initializes all components correctly."""
        # Act
        rag = RAGSystem(mock_config)
        
        # Assert
        assert rag.config == mock_config
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_tool_registration(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test that tools are properly registered with the tool manager."""
        # Act
        rag = RAGSystem(mock_config)
        
        # Assert - check that tools were registered
        tool_definitions = rag.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2  # search_tool + outline_tool
        
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Test the core query functionality of the RAG system."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_without_session(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test basic query without session ID."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Test response about computer use"
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.reset_sources.return_value = None
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_manager
        
        # Act
        response, sources = rag.query("What is computer use?")
        
        # Assert
        assert response == "Test response about computer use"
        assert sources == []
        
        # Verify AI generator was called with correct parameters
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        
        assert "What is computer use?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query with session ID and conversation history."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Follow-up response about computer use"
        
        mock_session_instance = mock_session.return_value
        mock_session_instance.get_conversation_history.return_value = "Previous conversation context"
        mock_session_instance.add_exchange.return_value = None
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.reset_sources.return_value = None
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_manager
        rag.session_manager = mock_session_instance
        
        # Act
        response, sources = rag.query("Tell me more", session_id="test_session")
        
        # Assert
        mock_session_instance.get_conversation_history.assert_called_once_with("test_session")
        mock_session_instance.add_exchange.assert_called_once_with("test_session", "Tell me more", "Follow-up response about computer use")
        
        # Verify AI generator received conversation history
        call_args = mock_ai_gen_instance.generate_response.call_args[1]
        assert call_args["conversation_history"] == "Previous conversation context"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_sources_from_tools(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query that generates sources from tool usage."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response with sources"
        
        mock_sources = [
            {
                "text": "Building Towards Computer Use with Anthropic - Lesson 0",
                "link": "https://learn.deeplearning.ai/lesson/0"
            }
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = mock_sources
        mock_tool_manager.reset_sources.return_value = None
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_manager
        
        # Act
        response, sources = rag.query("What is computer use?")
        
        # Assert
        assert response == "Response with sources"
        assert sources == mock_sources
        
        # Verify sources were retrieved and reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()


class TestRAGSystemErrorScenarios:
    """Test error scenarios that could cause 'query failed'."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_ai_generator_exception(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query when AI generator throws an exception."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.side_effect = Exception("API key not found")
        
        rag = RAGSystem(mock_config)
        
        # Act
        response, sources = rag.query("What is computer use?")
        
        # Assert - should handle error gracefully and return error message
        # The system detects "API key" in the error and returns a specific message
        assert "Query failed: Invalid API key. Please configure your Anthropic API key in the .env file." == response
        assert sources == []

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_tool_manager_error(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query when tool manager encounters errors."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Error response"
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []
        mock_tool_manager.get_last_sources.side_effect = Exception("Tool manager error")
        
        rag = RAGSystem(mock_config)
        rag.tool_manager = mock_tool_manager
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            rag.query("What is computer use?")
        
        assert "Tool manager error" in str(exc_info.value)


class TestRAGSystemRealWorldScenarios:
    """Test realistic end-to-end scenarios that users would encounter."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')  
    @patch('rag_system.DocumentProcessor')
    def test_successful_content_query_with_tool_use(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test a successful content query that uses tools and returns proper results."""
        # Arrange - Mock AI generator to simulate tool calling flow
        mock_ai_gen_instance = mock_ai_gen.return_value
        
        # Mock tool call response - AI decides to use search tool
        mock_ai_gen_instance.generate_response.return_value = "Computer use refers to the ability of AI models to interact with computers by taking screenshots and generating mouse clicks or keystrokes to execute tasks."
        
        # Mock vector store with realistic search results
        mock_vector_store_instance = mock_vector_store.return_value
        mock_search_results = SearchResults(
            documents=[
                "Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches about computer use capabilities and how AI can interact with computers.",
                "Course Building Towards Computer Use with Anthropic Lesson 1 content: Computer use allows models to look at screens, take screenshots and generate actions."
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
                    "chunk_index": 1
                }
            ],
            distances=[0.2, 0.3]
        )
        mock_vector_store_instance.search.return_value = mock_search_results
        mock_vector_store_instance.get_lesson_link.return_value = "https://learn.deeplearning.ai/lesson/0"
        
        rag = RAGSystem(mock_config)
        
        # Act
        response, sources = rag.query("What is computer use?")
        
        # Assert
        assert response == "Computer use refers to the ability of AI models to interact with computers by taking screenshots and generating mouse clicks or keystrokes to execute tasks."
        assert isinstance(sources, list)
        
        # Verify the complete flow worked
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args[1]
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_query_that_should_fail_gracefully(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query with conditions that might cause 'query failed' error."""
        # Arrange - Set up conditions that could cause failure
        mock_ai_gen_instance = mock_ai_gen.return_value
        
        # Simulate API key error
        mock_ai_gen_instance.generate_response.side_effect = Exception("Invalid API key")
        
        rag = RAGSystem(mock_config)
        
        # Act
        response, sources = rag.query("What is computer use?")
        
        # Assert - should handle error gracefully
        assert "Query failed:" in response
        assert "Invalid API key" in response
        assert sources == []

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_empty_search_results_handling(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test how the system handles empty search results."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "I couldn't find any relevant information about that topic."
        
        # Mock empty search results
        mock_vector_store_instance = mock_vector_store.return_value
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store_instance.search.return_value = empty_results
        
        rag = RAGSystem(mock_config)
        
        # Act
        response, sources = rag.query("Tell me about a nonexistent topic")
        
        # Assert
        assert response == "I couldn't find any relevant information about that topic."
        assert sources == []

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_course_outline_query(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test query that should use the course outline tool."""
        # Arrange
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = """**Course:** Building Towards Computer Use with Anthropic
**Instructor:** Colt Steele
**Course Link:** https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/

**Lessons (2 total):**
  0. Introduction
  1. Getting Started with Anthropic"""
        
        # Mock course metadata
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance._resolve_course_name.return_value = "Building Towards Computer Use with Anthropic"
        mock_vector_store_instance.get_all_courses_metadata.return_value = [
            {
                "title": "Building Towards Computer Use with Anthropic",
                "instructor": "Colt Steele", 
                "course_link": "https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
                "lessons": [
                    {"lesson_number": 0, "lesson_title": "Introduction"},
                    {"lesson_number": 1, "lesson_title": "Getting Started with Anthropic"}
                ]
            }
        ]
        
        rag = RAGSystem(mock_config)
        
        # Act
        response, sources = rag.query("What courses are available?")
        
        # Assert
        assert "Building Towards Computer Use with Anthropic" in response
        assert "Colt Steele" in response
        assert "Introduction" in response


class TestRAGSystemDocumentProcessing:
    """Test document processing and course loading functionality."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_success(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, sample_course, sample_course_chunks, mock_config):
        """Test successful course document addition."""
        # Arrange
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.add_course_metadata.return_value = None
        mock_vector_store_instance.add_course_content.return_value = None
        
        rag = RAGSystem(mock_config)
        
        # Act
        course, chunk_count = rag.add_course_document("/path/to/course.txt")
        
        # Assert
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)
        
        mock_doc_proc_instance.process_course_document.assert_called_once_with("/path/to/course.txt")
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store_instance.add_course_content.assert_called_once_with(sample_course_chunks)

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_error(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test course document addition with processing error."""
        # Arrange
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.side_effect = Exception("File not found")
        
        rag = RAGSystem(mock_config)
        
        # Act
        course, chunk_count = rag.add_course_document("/invalid/path.txt")
        
        # Assert - should handle error gracefully
        assert course is None
        assert chunk_count == 0


class TestRAGSystemAnalytics:
    """Test RAG system analytics and statistics functionality."""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(self, mock_doc_proc, mock_session, mock_ai_gen, mock_vector_store, mock_config):
        """Test getting course analytics."""
        # Arrange
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.get_course_count.return_value = 3
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Building Towards Computer Use with Anthropic",
            "Introduction to Machine Learning",
            "Deep Learning Fundamentals"
        ]
        
        rag = RAGSystem(mock_config)
        
        # Act
        analytics = rag.get_course_analytics()
        
        # Assert
        assert analytics["total_courses"] == 3
        assert len(analytics["course_titles"]) == 3
        assert "Building Towards Computer Use with Anthropic" in analytics["course_titles"]