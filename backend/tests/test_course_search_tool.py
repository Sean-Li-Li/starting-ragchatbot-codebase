"""
Comprehensive tests for CourseSearchTool functionality.

This module tests the CourseSearchTool.execute() method and related functionality
to identify issues with course content search and retrieval.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool execute method."""

    def test_execute_with_valid_query_and_results(self, mock_vector_store, sample_search_results):
        """Test execute() with a valid query that returns results."""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("computer use")
        
        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="computer use",
            course_name=None,
            lesson_number=None
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Lesson 0" in result
        assert "computer use capabilities" in result

    def test_execute_with_course_name_filter(self, mock_vector_store, sample_search_results):
        """Test execute() with course name filter."""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("computer use", course_name="Anthropic")
        
        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="computer use",
            course_name="Anthropic",
            lesson_number=None
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_lesson_number_filter(self, mock_vector_store, sample_search_results):
        """Test execute() with lesson number filter."""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("API basics", lesson_number=1)
        
        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="API basics",
            course_name=None,
            lesson_number=1
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_both_filters(self, mock_vector_store, sample_search_results):
        """Test execute() with both course name and lesson number filters."""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("computer use", course_name="Anthropic", lesson_number=0)
        
        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="computer use",
            course_name="Anthropic",
            lesson_number=0
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execute() when search returns no results."""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("nonexistent topic")
        
        # Assert
        mock_vector_store.search.assert_called_once()
        assert result == "No relevant content found."

    def test_execute_with_empty_results_and_course_filter(self, mock_vector_store, empty_search_results):
        """Test execute() with empty results and course filter."""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("nonexistent topic", course_name="Nonexistent Course")
        
        # Assert
        assert result == "No relevant content found in course 'Nonexistent Course'."

    def test_execute_with_empty_results_and_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test execute() with empty results and lesson filter."""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("nonexistent topic", lesson_number=999)
        
        # Assert
        assert result == "No relevant content found in lesson 999."

    def test_execute_with_empty_results_and_both_filters(self, mock_vector_store, empty_search_results):
        """Test execute() with empty results and both filters."""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("nonexistent topic", course_name="Test Course", lesson_number=5)
        
        # Assert
        assert result == "No relevant content found in course 'Test Course' in lesson 5."

    def test_execute_with_vector_store_error(self, mock_vector_store, error_search_results):
        """Test execute() when vector store returns an error."""
        # Arrange
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("any query")
        
        # Assert
        mock_vector_store.search.assert_called_once()
        assert result == "Vector store connection failed"

    def test_execute_with_vector_store_exception(self, mock_vector_store):
        """Test execute() when vector store throws an exception."""
        # Arrange
        mock_vector_store.search.side_effect = Exception("Database connection error")
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("any query")
        
        # Assert - should now handle exceptions gracefully
        assert result == "Search failed: Database connection error"

    def test_format_results_basic(self, mock_vector_store, sample_search_results):
        """Test _format_results() method with basic search results."""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool._format_results(sample_search_results)
        
        # Assert
        assert isinstance(result, str)
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 1]" in result
        assert "computer use capabilities" in result
        assert "API and basic requests" in result

    def test_format_results_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that _format_results() properly tracks sources."""
        # Arrange
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        tool._format_results(sample_search_results)
        
        # Assert
        assert len(tool.last_sources) == 2
        assert all("text" in source and "link" in source for source in tool.last_sources)
        assert "Building Towards Computer Use with Anthropic - Lesson 0" == tool.last_sources[0]["text"]
        assert "Building Towards Computer Use with Anthropic - Lesson 1" == tool.last_sources[1]["text"]

    def test_format_results_with_lesson_links(self, mock_vector_store, sample_search_results):
        """Test _format_results() retrieves and includes lesson links."""
        # Arrange
        expected_link = "https://learn.deeplearning.ai/courses/test/lesson1"
        mock_vector_store.get_lesson_link.return_value = expected_link
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        tool._format_results(sample_search_results)
        
        # Assert
        mock_vector_store.get_lesson_link.assert_called()
        assert tool.last_sources[0]["link"] == expected_link
        assert tool.last_sources[1]["link"] == expected_link

    def test_format_results_handles_missing_metadata(self, mock_vector_store):
        """Test _format_results() handles missing metadata gracefully."""
        # Arrange
        results_with_missing_metadata = SearchResults(
            documents=["Some content"],
            metadata=[{}],  # Empty metadata
            distances=[0.5]
        )
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool._format_results(results_with_missing_metadata)
        
        # Assert
        assert isinstance(result, str)
        assert "[unknown]" in result  # Should use default for missing course_title

    def test_get_tool_definition(self, mock_vector_store):
        """Test get_tool_definition() returns proper Anthropic tool format."""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        definition = tool.get_tool_definition()
        
        # Assert
        assert isinstance(definition, dict)
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    @pytest.mark.parametrize("query,course_name,lesson_number,expected_calls", [
        ("test query", None, None, 1),
        ("test query", "Course Name", None, 1),
        ("test query", None, 5, 1),
        ("test query", "Course Name", 5, 1),
        ("", None, None, 1),  # Empty query should still make a call
    ])
    def test_execute_parameter_combinations(self, mock_vector_store, sample_search_results, 
                                          query, course_name, lesson_number, expected_calls):
        """Test execute() with various parameter combinations."""
        # Arrange
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute(query, course_name, lesson_number)
        
        # Assert
        assert mock_vector_store.search.call_count == expected_calls
        assert isinstance(result, str)

class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with more realistic scenarios."""

    def test_execute_real_world_query_flow(self, mock_vector_store):
        """Test a realistic query flow that mimics user interaction."""
        # Arrange - set up realistic search results
        realistic_results = SearchResults(
            documents=[
                "Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities and how AI can interact with computers.",
                "Course Building Towards Computer Use with Anthropic Lesson 1 content: In this lesson, you'll learn to make basic API requests to Anthropic's Claude model."
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
            distances=[0.2, 0.4]
        )
        
        mock_vector_store.search.return_value = realistic_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://learn.deeplearning.ai/lesson/0",
            "https://learn.deeplearning.ai/lesson/1"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("What is computer use and how do I make API requests?")
        
        # Assert
        assert isinstance(result, str)
        assert len(result) > 0
        assert "computer use capabilities" in result
        assert "API requests" in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 0]" in result
        assert "[Building Towards Computer Use with Anthropic - Lesson 1]" in result
        
        # Check sources were properly tracked
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["link"] == "https://learn.deeplearning.ai/lesson/0"
        assert tool.last_sources[1]["link"] == "https://learn.deeplearning.ai/lesson/1"

    def test_execute_handles_course_name_resolution_failure(self, mock_vector_store):
        """Test execute() when course name cannot be resolved."""
        # Arrange - vector store search returns error for course resolution
        error_result = SearchResults.empty("No course found matching 'Nonexistent Course'")
        mock_vector_store.search.return_value = error_result
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Act
        result = tool.execute("some query", course_name="Nonexistent Course")
        
        # Assert
        assert result == "No course found matching 'Nonexistent Course'"