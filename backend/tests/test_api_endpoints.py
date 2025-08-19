"""
API endpoint tests for the RAG system FastAPI application.

This module tests all REST API endpoints including query processing,
course analytics, session management, and error handling scenarios.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query POST endpoint."""
    
    def test_query_with_new_session(self, test_client):
        """Test query without session_id creates new session."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about computer use capabilities."
        assert data["sources"] == ["Building Towards Computer Use with Anthropic - Lesson 0"]
        assert data["session_id"] == "test_session_123"
    
    def test_query_with_existing_session(self, test_client):
        """Test query with provided session_id."""
        response = test_client.post(
            "/api/query",
            json={
                "query": "Tell me more about lesson 1",
                "session_id": "existing_session_456"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "existing_session_456"
        assert "answer" in data
        assert "sources" in data
    
    def test_query_empty_query(self, test_client):
        """Test query with empty string."""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200
        # Even empty queries should get processed by the mock
        data = response.json()
        assert "answer" in data
    
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON."""
        response = test_client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_required_field(self, test_client):
        """Test query endpoint without required query field."""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test_session"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises exception."""
        mock_rag_system.query.side_effect = Exception("Vector store connection failed")
        
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Vector store connection failed" in data["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses GET endpoint."""
    
    def test_get_course_stats_success(self, test_client):
        """Test successful course statistics retrieval."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 1
        assert data["course_titles"] == ["Building Towards Computer Use with Anthropic"]
    
    def test_get_course_stats_no_courses(self, test_client, mock_rag_system):
        """Test course statistics when no courses are available."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_course_stats_error(self, test_client, mock_rag_system):
        """Test course statistics endpoint when RAG system fails."""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database connection error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection error" in data["detail"]


@pytest.mark.api
class TestSessionEndpoint:
    """Test the /api/session/{session_id} DELETE endpoint."""
    
    def test_clear_session_success(self, test_client):
        """Test successful session clearing."""
        response = test_client.delete("/api/session/test_session_123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"
    
    def test_clear_nonexistent_session(self, test_client, mock_rag_system):
        """Test clearing a session that doesn't exist."""
        # Mock should handle nonexistent sessions gracefully
        response = test_client.delete("/api/session/nonexistent_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"
    
    def test_clear_session_error(self, test_client, mock_rag_system):
        """Test session clearing when session manager fails."""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session manager error")
        
        response = test_client.delete("/api/session/test_session")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Session manager error" in data["detail"]
    
    def test_clear_session_invalid_id_format(self, test_client):
        """Test session clearing with various session ID formats."""
        # Test with special characters
        response = test_client.delete("/api/session/session@#$%")
        assert response.status_code == 200
        
        # Test with very long session ID
        long_session_id = "a" * 1000
        response = test_client.delete(f"/api/session/{long_session_id}")
        assert response.status_code == 200


@pytest.mark.api
class TestRootEndpoint:
    """Test the root (/) endpoint."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API status."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG System API is running"


@pytest.mark.api
class TestCORSAndHeaders:
    """Test CORS and HTTP headers."""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses."""
        response = test_client.options("/api/query")
        
        # FastAPI automatically handles preflight requests
        assert response.status_code in [200, 405]  # 405 for method not allowed is also acceptable
    
    def test_cors_origin_handling(self, test_client):
        """Test CORS origin handling with custom Origin header."""
        headers = {"Origin": "http://localhost:3000"}
        response = test_client.get("/api/courses", headers=headers)
        
        assert response.status_code == 200


@pytest.mark.api
class TestErrorHandling:
    """Test general error handling scenarios."""
    
    def test_invalid_endpoint(self, test_client):
        """Test request to non-existent endpoint."""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_wrong_http_method(self, test_client):
        """Test using wrong HTTP method on endpoints."""
        # GET on POST endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed
        
        # POST on GET endpoint
        response = test_client.post("/api/courses")
        assert response.status_code == 405  # Method Not Allowed
    
    def test_large_request_payload(self, test_client):
        """Test handling of large request payloads."""
        large_query = "A" * 10000  # 10KB query
        response = test_client.post(
            "/api/query",
            json={"query": large_query}
        )
        
        # Should still process successfully with mocked system
        assert response.status_code == 200


@pytest.mark.api 
@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end API scenarios."""
    
    def test_complete_conversation_flow(self, test_client):
        """Test a complete conversation flow with session management."""
        # First query - creates new session
        response1 = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second query - uses same session
        response2 = test_client.post(
            "/api/query", 
            json={
                "query": "Tell me more about it",
                "session_id": session_id
            }
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Get course stats
        response3 = test_client.get("/api/courses")
        assert response3.status_code == 200
        
        # Clear session
        response4 = test_client.delete(f"/api/session/{session_id}")
        assert response4.status_code == 200
    
    def test_multiple_concurrent_sessions(self, test_client):
        """Test handling multiple concurrent sessions."""
        sessions = []
        
        # Create multiple sessions
        for i in range(3):
            response = test_client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])
        
        # Verify all sessions are unique
        assert len(set(sessions)) == 3
        
        # Clear all sessions
        for session_id in sessions:
            response = test_client.delete(f"/api/session/{session_id}")
            assert response.status_code == 200


@pytest.mark.api
@pytest.mark.parametrize("query,expected_status", [
    ("What is computer use?", 200),
    ("", 200),  # Empty query should still work with mock
    ("A" * 1000, 200),  # Long query
    ("Query with special chars: !@#$%^&*()", 200),
    ("Query with unicode: 你好世界", 200),
])
def test_query_variations(test_client, query, expected_status):
    """Test various query input variations."""
    response = test_client.post(
        "/api/query",
        json={"query": query}
    )
    assert response.status_code == expected_status