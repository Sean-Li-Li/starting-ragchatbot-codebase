"""
Simplified API endpoint tests without heavy dependencies.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock


def create_test_app():
    """Create a minimal test app for API testing."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="Test RAG API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mock RAG system
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "Test response about computer use",
        ["Test source"]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test_session_123"
    mock_session_manager.clear_session.return_value = None
    mock_rag.session_manager = mock_session_manager
    
    # Pydantic models
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
    
    # Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            mock_rag.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "RAG System API is running"}
    
    return app


@pytest.fixture
def test_client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints with minimal dependencies."""
    
    def test_query_endpoint(self, test_client):
        """Test query endpoint basic functionality."""
        response = test_client.post(
            "/api/query",
            json={"query": "What is computer use?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
    
    def test_courses_endpoint(self, test_client):
        """Test courses endpoint basic functionality."""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 1
    
    def test_session_endpoint(self, test_client):
        """Test session deletion endpoint."""
        response = test_client.delete("/api/session/test_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG System API is running"
    
    def test_invalid_endpoint(self, test_client):
        """Test invalid endpoint returns 404."""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
    
    def test_query_validation_error(self, test_client):
        """Test query endpoint with missing required field."""
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422
    
    def test_wrong_http_method(self, test_client):
        """Test wrong HTTP method returns 405."""
        response = test_client.get("/api/query")
        assert response.status_code == 405