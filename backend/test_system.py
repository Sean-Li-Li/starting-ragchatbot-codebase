#!/usr/bin/env python3
"""
Simple test script to diagnose the "query failed" issue.
"""

import sys
import os
from config import config
from rag_system import RAGSystem

def test_api_key():
    """Test if API key is configured."""
    print("=== API Key Test ===")
    if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-anthropic-api-key-here":
        print("❌ API key not configured properly")
        print(f"Current value: {config.ANTHROPIC_API_KEY}")
        return False
    else:
        print("✅ API key is configured")
        print(f"Key starts with: {config.ANTHROPIC_API_KEY[:8]}...")
        return True

def test_vector_store_setup():
    """Test if vector store has data."""
    print("\n=== Vector Store Test ===")
    try:
        rag = RAGSystem(config)
        analytics = rag.get_course_analytics()
        print(f"Total courses: {analytics['total_courses']}")
        print(f"Course titles: {analytics['course_titles']}")
        
        if analytics['total_courses'] == 0:
            print("❌ No courses loaded in vector store")
            return False
        else:
            print("✅ Vector store has course data")
            return True
            
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def test_simple_query():
    """Test a simple query to reproduce the issue."""
    print("\n=== Simple Query Test ===")
    try:
        rag = RAGSystem(config)
        response, sources = rag.query("What is computer use?")
        print(f"Response: {response}")
        print(f"Sources: {sources}")
        
        if "query failed" in response.lower():
            print("❌ Found 'query failed' error!")
            return False
        else:
            print("✅ Query executed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Query failed with exception: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("RAG System Diagnostic Test")
    print("=" * 40)
    
    results = []
    results.append(("API Key", test_api_key()))
    results.append(("Vector Store", test_vector_store_setup()))
    results.append(("Simple Query", test_simple_query()))
    
    print("\n=== Summary ===")
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    failed_tests = [name for name, passed in results if not passed]
    if failed_tests:
        print(f"\n🔍 Failed tests indicate potential causes of 'query failed': {failed_tests}")
    else:
        print("\n🎉 All tests passed - system appears to be working correctly")

if __name__ == "__main__":
    main()