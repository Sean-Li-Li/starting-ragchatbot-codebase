# RAG System Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                      │
│  ┌─────────────────┐    1. User Input     ┌──────────────────────────────────┐  │
│  │  User Interface │ ───────────────────► │      script.js                  │  │
│  │   (HTML/CSS)    │                      │   - sendMessage()                │  │
│  │                 │                      │   - Shows loading animation      │  │
│  │                 │                      │   - POST /api/query              │  │
│  └─────────────────┘                      └──────────────────────────────────┘  │
│                                                           │                     │
└───────────────────────────────────────────────────────────┼─────────────────────┘
                                                            │ 2. HTTP Request
                                                            │ {query, session_id}
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   BACKEND                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                            FastAPI (app.py)                             │  │
│  │   POST /api/query  ───► QueryRequest ───► rag_system.query()           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 3. Process Query                   │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                         RAG System (rag_system.py)                      │  │
│  │   - Build prompt: "Answer this question about course materials..."      │  │
│  │   - Get conversation history from SessionManager                        │  │
│  │   - Call ai_generator with tools and history                           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 4. Generate Response               │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        AI Generator (ai_generator.py)                   │  │
│  │   - System prompt with search tool instructions                         │  │
│  │   - Claude API call with tool_choice: "auto"                           │  │
│  │   - Decides whether to use search tool                                 │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 5. Tool Execution (if needed)      │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                       Tool Manager (search_tools.py)                    │  │
│  │   - execute_tool("search_course_content", query, filters)              │  │
│  │   - Routes to CourseSearchTool.execute()                               │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 6. Vector Search                   │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        Vector Store (vector_store.py)                   │  │
│  │   - search(query, course_name, lesson_number)                          │  │
│  │   - Resolve course name (fuzzy matching)                               │  │
│  │   - Build ChromaDB filters                                             │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 7. Semantic Search                 │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                            ChromaDB                                     │  │
│  │   - course_content.query() with embeddings                             │  │
│  │   - Semantic similarity search                                         │  │
│  │   - Return documents + metadata                                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 8. Format Results                  │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                     Course Search Tool (search_tools.py)                │  │
│  │   - _format_results() with course/lesson context                       │  │
│  │   - Track sources: ["Course A - Lesson 1", "Course B"]                 │  │
│  │   - Return formatted text to AI Generator                              │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 9. AI Synthesis                    │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        AI Generator (ai_generator.py)                   │  │
│  │   - Claude receives tool results                                       │  │
│  │   - Synthesizes search results into natural answer                     │  │
│  │   - Returns final response text                                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 10. Session Update                 │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                       Session Manager (session_manager.py)              │  │
│  │   - add_exchange(session_id, query, response)                          │  │
│  │   - Store for conversation context                                     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 11. Response Return                │
│                                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                            FastAPI (app.py)                             │  │
│  │   QueryResponse: {answer, sources, session_id}                         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 12. HTTP Response                  │
└───────────────────────────────────────────┼─────────────────────────────────────┘
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                      │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                            script.js                                    │  │
│  │   - Remove loading animation                                            │  │
│  │   - Display answer with markdown rendering                             │  │
│  │   - Show collapsible sources section                                   │  │
│  │   - Re-enable input for next query                                     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                           │ 13. Display to User                │
│                                           ▼                                    │
│  ┌─────────────────┐                                                          │
│  │  User Interface │ ◄─── Response displayed with sources                     │
│  │   (HTML/CSS)    │                                                          │
│  └─────────────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

FLOW SUMMARY:
User Input → FastAPI → RAG System → AI Generator → Tool Manager → 
Vector Store → ChromaDB Search → Result Formatting → AI Synthesis → 
Session Update → Frontend Display

KEY COMPONENTS:
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Component       │ Primary Function                                         │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ Frontend        │ User interface, HTTP requests, response display         │
│ FastAPI         │ HTTP endpoint routing, request/response handling        │
│ RAG System      │ Main orchestrator, coordinates all components           │
│ AI Generator    │ Claude API integration, tool calling, response synthesis│
│ Tool Manager    │ Tool registration and execution routing                 │
│ Vector Store    │ Semantic search coordination, ChromaDB interface       │
│ ChromaDB        │ Vector database, embedding storage, similarity search  │
│ Session Manager │ Conversation history, context preservation              │
└─────────────────┴──────────────────────────────────────────────────────────┘

DATA MODELS:
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Model           │ Purpose                                                  │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ Course          │ Course metadata (title, instructor, lessons)            │
│ Lesson          │ Individual lesson info (number, title, link)            │
│ CourseChunk     │ Text chunks for vector storage with course context      │
│ QueryRequest    │ API input (query, session_id)                          │
│ QueryResponse   │ API output (answer, sources, session_id)               │
│ SearchResults   │ Vector search results (documents, metadata, distances)  │
└─────────────────┴──────────────────────────────────────────────────────────┘
```