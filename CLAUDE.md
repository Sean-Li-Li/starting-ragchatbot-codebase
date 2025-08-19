# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. It consists of:
- **Backend**: FastAPI application with RAG pipeline using ChromaDB for vector storage and Anthropic's Claude for AI generation
- **Frontend**: Simple HTML/CSS/JS web interface 
- **Core Architecture**: Modular design with separate components for document processing, vector storage, AI generation, and session management

## Development Commands

### Installation & Setup
```bash
# Install dependencies
uv sync

# Set up environment (create .env file with ANTHROPIC_API_KEY)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Code Quality & Development
```bash
# Format code (black + isort)
./scripts/format.sh

# Check code quality (linting)
./scripts/lint.sh

# Run tests
./scripts/test.sh

# Complete quality check (format + lint + test)
./scripts/quality-check.sh

# Individual tool commands
uv run black .                    # Format with black
uv run isort .                    # Sort imports
uv run flake8 .                   # Lint with flake8
cd backend && uv run pytest -v    # Run tests
```

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Components

### Core System (`backend/rag_system.py`)
The main orchestrator that coordinates:
- `DocumentProcessor`: Chunks course documents for vector storage
- `VectorStore`: ChromaDB-based semantic search using sentence transformers
- `AIGenerator`: Anthropic Claude integration for response generation
- `SessionManager`: Conversation history management
- `ToolManager`: Search tool registration and execution

### Key Backend Modules
- `app.py`: FastAPI application with CORS and static file serving
- `models.py`: Pydantic models for Course, Lesson, CourseChunk data structures
- `config.py`: Configuration management
- `search_tools.py`: Tool system for semantic search capabilities
- `vector_store.py`: ChromaDB vector database operations
- `document_processor.py`: Text chunking and preprocessing

### Frontend Structure
- `frontend/index.html`: Main web interface
- `frontend/script.js`: Client-side interaction logic
- `frontend/style.css`: UI styling

## Important Notes

- **CRITICAL**: Always use `uv` as the Python package manager - NEVER use pip directly
- The system uses `uv` for all Python operations (install, run, sync)
- Course documents are stored in `docs/` directory as `.txt` files
- Vector embeddings use sentence-transformers model
- Session state is managed in-memory (not persistent across restarts)
- The application serves both API endpoints and static frontend files

## Code Quality Standards

- **Black**: Python code formatter (88 character line length)
- **isort**: Import sorting (compatible with black)  
- **Flake8**: Linting and code quality checks
- **Pytest**: Testing framework
- All Python code must pass formatting, linting, and tests before commits
- Use `./scripts/quality-check.sh` to run all checks at once

## Package Management Rules

**ALWAYS USE `uv` - NEVER `pip`**:
```bash
# Correct commands
uv sync                    # Install dependencies
uv add package_name        # Add new dependency
uv run uvicorn app:app     # Run server
uv run python script.py   # Run Python scripts

# NEVER use these
pip install package_name   # ❌ WRONG
pip freeze                 # ❌ WRONG  
python -m pip install     # ❌ WRONG
```