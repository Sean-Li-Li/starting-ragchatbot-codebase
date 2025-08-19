"""
Comprehensive tests for AIGenerator functionality.

This module tests the AIGenerator's ability to correctly call tools,
handle tool responses, and generate appropriate responses.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator basic functionality."""

    def test_init_with_valid_params(self):
        """Test AIGenerator initialization with valid parameters."""
        # Act
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Assert
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test generate_response() without tools (basic query)."""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [Mock(text="This is a basic response without tools.")]
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Act
        result = generator.generate_response("What is AI?")
        
        # Assert
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert "tools" not in call_args
        
        assert result == "This is a basic response without tools."

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test generate_response() includes conversation history in system prompt."""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [Mock(text="Response with history context.")]
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        history = "Previous conversation context"
        
        # Act
        result = generator.generate_response("Follow up question", conversation_history=history)
        
        # Assert
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation context" in call_args["system"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test generate_response() with tools available but not used."""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [Mock(text="Direct response without using tools.")]
        
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        
        mock_tool_manager = Mock()
        
        # Act
        result = generator.generate_response(
            "What is 2 + 2?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        # Assert
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Should not call tool manager since no tools were used
        mock_tool_manager.execute_tool.assert_not_called()
        
        assert result == "Direct response without using tools."

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test generate_response() when AI decides to use tools."""
        # Arrange
        mock_client = Mock()
        
        # First response: tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        # Mock tool content block
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_call_123"
        mock_tool_content.input = {"query": "computer use"}
        
        mock_tool_response.content = [mock_tool_content]
        
        # Second response: final answer
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Based on the search results, computer use refers to AI interacting with computers.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about computer use"
        
        # Act
        result = generator.generate_response(
            "What is computer use?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        # Assert
        # Should have made 2 API calls
        assert mock_client.messages.create.call_count == 2
        
        # First call should include tools
        first_call_args = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_args
        
        # Tool manager should have been called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="computer use"
        )
        
        # Second call should include tool results
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]
        
        # Should have: user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Tool results message should contain our mock result
        tool_results = messages[2]["content"]
        assert isinstance(tool_results, list)
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_call_123"
        assert tool_results[0]["content"] == "Search results about computer use"
        
        assert result == "Based on the search results, computer use refers to AI interacting with computers."

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic):
        """Test _handle_tool_execution() with multiple tool calls in one response."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [
            Mock(
                type="tool_use",
                name="search_course_content",
                id="tool_1",
                input={"query": "computer use"}
            ),
            Mock(
                type="tool_use", 
                name="get_course_outline",
                id="tool_2",
                input={"course_title": "Anthropic Course"}
            )
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search results for computer use",
            "Course outline results"
        ]
        
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Final response using both tool results")]
        mock_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Tell me about computer use and course outline"}],
            "system": "System prompt",
            "tools": [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        }
        
        # Act
        result = generator._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)
        
        # Assert
        # Should execute both tools
        assert mock_tool_manager.execute_tool.call_count == 2
        # Check that the correct tool names and parameters were used
        call_args_list = mock_tool_manager.execute_tool.call_args_list
        assert len(call_args_list) == 2
        
        # First call should be for search_course_content with query
        first_call_args, first_call_kwargs = call_args_list[0]
        assert first_call_kwargs == {"query": "computer use"}
        
        # Second call should be for get_course_outline with course_title
        second_call_args, second_call_kwargs = call_args_list[1]
        assert second_call_kwargs == {"course_title": "Anthropic Course"}
        
        # Final API call should include both tool results
        call_args = mock_client.messages.create.call_args[1]
        messages = call_args["messages"]
        
        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        tool_results = messages[2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        
        assert result == "Final response using both tool results"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic):
        """Test sequential tool calling with 2 rounds."""
        # Arrange
        mock_client = Mock()
        
        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_round1_content = Mock()
        mock_round1_content.type = "tool_use"
        mock_round1_content.name = "get_course_outline"
        mock_round1_content.id = "tool_round1"
        mock_round1_content.input = {"course_title": "Course X"}
        mock_round1_response.content = [mock_round1_content]
        
        # Round 2: Another tool use response
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_round2_content = Mock()
        mock_round2_content.type = "tool_use"
        mock_round2_content.name = "search_course_content"
        mock_round2_content.id = "tool_round2"
        mock_round2_content.input = {"query": "lesson 4 topic"}
        mock_round2_response.content = [mock_round2_content]
        
        # Final response: No more tools
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Based on both searches, here's the comprehensive answer.")]
        
        mock_client.messages.create.side_effect = [
            mock_round1_response, 
            mock_round2_response, 
            mock_final_response
        ]
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search course content"}
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course X outline with lesson 4: Advanced Topics",
            "Found courses discussing advanced topics"
        ]
        
        # Act
        result = generator.generate_response(
            "Find courses discussing same topic as lesson 4 of Course X",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        # Should have made 3 API calls (initial + 2 rounds)
        assert mock_client.messages.create.call_count == 3
        
        # Both tools should have been executed
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool execution parameters
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        assert tool_calls[0][0] == ("get_course_outline",)
        assert tool_calls[1][0] == ("search_course_content",)
        
        # Final response should be from the third API call
        assert result == "Based on both searches, here's the comprehensive answer."

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_early_termination(self, mock_anthropic):
        """Test sequential tool calling with early termination (no tools in round 1)."""
        # Arrange
        mock_client = Mock()
        
        # Round 1: Tool use response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_round1_content = Mock()
        mock_round1_content.type = "tool_use"
        mock_round1_content.name = "search_course_content"
        mock_round1_content.id = "tool_1"
        mock_round1_content.input = {"query": "test query"}
        mock_round1_response.content = [mock_round1_content]
        
        # Round 2: No tool use - early termination
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Direct answer without more tools.")]
        
        mock_client.messages.create.side_effect = [mock_round1_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [{"name": "search_course_content", "description": "Search content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        # Act
        result = generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        # Should have made 2 API calls (initial + 1 round, then terminated)
        assert mock_client.messages.create.call_count == 2
        
        # Only one tool should have been executed
        assert mock_tool_manager.execute_tool.call_count == 1
        
        assert result == "Direct answer without more tools."

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_reached(self, mock_anthropic):
        """Test sequential tool calling reaches max rounds (2)."""
        # Arrange
        mock_client = Mock()
        
        # Round 1: Tool use
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_round1_content = Mock()
        mock_round1_content.type = "tool_use"
        mock_round1_content.name = "search_course_content"
        mock_round1_content.id = "tool_1"
        mock_round1_content.input = {"query": "query1"}
        mock_round1_response.content = [mock_round1_content]
        
        # Round 2: Tool use (max rounds reached)
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_round2_content = Mock()
        mock_round2_content.type = "tool_use"
        mock_round2_content.name = "search_course_content"
        mock_round2_content.id = "tool_2"
        mock_round2_content.input = {"query": "query2"}
        mock_round2_response.content = [mock_round2_content]
        
        # Round 3: Final response (no tools since max reached)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Final response after max rounds.")]
        
        mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
            mock_final_response
        ]
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [{"name": "search_course_content", "description": "Search content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        # Act
        result = generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        # Should have made 3 API calls (initial + 2 tool rounds + final without tools)
        assert mock_client.messages.create.call_count == 3
        
        # Should have executed 2 tools (one per round)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Third API call should not include tools (max rounds reached)
        third_call_args = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_args
        
        assert result == "Final response after max rounds."

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_with_tool_error(self, mock_anthropic):
        """Test _handle_tool_execution() when tool execution fails."""
        # Arrange
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [
            Mock(
                type="tool_use",
                name="search_course_content",
                id="tool_1",
                input={"query": "computer use"}
            )
        ]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="I encountered an error while searching.")]
        mock_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Search query"}],
            "system": "System prompt",
            "tools": [{"name": "search_course_content"}]
        }
        
        # Act
        result = generator._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)
        
        # Assert
        # Should still pass the error message as tool result
        call_args = mock_client.messages.create.call_args[1]
        messages = call_args["messages"]
        tool_results = messages[2]["content"]
        
        assert tool_results[0]["content"] == "Tool execution failed: Database error"
        assert result == "I encountered an error while searching."

    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_anthropic):
        """Test how AIGenerator handles Anthropic API errors."""
        # Arrange
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API rate limit exceeded")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")
        
        assert "API rate limit exceeded" in str(exc_info.value)


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with realistic scenarios."""

    @patch('ai_generator.anthropic.Anthropic')
    def test_realistic_tool_calling_flow(self, mock_anthropic, mock_tool_manager):
        """Test a realistic flow from user query to tool execution to final response."""
        # Arrange
        mock_client = Mock()
        
        # Tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "computer use", "course_name": "Anthropic"}
        mock_tool_response.content = [mock_tool_content]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [Mock(text="Computer use refers to AI models' ability to interact with computers through screenshots and actions.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Configure mock tool manager
        mock_tool_manager.execute_tool.return_value = "[Building Towards Computer Use with Anthropic - Lesson 0]\nWelcome to Building Toward Computer Use with Anthropic. This course teaches you about computer use capabilities."
        
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
        
        # Act
        result = generator.generate_response(
            "What is computer use in the Anthropic course?",
            conversation_history="User: Hello\nAssistant: Hi there!",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Assert
        # Verify the complete flow
        assert mock_client.messages.create.call_count == 2
        
        # First call should have tools and conversation history
        first_call = mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call
        assert "Previous conversation" in first_call["system"]
        
        # Tool should have been executed with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="computer use",
            course_name="Anthropic"
        )
        
        # Final response should incorporate tool results
        assert result == "Computer use refers to AI models' ability to interact with computers through screenshots and actions."

    def test_system_prompt_content(self):
        """Test that the system prompt contains expected instructions."""
        # Arrange
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Act & Assert
        system_prompt = generator.SYSTEM_PROMPT
        
        assert "course materials" in system_prompt.lower()
        assert "search tools" in system_prompt.lower()
        assert "tool usage guidelines" in system_prompt.lower()
        assert "course outline" in system_prompt.lower()
        assert "course content" in system_prompt.lower()
        assert "brief, concise" in system_prompt.lower()

    def test_base_params_configuration(self):
        """Test that base parameters are configured correctly."""
        # Arrange & Act
        generator = AIGenerator("test_api_key", "claude-sonnet-4-20250514")
        
        # Assert
        base_params = generator.base_params
        assert base_params["model"] == "claude-sonnet-4-20250514"
        assert base_params["temperature"] == 0  # Should be deterministic
        assert base_params["max_tokens"] == 800  # Should be reasonable limit
        assert "system" not in base_params  # System should be added per request