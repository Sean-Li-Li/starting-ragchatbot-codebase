import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Course Content Search** - For searching specific course materials and detailed educational content
2. **Course Outline** - For getting complete course structure, including course title, course link, and all lessons with their numbers and titles

Tool Usage Guidelines:
- **Course outline queries**: Use the course outline tool for questions about course structure, lesson lists, or complete course information
- **Course content queries**: Use the content search tool for questions about specific topics, concepts, or detailed materials within courses
- **Maximum 2 sequential tool rounds per query** - You can make additional tool calls based on previous results to provide comprehensive answers
- **Sequential reasoning**: Use results from previous tool calls to inform subsequent tool usage
- **Tool chaining**: For complex queries, you may first get course outlines, then search specific content based on that information
- Synthesize all tool results into accurate, fact-based responses
- If any tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Complex queries**: May require multiple tool calls for comprehensive answers (max 2 rounds)
- **Course outline + content**: First get outline, then search specific content as needed
- **Comparative queries**: Use multiple searches to gather information for comparisons
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or query analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value  
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle sequential tool execution with up to 2 rounds.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Track state
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 1
        max_rounds = 2
        
        # Sequential tool execution loop
        while (round_count <= max_rounds and 
               current_response.stop_reason == "tool_use" and 
               self._has_tool_use_blocks(current_response)):
            
            # Add assistant's tool use response to conversation
            messages.append({"role": "assistant", "content": current_response.content})
            
            # Execute tools and get results
            tool_results = self._execute_tools(current_response, tool_manager)
            if not tool_results:  # Tool execution failed
                break
                
            # Add tool results to conversation  
            messages.append({"role": "user", "content": tool_results})
            
            # Prepare next API call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }
            
            # Keep tools available for potential next round
            if round_count < max_rounds:
                next_params["tools"] = base_params.get("tools", [])
                next_params["tool_choice"] = {"type": "auto"}
            
            # Get next response
            try:
                current_response = self.client.messages.create(**next_params)
                round_count += 1
            except Exception as e:
                # Handle API errors gracefully
                return f"Tool execution failed in round {round_count}: {str(e)}"
        
        # Return final response text
        return current_response.content[0].text
    
    def _has_tool_use_blocks(self, response) -> bool:
        """Check if response contains tool_use blocks"""
        return any(block.type == "tool_use" for block in response.content)
    
    def _execute_tools(self, response, tool_manager):
        """Execute all tool calls in a response and return results"""
        tool_results = []
        
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Return error result for this tool
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        return tool_results if tool_results else None