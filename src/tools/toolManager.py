"""
Tool Manager

Manages tool execution outputs for context assembly.
In a real system, this would track actual tool calls (web search, calculator, etc.)
For this demo, we simulate tool outputs to show budget management.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque

from src.core.prioritizer import ToolOutput
from src.utils.token_counter import get_token_counter


class ToolManager:
    """
    Manages tool execution history.
    
    Tracks recent tool calls and their outputs for inclusion
    in context when relevant.
    """
    
    def __init__(self, max_outputs: int = 10):
        """
        Initialize tool manager.
        
        Args:
            max_outputs: Maximum number of outputs to keep
        """
        self.logger = logging.getLogger(__name__)
        self.token_counter = get_token_counter()
        
        # Store outputs as deque for efficient operations
        self.outputs = deque(maxlen=max_outputs)
        
        self.logger.info(f"Initialized ToolManager (max_outputs={max_outputs})")
    
    def add_tool_output(
        self,
        tool_name: str,
        output: str,
        success: bool = True
    ) -> None:
        """
        Add a tool execution output.
        
        Args:
            tool_name: Name of the tool that was executed
            output: The tool's output
            success: Whether execution was successful
        """
        tool_output = ToolOutput(
            content=f"[{tool_name}]\n{output}",
            timestamp=datetime.now().timestamp(),
            success=success,
            tokens=self.token_counter.count_tokens(output)
        )
        
        self.outputs.append(tool_output)
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.debug(
            f"Added tool output: {tool_name} ({status}, {tool_output.tokens} tokens)"
        )
    
    def get_recent_outputs(self, n: Optional[int] = None) -> List[ToolOutput]:
        """
        Get recent tool outputs.
        
        Args:
            n: Number of recent outputs (None = all)
            
        Returns:
            List of ToolOutput objects (newest last)
        """
        if n is None:
            return list(self.outputs)
        
        return list(self.outputs)[-n:] if n > 0 else []
    
    def get_successful_outputs(self) -> List[ToolOutput]:
        """
        Get only successful tool outputs.
        
        Returns:
            List of successful ToolOutput objects
        """
        return [output for output in self.outputs if output.success]
    
    def format_for_context(self) -> str:
        """
        Format recent outputs for context assembly.
        
        Returns:
            Formatted tool outputs string
        """
        if not self.outputs:
            return ""
        
        formatted_parts = []
        
        for output in self.outputs:
            status = "SUCCESS" if output.success else "FAILED"
            formatted_parts.append(
                f"[Tool Output - {status}]\n{output.content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def clear(self) -> None:
        """Clear all tool outputs"""
        self.outputs.clear()
        self.logger.info("Cleared tool outputs")
    
    def get_stats(self) -> Dict:
        """
        Get tool execution statistics.
        
        Returns:
            Dictionary with stats
        """
        total = len(self.outputs)
        successful = sum(1 for o in self.outputs if o.success)
        
        return {
            'total_outputs': total,
            'successful': successful,
            'failed': total - successful
        }
    
    def simulate_search(self, query: str, results: List[str]) -> None:
        """
        Simulate a search tool execution.
        
        Args:
            query: Search query
            results: List of result strings
        """
        output = f"Search Query: {query}\n\nResults:\n"
        output += "\n".join(f"- {result}" for result in results)
        
        self.add_tool_output("web_search", output, success=True)


_tool_manager_instance: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    """
    Get the global ToolManager instance.
    
    Returns:
        Singleton ToolManager instance
    """
    global _tool_manager_instance
    if _tool_manager_instance is None:
        _tool_manager_instance = ToolManager()
    return _tool_manager_instance

