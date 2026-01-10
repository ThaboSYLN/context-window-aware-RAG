"""
Prioritizer - Intelligent Truncation Strategies

This module implements context-aware truncation strategies for each
section type. The goal is to preserve meaning and intent when budgets
are exceeded, not just blindly cut text.

Truncation Strategies:
- Instructions: NEVER truncate (critical system behavior)
- Goal: Keep start + end, remove middle (preserves intent)
- Memory: FIFO queue (keep most recent)
- Retrieval: Rank by relevance score (keep best matches)
- Tool Outputs: Keep most recent successful outputs
"""

import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from typing import Optional
from src.utils.token_counter import get_token_counter


@dataclass
class RetrievalChunk:
    """Represents a retrieved document chunk with metadata"""
    content: str
    score: float
    source: str
    tokens: int


@dataclass
class ToolOutput:
    """Represents a tool execution output"""
    content: str
    timestamp: float
    success: bool
    tokens: int


class Prioritizer:
    """
    Implements intelligent truncation strategies for context sections.
    
    The key insight: different types of content require different
    truncation approaches to preserve utility.
    """
    
    def __init__(self):
        """Initialize prioritizer"""
        self.token_counter = get_token_counter()
        self.logger = logging.getLogger(__name__)
    
    def truncate_instructions(self, instructions: str, budget: int) -> str:
        """
        Truncate instructions section.
        
        Strategy: DO NOT TRUNCATE
        Instructions define core system behavior and must never be cut.
        If instructions exceed budget, this is a configuration error.
        
        Args:
            instructions: Instruction text
            budget: Token budget
            
        Returns:
            Original instructions (with error logged if over budget)
        """
        tokens = self.token_counter.count_tokens(instructions)
        
        if tokens > budget:
            self.logger.error(
                f"Instructions exceed budget ({tokens} > {budget}). "
                f"This is a configuration error. Instructions will NOT be truncated."
            )
        
        return instructions
    
    def truncate_goal(self, goal: str, budget: int) -> str:
        """
        Truncate goal/query while preserving intent.
        
        Strategy: Keep beginning and end, remove middle
        - Beginning: Context and setup (40% of budget)
        - End: Actual question/request (40% of budget)
        - Middle: Often contains less critical details (removed first)
        
        This preserves "I'm working on X... what is the best way to do Y?"
        instead of "I'm working on X and have tried A, B, C..."
        
        Args:
            goal: User's query/goal
            budget: Token budget
            
        Returns:
            Truncated goal that preserves intent
        """
        tokens = self.token_counter.count_tokens(goal)
        
        if tokens <= budget:
            return goal
        
        self.logger.info(f"Truncating goal from {tokens} to ~{budget} tokens")
        
        # Calculate character budgets (approximate)
        chars_per_token = self.token_counter.CHARS_PER_TOKEN
        total_chars = len(goal)
        target_chars = budget * chars_per_token
        
        # Keep 40% from start, 40% from end
        start_chars = int(target_chars * 0.4)
        end_chars = int(target_chars * 0.4)
        
        # Extract parts
        start_part = goal[:start_chars].strip()
        end_part = goal[-end_chars:].strip()
        
        # Try to break at sentence boundaries
        start_last_period = start_part.rfind('.')
        if start_last_period > start_chars * 0.7:
            start_part = start_part[:start_last_period + 1]
        
        end_first_period = end_part.find('.')
        if end_first_period != -1 and end_first_period < end_chars * 0.3:
            end_part = end_part[end_first_period + 1:].strip()
        
        # Combine with ellipsis
        truncated = f"{start_part} [...] {end_part}"
        
        final_tokens = self.token_counter.count_tokens(truncated)
        self.logger.debug(f"Goal truncated to {final_tokens} tokens")
        
        return truncated
    
    def truncate_memory(
        self,
        memory_items: List[str],
        budget: int
    ) -> str:
        """
        Truncate conversation memory.
        
        Strategy: FIFO queue - keep most recent exchanges
        Recent context is more relevant than older history.
        
        Args:
            memory_items: List of memory strings (newest last)
            budget: Token budget
            
        Returns:
            Truncated memory string
        """
        if not memory_items:
            return ""
        
        # Start with most recent and work backwards
        selected = []
        total_tokens = 0
        
        for item in reversed(memory_items):
            item_tokens = self.token_counter.count_tokens(item)
            
            if total_tokens + item_tokens <= budget:
                selected.insert(0, item)
                total_tokens += item_tokens
            else:
                break
        
        if len(selected) < len(memory_items):
            self.logger.info(
                f"Truncated memory: kept {len(selected)}/{len(memory_items)} items "
                f"({total_tokens} tokens)"
            )
        
        return "\n".join(selected)
    
    def truncate_retrieval(
        self,
        chunks: List[RetrievalChunk],
        budget: int
    ) -> str:
        """
        Truncate retrieval results.
        
        Strategy: Rank by relevance score, take top chunks
        Higher similarity scores = more relevant to query
        
        Args:
            chunks: List of retrieved chunks with scores
            budget: Token budget
            
        Returns:
            Truncated retrieval string with best matches
        """
        if not chunks:
            return ""
        
        # Sort by score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        
        # Select chunks that fit budget
        selected = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if total_tokens + chunk.tokens <= budget:
                selected.append(chunk)
                total_tokens += chunk.tokens
            else:
                # Try to fit a partial chunk if there's space
                remaining_budget = budget - total_tokens
                if remaining_budget > 50:  # Only if meaningful space left
                    partial_content = self.token_counter.truncate_to_budget(
                        chunk.content,
                        remaining_budget
                    )
                    partial_chunk = RetrievalChunk(
                        content=partial_content,
                        score=chunk.score,
                        source=chunk.source,
                        tokens=remaining_budget
                    )
                    selected.append(partial_chunk)
                break
        
        if len(selected) < len(chunks):
            self.logger.info(
                f"Truncated retrieval: kept {len(selected)}/{len(chunks)} chunks "
                f"({total_tokens} tokens)"
            )
        
        # Format selected chunks
        formatted_parts = []
        for i, chunk in enumerate(selected, 1):
            formatted_parts.append(
                f"[Source {i}: {chunk.source}, Relevance: {chunk.score:.3f}]\n{chunk.content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def truncate_tool_outputs(
        self,
        outputs: List[ToolOutput],
        budget: int
    ) -> str:
        """
        Truncate tool outputs.
        
        Strategy: Keep most recent successful outputs
        - Prioritize successful executions over failures
        - Keep most recent (more relevant to current task)
        
        Args:
            outputs: List of tool outputs
            budget: Token budget
            
        Returns:
            Truncated tool outputs string
        """
        if not outputs:
            return ""
        
        # Sort by success (successful first), then timestamp (recent first)
        sorted_outputs = sorted(
            outputs,
            key=lambda x: (x.success, x.timestamp),
            reverse=True
        )
        
        # Select outputs that fit budget
        selected = []
        total_tokens = 0
        
        for output in sorted_outputs:
            if total_tokens + output.tokens <= budget:
                selected.append(output)
                total_tokens += output.tokens
            else:
                break
        
        if len(selected) < len(outputs):
            self.logger.info(
                f"Truncated tool outputs: kept {len(selected)}/{len(outputs)} outputs "
                f"({total_tokens} tokens)"
            )
        
        # Format selected outputs
        formatted_parts = []
        for i, output in enumerate(selected, 1):
            status = "SUCCESS" if output.success else "FAILED"
            formatted_parts.append(f"[Output {i} - {status}]\n{output.content}")
        
        return "\n\n".join(formatted_parts)
    
    def prioritize_sections(
        self,
        overages: Dict[str, int]
    ) -> List[Tuple[str, int, str]]:
        """
        Determine truncation order when multiple sections exceed budget.
        
        Priority (most to least important):
        1. Instructions - NEVER truncate
        2. Goal - User's current need
        3. Retrieval - Relevant knowledge
        4. Tool Outputs - Recent actions
        5. Memory - Conversation history
        
        Args:
            overages: Dictionary of section -> overage amount
            
        Returns:
            List of (section, overage, strategy) tuples in truncation order
        """
        priority_order = [
            ('memory', 'FIFO queue'),
            ('tool_outputs', 'Keep recent successful'),
            ('retrieval', 'Keep highest relevance'),
            ('goal', 'Keep start and end'),
            ('instructions', 'NEVER TRUNCATE'),
        ]
        
        truncation_plan = []
        for section, strategy in priority_order:
            if section in overages:
                truncation_plan.append((section, overages[section], strategy))
        
        return truncation_plan


# Global singleton instance
_prioritizer_instance: Optional['Prioritizer'] = None


def get_prioritizer() -> Prioritizer:
    """
    Get the global Prioritizer instance.
    
    Returns:
        Singleton Prioritizer instance
    """
    global _prioritizer_instance
    if _prioritizer_instance is None:
        _prioritizer_instance = Prioritizer()
    return _prioritizer_instance


