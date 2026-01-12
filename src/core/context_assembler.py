"""
Context Assembler - Main Orchestrator

This is the core of the RAG system. It:
1. Takes a user query
2. Gathers context from all sources (instructions, memory, retrieval, tools)
3. Enforces token budgets
4. Applies prioritization when budgets are exceeded
5. Assembles the final context for the LLM

This demonstrates explicit context window management - the key innovation
being evaluated in this project.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.core.budget_manager import get_budget_manager, BudgetAllocation
from src.core.prioritizer import get_prioritizer
from src.retrieval.retriever import get_retriever
from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences
from src.tools.toolManager import get_tool_manager
from src.utils.token_counter import get_token_counter


@dataclass
class AssembledContext:
    """
    Represents the final assembled context ready for LLM.
    
    Contains both the context parts and metadata about the assembly process.
    """
    context_parts: Dict[str, str]
    allocation: BudgetAllocation
    truncation_applied: bool
    truncation_details: Dict[str, str]
    
    def get_full_context(self) -> str:
        """
        Get the complete assembled context as a single string.
        
        Returns:
            Formatted context ready for LLM
        """
        parts = []
        
        if self.context_parts.get('instructions'):
            parts.append(f"=== INSTRUCTIONS ===\n{self.context_parts['instructions']}\n")
        
        if self.context_parts.get('memory'):
            parts.append(f"=== CONVERSATION HISTORY ===\n{self.context_parts['memory']}\n")
        
        if self.context_parts.get('retrieval'):
            parts.append(f"=== RELEVANT KNOWLEDGE ===\n{self.context_parts['retrieval']}\n")
        
        if self.context_parts.get('tool_outputs'):
            parts.append(f"=== RECENT ACTIONS ===\n{self.context_parts['tool_outputs']}\n")
        
        if self.context_parts.get('goal'):
            parts.append(f"=== USER QUERY ===\n{self.context_parts['goal']}")
        
        return "\n".join(parts)


class ContextAssembler:
    """
    Orchestrates context assembly with explicit budget management.
    
    This is the main class that demonstrates context economics -
    making deliberate choices about what information to include
    when space is limited.
    """
    
    def __init__(self, instructions: Optional[str] = None):
        """
        Initialize context assembler.
        
        Args:
            instructions: System instructions (defaults to built-in)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.budget_manager = get_budget_manager()
        self.prioritizer = get_prioritizer()
        self.retriever = get_retriever()
        self.conversation_memory = get_conversation_memory()
        self.user_preferences = get_user_preferences()
        self.tool_manager = get_tool_manager()
        self.token_counter = get_token_counter()
        
        # Set default instructions
        self.instructions = instructions or self._get_default_instructions()
        
        self.logger.info("Initialized ContextAssembler")
    
    def _get_default_instructions(self) -> str:
        """
        Get default system instructions.
        
        Returns:
            Default instruction text
        """
        return """You are a helpful AI assistant with access to relevant knowledge.
Your role is to provide accurate, informative responses based on:
1. The retrieved knowledge provided
2. Your general understanding
3. The conversation context

Always cite sources when using retrieved information.
Be concise but thorough in your explanations."""
    
    def assemble(
        self,
        query: str,
        include_retrieval: bool = True,
        include_memory: bool = True,
        include_tools: bool = True
    ) -> AssembledContext:
        """
        Assemble context for a user query.
        
        This is the main method that demonstrates context window management.
        
        Args:
            query: User's query/goal
            include_retrieval: Whether to include retrieval results
            include_memory: Whether to include conversation memory
            include_tools: Whether to include tool outputs
            
        Returns:
            AssembledContext with all parts and metadata
        """
        self.logger.info(f"Assembling context for query: {query[:50]}...")
        
        # Step 1: Gather raw context from all sources
        raw_context = self._gather_context(
            query,
            include_retrieval,
            include_memory,
            include_tools
        )
        
        # Step 2: Validate against budgets
        valid, allocation, overages = self.budget_manager.validate_context(raw_context)
        
        # Step 3: Apply truncation if needed
        if not valid:
            self.logger.warning("Budget exceeded, applying truncation strategies")
            final_context, truncation_details = self._apply_truncation(
                raw_context,
                overages
            )
            truncation_applied = True
        else:
            self.logger.info("All sections within budget")
            final_context = raw_context
            truncation_details = {}
            truncation_applied = False
        
        # Step 4: Validate final context
        final_valid, final_allocation, final_overages = \
            self.budget_manager.validate_context(final_context)
        
        if not final_valid:
            self.logger.error(
                f"Context still exceeds budget after truncation: {final_overages}"
            )
        
        # Step 5: Create assembled context
        assembled = AssembledContext(
            context_parts=final_context,
            allocation=final_allocation,
            truncation_applied=truncation_applied,
            truncation_details=truncation_details
        )
        
        self.logger.info(f"Context assembled: {final_allocation.total} tokens")
        
        return assembled
    
    def _gather_context(
        self,
        query: str,
        include_retrieval: bool,
        include_memory: bool,
        include_tools: bool
    ) -> Dict[str, str]:
        """
        Gather context from all sources.
        
        Args:
            query: User query
            include_retrieval: Include retrieval
            include_memory: Include memory
            include_tools: Include tools
            
        Returns:
            Dictionary of context parts
        """
        context = {}
        
        # Instructions (always included)
        context['instructions'] = self.instructions
        
        # Goal (user query + preferences)
        goal_parts = [query]
        
        user_prefs = self.user_preferences.format_for_context()
        if user_prefs:
            goal_parts.append(f"\n{user_prefs}")
        
        context['goal'] = "\n".join(goal_parts)
        
        # Memory (conversation history)
        if include_memory:
            memory_text = self.conversation_memory.format_for_context()
            context['memory'] = memory_text if memory_text else ""
        else:
            context['memory'] = ""
        
        # Retrieval (semantic search)
        if include_retrieval:
            retrieval_budget = self.budget_manager.config.retrieval
            retrieval_text = self.retriever.retrieve_formatted(query, retrieval_budget)
            context['retrieval'] = retrieval_text if retrieval_text else ""
        else:
            context['retrieval'] = ""
        
        # Tool outputs
        if include_tools:
            tool_text = self.tool_manager.format_for_context()
            context['tool_outputs'] = tool_text if tool_text else ""
        else:
            context['tool_outputs'] = ""
        
        return context
    
    def _apply_truncation(
        self,
        context: Dict[str, str],
        overages: Dict[str, int]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Apply truncation strategies to exceed sections.
        
        Args:
            context: Raw context parts
            overages: Sections that exceed budget
            
        Returns:
            Tuple of (truncated_context, truncation_details)
        """
        truncated = context.copy()
        details = {}
        
        # Get truncation priority order
        truncation_plan = self.prioritizer.prioritize_sections(overages)
        
        for section, overage, strategy in truncation_plan:
            self.logger.info(f"Truncating {section} (overage: {overage}, strategy: {strategy})")
            
            budget = self.budget_manager.get_section_budget(section)
            
            if section == 'instructions':
                # Should never happen, but log error
                self.logger.error("Instructions exceed budget - configuration error!")
                truncated[section] = self.prioritizer.truncate_instructions(
                    context[section],
                    budget
                )
                details[section] = "ERROR: Instructions should never be truncated"
            
            elif section == 'goal':
                truncated[section] = self.prioritizer.truncate_goal(
                    context[section],
                    budget
                )
                details[section] = f"Truncated using start+end strategy to {budget} tokens"
            
            elif section == 'memory':
                memory_items = self.conversation_memory.get_memory_items()
                truncated[section] = self.prioritizer.truncate_memory(
                    memory_items,
                    budget
                )
                details[section] = f"Kept most recent exchanges within {budget} tokens"
            
            elif section == 'retrieval':
                # Already handled by retriever, but might need additional truncation
                current_tokens = self.token_counter.count_tokens(context[section])
                if current_tokens > budget:
                    truncated[section] = self.token_counter.truncate_to_budget(
                        context[section],
                        budget
                    )
                    details[section] = f"Hard truncated to {budget} tokens"
                else:
                    truncated[section] = context[section]
            
            elif section == 'tool_outputs':
                tool_outputs = self.tool_manager.get_recent_outputs()
                truncated[section] = self.prioritizer.truncate_tool_outputs(
                    tool_outputs,
                    budget
                )
                details[section] = f"Kept recent successful outputs within {budget} tokens"
        
        return truncated, details
    
    def get_assembly_report(self, assembled: AssembledContext) -> str:
        """
        Generate a detailed report of the assembly process.
        
        Args:
            assembled: Assembled context
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "CONTEXT ASSEMBLY REPORT",
            "=" * 70,
        ]
        
        # Budget report
        lines.append("\n" + self.budget_manager.format_budget_report(assembled.allocation))
        
        # Truncation info
        if assembled.truncation_applied:
            lines.append("\nTRUNCATION APPLIED:")
            lines.append("-" * 70)
            for section, detail in assembled.truncation_details.items():
                lines.append(f"  {section}: {detail}")
        else:
            lines.append("\nNo truncation needed - all sections within budget")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


_context_assembler_instance: Optional[ContextAssembler] = None


def get_context_assembler() -> ContextAssembler:
    """
    Get the global ContextAssembler instance.
    
    Returns:
        Singleton ContextAssembler instance
    """
    global _context_assembler_instance
    if _context_assembler_instance is None:
        _context_assembler_instance = ContextAssembler()
    return _context_assembler_instance

