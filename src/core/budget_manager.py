"""
Budget Manager

Enforces token budgets across all context sections.
This is the core constraint system that ensures we never exceed
the total context window limit.

Budget Allocation:
- Instructions: 255 tokens (NEVER truncated)
- Goal: 1,500 tokens
- Memory: 55 tokens
- Retrieval: 550 tokens
- Tool Outputs: 855 tokens
- Total: 3,215 tokens
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.utils.token_counter import get_token_counter


@dataclass
class BudgetConfig:
    """Configuration for token budgets"""
    instructions: int = 255
    goal: int = 1500
    memory: int = 55
    retrieval: int = 550
    tool_outputs: int = 855
    
    @property
    def total(self) -> int:
        """Calculate total budget"""
        return (
            self.instructions +
            self.goal +
            self.memory +
            self.retrieval +
            self.tool_outputs
        )


@dataclass
class BudgetAllocation:
    """Actual token allocation for a context assembly"""
    instructions: int
    goal: int
    memory: int
    retrieval: int
    tool_outputs: int
    
    @property
    def total(self) -> int:
        """Calculate total allocated tokens"""
        return (
            self.instructions +
            self.goal +
            self.memory +
            self.retrieval +
            self.tool_outputs
        )
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for display"""
        return {
            'instructions': self.instructions,
            'goal': self.goal,
            'memory': self.memory,
            'retrieval': self.retrieval,
            'tool_outputs': self.tool_outputs,
            'total': self.total
        }


class BudgetManager:
    """
    Manages token budgets for context assembly.
    """
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        """
        Initialize budget manager.
        """
        self.config = config or BudgetConfig()
        self.token_counter = get_token_counter()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized BudgetManager with total budget: {self.config.total} tokens")
    
    def check_section_budget(
        self,
        content: str,
        section: str,
        budget: int
    ) -> Tuple[bool, int, int]:
        """
        Check if content fits within section budget.
        
        Args:
            content: Text content to check
            section: Section name (for logging)
            budget: Token budget for this section
            
        Returns:
            Tuple of (fits_budget, actual_tokens, budget)
        """
        actual_tokens = self.token_counter.count_tokens(content)
        fits = actual_tokens <= budget
        
        if not fits:
            self.logger.warning(
                f"Section '{section}' exceeds budget: {actual_tokens} > {budget} "
                f"(overage: {actual_tokens - budget} tokens)"
            )
        else:
            self.logger.debug(
                f"Section '{section}' within budget: {actual_tokens}/{budget} tokens"
            )
        
        return fits, actual_tokens, budget
    
    def validate_context(
        self,
        context_parts: Dict[str, str]
    ) -> Tuple[bool, BudgetAllocation, Dict[str, int]]:
        """
        Validate all context parts against budgets.
        
        Args:
            context_parts: Dictionary with section names as keys
            
        Returns:
            Tuple of (all_within_budget, allocation, overages)
            where overages is a dict of section -> overage amount
        """
        allocation = BudgetAllocation(
            instructions=self.token_counter.count_tokens(
                context_parts.get('instructions', '')
            ),
            goal=self.token_counter.count_tokens(
                context_parts.get('goal', '')
            ),
            memory=self.token_counter.count_tokens(
                context_parts.get('memory', '')
            ),
            retrieval=self.token_counter.count_tokens(
                context_parts.get('retrieval', '')
            ),
            tool_outputs=self.token_counter.count_tokens(
                context_parts.get('tool_outputs', '')
            )
        )
        
        # Calculate overages
        overages = {}
        
        if allocation.instructions > self.config.instructions:
            overages['instructions'] = allocation.instructions - self.config.instructions
        
        if allocation.goal > self.config.goal:
            overages['goal'] = allocation.goal - self.config.goal
        
        if allocation.memory > self.config.memory:
            overages['memory'] = allocation.memory - self.config.memory
        
        if allocation.retrieval > self.config.retrieval:
            overages['retrieval'] = allocation.retrieval - self.config.retrieval
        
        if allocation.tool_outputs > self.config.tool_outputs:
            overages['tool_outputs'] = allocation.tool_outputs - self.config.tool_outputs
        
        all_within_budget = len(overages) == 0
        
        if not all_within_budget:
            self.logger.warning(f"Budget violations detected: {overages}")
        
        # Check total budget
        if allocation.total > self.config.total:
            total_overage = allocation.total - self.config.total
            self.logger.error(
                f"TOTAL BUDGET EXCEEDED: {allocation.total} > {self.config.total} "
                f"(overage: {total_overage} tokens)"
            )
        
        return all_within_budget, allocation, overages
    
    def get_section_budget(self, section: str) -> int:
        """
        Get budget for a specific section.
        
        Args:
            section: Section name
            
        Returns:
            Token budget for section
        """
        budgets = {
            'instructions': self.config.instructions,
            'goal': self.config.goal,
            'memory': self.config.memory,
            'retrieval': self.config.retrieval,
            'tool_outputs': self.config.tool_outputs
        }
        return budgets.get(section, 0)
    
    def calculate_available_space(
        self,
        current_allocation: BudgetAllocation
    ) -> Dict[str, int]:
        """
        Calculate remaining space in each section.
        
        Args:
            current_allocation: Current token usage
            
        Returns:
            Dictionary of section -> remaining tokens
        """
        return {
            'instructions': self.config.instructions - current_allocation.instructions,
            'goal': self.config.goal - current_allocation.goal,
            'memory': self.config.memory - current_allocation.memory,
            'retrieval': self.config.retrieval - current_allocation.retrieval,
            'tool_outputs': self.config.tool_outputs - current_allocation.tool_outputs
        }
    
    def format_budget_report(
        self,
        allocation: BudgetAllocation
    ) -> str:
        """
        Create a human-readable budget report.
        
        Args:
            allocation: Current allocation
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "CONTEXT WINDOW BUDGET REPORT",
            "=" * 60,
        ]
        
        sections = [
            ('Instructions', allocation.instructions, self.config.instructions),
            ('Goal', allocation.goal, self.config.goal),
            ('Memory', allocation.memory, self.config.memory),
            ('Retrieval', allocation.retrieval, self.config.retrieval),
            ('Tool Outputs', allocation.tool_outputs, self.config.tool_outputs),
        ]
        
        for name, used, budget in sections:
            percentage = (used / budget * 100) if budget > 0 else 0
            status = "OK" if used <= budget else "EXCEEDED"
            bar_length = 30
            filled = int(bar_length * min(used / budget, 1.0))
            bar = "=" * filled + "-" * (bar_length - filled)
            
            lines.append(
                f"{name:15} [{bar}] {used:4}/{budget:4} ({percentage:5.1f}%) {status}"
            )
        
        lines.extend([
            "-" * 60,
            f"{'TOTAL':15} {allocation.total:4}/{self.config.total:4} tokens",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# Global singleton instance
_budget_manager_instance: Optional[BudgetManager] = None


def get_budget_manager() -> BudgetManager:
    """
    Get the global BudgetManager instance.
    
    Returns:
        Singleton BudgetManager instance
    """
    global _budget_manager_instance
    if _budget_manager_instance is None:
        _budget_manager_instance = BudgetManager()
    return _budget_manager_instance

