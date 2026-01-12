"""
CLI Interface for Context-Aware RAG System

Provides command-line interface for interacting with the RAG system.
Demonstrates context assembly, budget management, and retrieval in action.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from src.core.context_assembler import get_context_assembler
from src.llm.client import get_gemini_client
from src.retrieval.vector_store import get_vector_store
from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences #user_preferences.py
from src.tools.toolManager import get_tool_manager

console = Console()


@click.group()
def cli():
    """Context-Window-Aware RAG System"""
    pass


@cli.command()
@click.option('--corpus-dir', default='./data/corpus', help='Directory containing corpus files')
def init(corpus_dir):
    """Initialize the vector store with corpus documents"""
    console.print("\n[bold blue]Initializing Vector Store...[/bold blue]\n")
    
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        if stats['document_count'] > 0:
            console.print(f"[yellow]Vector store already contains {stats['document_count']} documents[/yellow]")
            
            if click.confirm("Clear and reload?"):
                vector_store.clear_collection()
            else:
                console.print("[green]Using existing vector store[/green]")
                return
        
        # Load corpus
        loaded = vector_store.load_corpus_from_directory(corpus_dir)
        
        if loaded > 0:
            console.print(f"[green]Successfully loaded {loaded} documents![/green]")
            
            # Show stats
            stats = vector_store.get_collection_stats()
            table = Table(title="Vector Store Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            for key, value in stats.items():
                table.add_row(key, str(value))
            
            console.print(table)
        else:
            console.print(f"[red]No documents found in {corpus_dir}[/red]")
            console.print("[yellow]Make sure your corpus files exist in data/corpus/[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error initializing vector store: {e}[/red]")


@cli.command()
@click.argument('query')
@click.option('--show-context', is_flag=True, help='Display assembled context')
@click.option('--show-budget', is_flag=True, help='Display budget report')
@click.option('--no-retrieval', is_flag=True, help='Disable retrieval')
@click.option('--no-memory', is_flag=True, help='Disable conversation memory')
def query(query, show_context, show_budget, no_retrieval, no_memory):
    """Ask a question to the RAG system"""
    
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    
    try:
        # Assemble context
        assembler = get_context_assembler()
        assembled = assembler.assemble(
            query,
            include_retrieval=not no_retrieval,
            include_memory=not no_memory,
            include_tools=True
        )
        
        # Show budget report if requested
        if show_budget:
            report = assembler.get_assembly_report(assembled)
            console.print(Panel(report, title="Budget Report", border_style="blue"))
        
        # Show context if requested
        if show_context:
            full_context = assembled.get_full_context()
            console.print(Panel(
                full_context[:1000] + "\n...(truncated for display)",
                title="Assembled Context",
                border_style="green"
            ))
        
        # Generate response
        console.print("[yellow]Generating response...[/yellow]\n")
        
        client = get_gemini_client()
        response = client.generate_with_context(
            assembled.context_parts,
            temperature=0.7
        )
        
        # Display response
        console.print(Panel(
            Markdown(response),
            title="Response",
            border_style="green"
        ))
        
        # Add to memory
        memory = get_conversation_memory()
        memory.add_exchange(query, response)
        
        # Show token usage
        console.print(f"\n[dim]Token usage: {assembled.allocation.total}/{assembler.budget_manager.config.total}[/dim]")
        
        if assembled.truncation_applied:
            console.print("[yellow]Note: Truncation was applied to fit budget[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


@cli.command()
def demo_overflow():
    """Demonstrate budget overflow handling"""
    
    console.print("\n[bold magenta]BUDGET OVERFLOW DEMONSTRATION[/bold magenta]\n")
    console.print("This demo shows how the system handles context that exceeds budgets.\n")
    
    # Create overflow scenario
    memory = get_conversation_memory()
    memory.clear()
    
    console.print("[cyan]Step 1: Adding extensive conversation history...[/cyan]")
    for i in range(10):
        memory.add_exchange(
            f"Question {i+1} about machine learning, neural networks, and AI concepts",
            f"Detailed answer {i+1} explaining various technical aspects of deep learning systems"
        )
    console.print(f"  Added {len(memory.get_recent_exchanges())} exchanges\n")
    
    console.print("[cyan]Step 2: Adding tool execution outputs...[/cyan]")
    tools = get_tool_manager()
    tools.clear()
    
    for i in range(5):
        tools.add_tool_output(
            "web_search",
            f"Search result {i+1}: " + "Detailed technical information " * 50,
            success=True
        )
    console.print(f"  Added {len(tools.get_recent_outputs())} tool outputs\n")
    
    console.print("[cyan]Step 3: Creating long query...[/cyan]")
    long_query = """
    I'm working on understanding neural networks in depth. Can you explain:
    1. How backpropagation works with the chain rule
    2. Different optimization algorithms like SGD, Adam, RMSprop
    3. Regularization techniques including dropout and batch normalization
    4. Architecture patterns for CNNs and RNNs
    5. Best practices for hyperparameter tuning
    """ * 3
    
    console.print(f"  Query length: ~{len(long_query)} characters\n")
    
    console.print("[cyan]Step 4: Assembling context (will trigger truncation)...[/cyan]\n")
    
    assembler = get_context_assembler()
    assembled = assembler.assemble(long_query)
    
    # Show detailed report
    report = assembler.get_assembly_report(assembled)
    console.print(Panel(report, title="Assembly Report", border_style="yellow"))
    
    # Show truncation details
    if assembled.truncation_applied:
        console.print("\n[bold yellow]TRUNCATION DETAILS:[/bold yellow]")
        for section, detail in assembled.truncation_details.items():
            console.print(f"  • {section}: {detail}")
    
    console.print("\n[green]Demo complete! The system successfully handled overflow.[/green]")


@cli.command()
def stats():
    """Show system statistics"""
    
    console.print("\n[bold blue]SYSTEM STATISTICS[/bold blue]\n")
    
    # Vector store stats
    vector_store = get_vector_store()
    vs_stats = vector_store.get_collection_stats()
    
    table = Table(title="Vector Store")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in vs_stats.items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    # Memory stats
    memory = get_conversation_memory()
    mem_stats = memory.get_stats()
    
    table = Table(title="Conversation Memory")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in mem_stats.items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    # Tool stats
    tools = get_tool_manager()
    tool_stats = tools.get_stats()
    
    table = Table(title="Tool Manager")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in tool_stats.items():
        table.add_row(key, str(value))
    
    console.print(table)


@cli.command()
def clear():
    """Clear all conversation memory and tool history"""
    
    if click.confirm("Clear conversation memory and tool history?"):
        memory = get_conversation_memory()
        memory.clear()
        
        tools = get_tool_manager()
        tools.clear()
        
        console.print("[green]Memory and tools cleared![/green]")


@cli.command()
@click.option('--key', required=True, help='Preference key')
@click.option('--value', required=True, help='Preference value')
def set_pref(key, value):
    """Set a user preference"""
    
    prefs = get_user_preferences()
    prefs.set_preference(key, value)
    
    console.print(f"[green]Set preference: {key} = {value}[/green]")


@cli.command()
def show_prefs():
    """Show all user preferences"""
    
    prefs = get_user_preferences()
    all_prefs = prefs.get_all()
    
    if all_prefs:
        table = Table(title="User Preferences")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in all_prefs.items():
            table.add_row(key, str(value))
        
        console.print(table)
    else:
        console.print("[yellow]No preferences set[/yellow]")


if __name__ == '__main__':
    cli()


