"""
CLI Interface for Enhanced Context-Aware RAG System with Web Scraping

Provides command-line interface demonstrating:
- Automatic web scraping based on queries
- Smart caching to avoid re-scraping
- Budget enforcement with corpus + web sources
- Source transparency (corpus vs web breakdown)
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
from rich.progress import track

from src.core.context_assembler_enhanced import get_context_assembler
from src.llm.client import get_gemini_client
from src.retrieval.vector_store import get_vector_store
from src.retrieval.enhanced_retriever import get_enhanced_retriever
from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences
from src.tools.toolManager import get_tool_manager

console = Console()


@click.group()
def cli():
    """Enhanced Context-Window-Aware RAG System with Web Scraping"""
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
@click.option('--show-sources', is_flag=True, help='Show retrieval source breakdown')
@click.option('--no-retrieval', is_flag=True, help='Disable retrieval')
@click.option('--no-memory', is_flag=True, help='Disable conversation memory')
@click.option('--no-web', is_flag=True, help='Disable web scraping (corpus only)')
def query(query, show_context, show_budget, show_sources, no_retrieval, no_memory, no_web):
    """Ask a question to the RAG system (with automatic web scraping)"""
    
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    
    try:
        # Assemble context (web scraping happens automatically if needed)
        assembler = get_context_assembler()
        
        console.print("[yellow]Assembling context (checking corpus + web)...[/yellow]\n")
        
        assembled = assembler.assemble(
            query,
            include_retrieval=not no_retrieval,
            include_memory=not no_memory,
            include_tools=True
        )
        
        # Show source breakdown if requested
        if show_sources or show_budget:
            table = Table(title="Retrieval Source Breakdown")
            table.add_column("Source", style="cyan")
            table.add_column("Count", style="magenta")
            
            table.add_row("Corpus Documents", str(assembled.retrieval_sources['corpus']))
            table.add_row("Web Sources", str(assembled.retrieval_sources['web']))
            
            console.print(table)
            console.print()
        
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
        
        # Show web scraping info
        if assembled.retrieval_sources['web'] > 0:
            console.print(f"[cyan]✓ Web sources used: {assembled.retrieval_sources['web']} pages scraped[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


@cli.command()
def demo_web():
    """Demonstrate web scraping with a recency-based query"""
    
    console.print("\n[bold magenta]WEB SCRAPING DEMONSTRATION[/bold magenta]\n")
    console.print("This demo shows automatic web scraping for queries needing current information.\n")
    
    # Queries that should trigger web scraping
    test_queries = [
        "What are the latest developments in AI?",
        "Current trends in machine learning 2026",
        "Recent breakthroughs in neural networks"
    ]
    
    console.print("[cyan]Testing queries that should trigger web search:[/cyan]\n")
    
    for test_query in test_queries:
        console.print(f"[yellow]Query:[/yellow] {test_query}")
        
        try:
            assembler = get_context_assembler()
            assembled = assembler.assemble(test_query)
            
            # Show results
            if assembled.retrieval_sources['web'] > 0:
                console.print(f"  [green]✓ Web scraping triggered ({assembled.retrieval_sources['web']} sources)[/green]")
            else:
                console.print(f"  [blue]○ Using corpus only ({assembled.retrieval_sources['corpus']} sources)[/blue]")
            
            console.print(f"  [dim]Total tokens: {assembled.allocation.retrieval}/{assembler.budget_manager.config.retrieval}[/dim]\n")
        
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]\n")
    
    console.print("[green]Demo complete![/green]")


@cli.command()
def cache_stats():
    """Show web scraping cache statistics"""
    
    console.print("\n[bold blue]WEB SCRAPING CACHE STATISTICS[/bold blue]\n")
    
    try:
        retriever = get_enhanced_retriever()
        stats = retriever.get_stats()
        
        # Web cache stats
        console.print("[cyan]Web Scraping Cache:[/cyan]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in stats['web_cache_stats'].items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
        # Corpus stats
        console.print("\n[cyan]Local Corpus:[/cyan]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in stats['corpus_stats'].items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def clear_web_cache():
    """Clear the web scraping cache"""
    
    if click.confirm("Clear web scraping cache?"):
        try:
            retriever = get_enhanced_retriever()
            retriever.web_scraper.clear_cache()
            console.print("[green]Web cache cleared![/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
def stats():
    """Show complete system statistics"""
    
    console.print("\n[bold blue]SYSTEM STATISTICS[/bold blue]\n")
    
    # Retriever stats (includes web cache)
    retriever = get_enhanced_retriever()
    retriever_stats = retriever.get_stats()
    
    # Corpus stats
    table = Table(title="Local Corpus")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in retriever_stats['corpus_stats'].items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    # Web cache stats
    table = Table(title="Web Scraping Cache")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for key, value in retriever_stats['web_cache_stats'].items():
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


