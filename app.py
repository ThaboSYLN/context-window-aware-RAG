"""
Streamlit UI for Context-Window-Aware RAG System

A clean interface demonstrating:
- Context assembly with budget management
- Real-time budget visualization
- Budget overflow demonstration
- System statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import streamlit as st
from src.core.context_assembler import get_context_assembler
from src.llm.client import get_gemini_client
from src.retrieval.vector_store import get_vector_store
from src.memory.convo_memory import get_conversation_memory
from src.memory.user_preferences import get_user_preferences
from src.tools.toolManager import get_tool_manager
from src.core.budget_manager import get_budget_manager

# Page configuration
st.set_page_config(
    page_title="Context-Aware RAG System",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []

def initialize_system():
    """Initialize the RAG system"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        
        if stats['document_count'] == 0:
            # Load corpus
            loaded = vector_store.load_corpus_from_directory('./data/corpus')
            if loaded > 0:
                st.success(f"[SUCCESS] Loaded {loaded} documents into vector store")
            else:
                st.warning("[WARNING] No documents found in ./data/corpus")
        
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"[ERROR] Initialization failed: {str(e)}")
        return False

def display_budget_bars(allocation):
    """Display budget usage as progress bars"""
    budget_manager = get_budget_manager()
    config = budget_manager.config
    
    sections = [
        ('Instructions', allocation.instructions, config.instructions),
        ('Goal', allocation.goal, config.goal),
        ('Memory', allocation.memory, config.memory),
        ('Retrieval', allocation.retrieval, config.retrieval),
        ('Tool Outputs', allocation.tool_outputs, config.tool_outputs),
    ]
    
    for name, used, budget in sections:
        percentage = (used / budget) if budget > 0 else 0
        status = "[OK]" if used <= budget else "[OVER]"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(min(percentage, 1.0))
        with col2:
            st.text(f"{status} {used}/{budget}")
        st.caption(f"{name} ({percentage*100:.1f}%)")

def display_context_section(title, content, max_chars=500):
    """Display a collapsible context section"""
    if content:
        with st.expander(f"[SECTION] {title} ({len(content)} chars)"):
            display_content = content[:max_chars]
            if len(content) > max_chars:
                display_content += "\n\n... (truncated for display)"
            st.text(display_content)
    else:
        st.caption(f"[SECTION] {title}: Empty")

# Main UI
st.title("Context-Window-Aware RAG System")
st.markdown("**Demonstrating explicit context budget management**")

# Sidebar
with st.sidebar:
    st.header("System Controls")
    
    # Initialize button
    if st.button("[INIT] Initialize System", use_container_width=True):
        initialize_system()
    
    # System stats
    if st.session_state.initialized:
        st.divider()
        st.subheader("System Statistics")
        
        try:
            # Vector store stats
            vector_store = get_vector_store()
            vs_stats = vector_store.get_collection_stats()
            st.metric("Documents in Store", vs_stats['document_count'])
            
            # Memory stats
            memory = get_conversation_memory()
            mem_stats = memory.get_stats()
            st.metric("Conversation Exchanges", mem_stats['exchanges_in_memory'])
            
            # Tool stats
            tools = get_tool_manager()
            tool_stats = tools.get_stats()
            st.metric("Tool Outputs", tool_stats['total_outputs'])
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    
    st.divider()
    
    # Clear buttons
    st.subheader("Clear Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Memory", use_container_width=True):
            get_conversation_memory().clear()
            st.success("Memory cleared")
    with col2:
        if st.button("Clear Tools", use_container_width=True):
            get_tool_manager().clear()
            st.success("Tools cleared")

# Main content area
if not st.session_state.initialized:
    st.info("[INFO] Click 'Initialize System' in the sidebar to begin")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Query System", "Demo Overflow", "View Context"])

# Tab 1: Query System
with tab1:
    st.header("Ask a Question")
    
    # Query input
    user_query = st.text_area(
        "Enter your query:",
        placeholder="e.g., How do neural networks learn?",
        height=100
    )
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        include_retrieval = st.checkbox("Include Retrieval", value=True)
    with col2:
        include_memory = st.checkbox("Include Memory", value=True)
    with col3:
        include_tools = st.checkbox("Include Tools", value=True)
    
    # Submit button
    if st.button("[SUBMIT] Submit Query", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Assembling context and generating response..."):
                try:
                    # Assemble context
                    assembler = get_context_assembler()
                    assembled = assembler.assemble(
                        user_query,
                        include_retrieval=include_retrieval,
                        include_memory=include_memory,
                        include_tools=include_tools
                    )
                    
                    # Display budget report
                    st.subheader("Budget Allocation")
                    display_budget_bars(assembled.allocation)
                    
                    # Total usage
                    total = assembled.allocation.total
                    budget_total = get_budget_manager().config.total
                    st.metric(
                        "Total Token Usage",
                        f"{total} / {budget_total}",
                        delta=f"{total - budget_total}" if total > budget_total else None,
                        delta_color="inverse"
                    )
                    
                    # Truncation warning
                    if assembled.truncation_applied:
                        st.warning("[WARNING] Budget exceeded - truncation was applied")
                        with st.expander("View Truncation Details"):
                            for section, detail in assembled.truncation_details.items():
                                st.text(f"• {section}: {detail}")
                    else:
                        st.success("[SUCCESS] All sections within budget")
                    
                    # Generate response
                    st.divider()
                    st.subheader("Response")
                    
                    client = get_gemini_client()
                    response = client.generate_with_context(
                        assembled.context_parts,
                        temperature=0.7
                    )
                    
                    st.markdown(response)
                    
                    # Add to memory
                    memory = get_conversation_memory()
                    memory.add_exchange(user_query, response)
                    
                    # Store in session
                    st.session_state.messages.append({
                        'query': user_query,
                        'response': response,
                        'allocation': assembled.allocation
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# Tab 2: Demo Overflow
with tab2:
    st.header("Budget Overflow Demonstration")
    st.markdown("This demonstrates how the system handles context that exceeds budgets.")
    
    if st.button("[DEMO] Run Overflow Demo", type="primary", use_container_width=True):
        with st.spinner("Creating overflow scenario..."):
            # Create overflow scenario
            memory = get_conversation_memory()
            memory.clear()
            
            st.info("Step 1: Adding extensive conversation history...")
            for i in range(10):
                memory.add_exchange(
                    f"Question {i+1} about machine learning, neural networks, and AI concepts with detailed technical terminology",
                    f"Detailed answer {i+1} explaining various technical aspects of deep learning systems, including architecture patterns, optimization algorithms, and best practices"
                )
            st.success(f"[DONE] Added {len(memory.get_recent_exchanges())} exchanges")
            
            st.info("Step 2: Adding tool execution outputs...")
            tools = get_tool_manager()
            tools.clear()
            
            for i in range(5):
                tools.add_tool_output(
                    "web_search",
                    f"Search result {i+1}: " + "Detailed technical information about neural networks and deep learning " * 50,
                    success=True
                )
            st.success(f"[DONE] Added {len(tools.get_recent_outputs())} tool outputs")
            
            st.info("Step 3: Creating long query...")
            long_query = """
            I'm working on understanding neural networks in depth. Can you explain:
            1. How backpropagation works with the chain rule
            2. Different optimization algorithms like SGD, Adam, RMSprop
            3. Regularization techniques including dropout and batch normalization
            4. Architecture patterns for CNNs and RNNs
            5. Best practices for hyperparameter tuning
            """ * 3
            st.success(f"[DONE] Query created (~{len(long_query)} characters)")
            
            st.info("Step 4: Assembling context (will trigger truncation)...")
            
            try:
                assembler = get_context_assembler()
                assembled = assembler.assemble(long_query)
                
                st.divider()
                st.subheader("Results")
                
                # Budget visualization
                display_budget_bars(assembled.allocation)
                
                # Total
                total = assembled.allocation.total
                budget_total = get_budget_manager().config.total
                st.metric(
                    "Total Token Usage",
                    f"{total} / {budget_total}",
                    delta=f"{total - budget_total}" if total > budget_total else None,
                    delta_color="inverse"
                )
                
                # Truncation details
                if assembled.truncation_applied:
                    st.warning("[WARNING] TRUNCATION APPLIED")
                    st.subheader("Truncation Details:")
                    for section, detail in assembled.truncation_details.items():
                        st.info(f"**{section}**: {detail}")
                else:
                    st.success("[SUCCESS] All sections within budget (unexpected)")
                
                st.success("[SUCCESS] Demo complete! System handled overflow gracefully.")
                
            except Exception as e:
                st.error(f"Error during demo: {str(e)}")

# Tab 3: View Context
with tab3:
    st.header("View Assembled Context")
    st.markdown("Inspect the raw context sections that would be sent to the LLM.")
    
    # Simple query input
    view_query = st.text_input(
        "Enter a query to see its assembled context:",
        placeholder="e.g., What is machine learning?"
    )
    
    if st.button("[VIEW] Assemble & View", use_container_width=True):
        if not view_query.strip():
            st.warning("Please enter a query")
        else:
            try:
                assembler = get_context_assembler()
                assembled = assembler.assemble(view_query)
                
                # Budget summary
                st.subheader("Budget Summary")
                display_budget_bars(assembled.allocation)
                
                st.divider()
                
                # Display each section
                st.subheader("Context Sections")
                
                sections = [
                    ("Instructions", assembled.context_parts.get('instructions', '')),
                    ("Goal", assembled.context_parts.get('goal', '')),
                    ("Memory", assembled.context_parts.get('memory', '')),
                    ("Retrieval", assembled.context_parts.get('retrieval', '')),
                    ("Tool Outputs", assembled.context_parts.get('tool_outputs', ''))
                ]
                
                for title, content in sections:
                    display_context_section(title, content)
                
                # Full context
                st.divider()
                with st.expander("View Full Assembled Context"):
                    full_context = assembled.get_full_context()
                    st.text_area("Complete Context:", full_context, height=400)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.caption("Context-Window-Aware RAG System | Demonstrating explicit budget management and prioritization")

