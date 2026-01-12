# Context-Window-Aware RAG System

> A production-grade Retrieval-Augmented Generation system with explicit context window management and intelligent budget enforcement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Architecture](#architecture)
- [Context Structure & Budgets](#context-structure--budgets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Prioritization & Truncation Rules](#prioritization--truncation-rules)
- [Memory vs Retrieval](#memory-vs-retrieval)
- [Worked Example](#worked-example)
- [Design Decisions](#design-decisions)
- [Evaluation Criteria Coverage](#evaluation-criteria-coverage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project implements a RAG system that **explicitly manages its context window** using strict token budgets. Unlike traditional RAG systems that indiscriminately pack context, this system demonstrates intelligent prioritization and graceful degradation when budgets are exceeded.

### The Problem

Most RAG systems suffer from:
- **Indiscriminate context packing** - throwing everything at the LLM
- **No budget awareness** - hoping it fits within limits
- **Poor prioritization** - treating all context equally
- **Silent failures** - truncation happens invisibly

### Our Solution

A RAG system that:
- ✅ **Explicitly manages** every token in the context window
- ✅ **Enforces hard budgets** for each section
- ✅ **Applies intelligent truncation** when limits are exceeded
- ✅ **Reports all decisions** with full transparency

---

## Key Innovation

### Explicit Context Economics

We treat tokens as a **scarce resource** with deliberate allocation:
```
Total Budget: 3,215 tokens

Instructions   [ 255 tokens ]  ████████░░░░░░░░░░░░  31.8%
Goal          [1500 tokens ]  ████████████████████  60.2%
Memory        [  55 tokens ]  ██░░░░░░░░░░░░░░░░░░  54.5%
Retrieval     [ 550 tokens ]  ████████████░░░░░░░░  55.7%
Tool Outputs  [ 855 tokens ]  ████████████████░░░░  65.5%
```

Each section has:
1. **Fixed budget** - hard token limit
2. **Source definition** - where content comes from
3. **Selection logic** - how content is chosen
4. **Fallback behavior** - what happens when exceeded

---

## Architecture

### High-Level Components
```
┌─────────────┐
│    User     │
└──────┬──────┘
       │ Query
       ▼
┌─────────────────────────────────────────┐
│         Context Assembler               │
│  ┌────────────────────────────────────┐ │
│  │ 1. Gather from all sources         │ │
│  │ 2. Validate against budgets        │ │
│  │ 3. Apply truncation if needed      │ │
│  │ 4. Assemble final context          │ │
│  └────────────────────────────────────┘ │
└────┬──────────────────────────────┬─────┘
     │                              │
     │  ┌────────────────────────┐  │
     ├──┤  Instructions (255)    │  │
     │  └────────────────────────┘  │
     │  ┌────────────────────────┐  │
     ├──┤  Goal (1500)           │  │
     │  └────────────────────────┘  │
     │  ┌────────────────────────┐  │
     ├──┤  Memory (55)           │◄─┤── Conversation History
     │  └────────────────────────┘  │
     │  ┌────────────────────────┐  │
     ├──┤  Retrieval (550)       │◄─┤── Vector Search
     │  └────────────────────────┘  │   (ChromaDB + Gemini Embeddings)
     │  ┌────────────────────────┐  │
     └──┤  Tool Outputs (855)    │◄─┘── Recent Actions
        └────────────────────────┘
                  │
                  ▼
            ┌──────────┐
            │   LLM    │ (Google Gemini)
            └──────────┘
                  │
                  ▼
            [ Response ]
```

### Core Modules

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `core/` | Context assembly orchestration | `ContextAssembler` |
| `core/` | Token budget enforcement | `BudgetManager` |
| `core/` | Truncation strategies | `Prioritizer` |
| `retrieval/` | Vector search | `VectorStore`, `Retriever` |
| `retrieval/` | Text embeddings | `EmbeddingGenerator` |
| `memory/` | Conversation state | `ConversationMemory` |
| `memory/` | User preferences | `UserPreferences` |
| `tools/` | Tool execution tracking | `ToolManager` |
| `llm/` | LLM integration | `GeminiClient` |
| `utils/` | Token counting | `TokenCounter` |

---

## Context Structure & Budgets

### 1. Instructions (255 tokens)

**Source:** Static configuration file  
**Selection Logic:** Always included verbatim  
**Fallback:** ERROR - should never exceed (configuration issue)  
**Priority:** HIGHEST
```python
# Example
"You are a helpful AI assistant with access to relevant knowledge.
Your role is to provide accurate, informative responses based on:
1. The retrieved knowledge provided
2. Your general understanding
3. The conversation context

Always cite sources when using retrieved information.
Be concise but thorough in your explanations."
```

**Why this budget?**
- Core system behavior must fit comfortably
- 255 tokens = ~1000 characters = adequate for guidelines
- If exceeded, it's a configuration error, not a runtime issue

---

### 2. Goal (1,500 tokens)

**Source:** User's current query + user preferences  
**Selection Logic:** Direct from user input  
**Fallback:** Keep first 40% + last 40%, remove middle  
**Priority:** HIGH
```python
# Truncation example
Original: "I'm working on X with Y and Z using A, B, C... what is best for Z?"
Truncated: "I'm working on X... what is best for Z?"

# Preserves:
✅ Context setup (what user is doing)
✅ Actual question (what user wants)

# Sacrifices:
❌ Implementation details (can be inferred)
```

**Why this budget?**
- Users can write long, detailed queries
- 1,500 tokens = ~6,000 characters = several paragraphs
- Preserving intent > preserving every detail

**Strategy Rationale:**
- Beginning: Sets context ("I'm working on...")
- End: Contains actual question ("What is best way to...")
- Middle: Often contains verbose details

---

### 3. Memory (55 tokens)

**Source:** Recent conversation history  
**Selection Logic:** FIFO queue, keep most recent exchanges  
**Fallback:** Drop oldest exchanges first  
**Priority:** LOW
```python
# Example
Exchange 1: User: "What is ML?" 
            Assistant: "ML is..."
Exchange 2: User: "How does it work?" 
            Assistant: "It uses algorithms..."

# When budget exceeded: Drop Exchange 1, keep Exchange 2
```

**Why this budget?**
- 55 tokens = ~1-2 recent exchanges
- Very recent context is most relevant
- Older context becomes less valuable

**Design Tradeoff:**
- ✅ Keeps conversation coherent
- ✅ Lightweight (doesn't dominate budget)
- ❌ Can't maintain long conversation history
- **Solution:** Most important info should be in retrieval or goal

---

### 4. Retrieval (550 tokens)

**Source:** Vector similarity search over corpus  
**Selection Logic:** Top K documents by cosine similarity (threshold > 0.3)  
**Fallback:** Keep highest scoring chunks, drop lowest  
**Priority:** MEDIUM
```python
# Example
Query: "How do neural networks learn?"

Retrieved chunks (sorted by relevance):
1. doc2.txt (score: 0.89) - "Neural networks learn through backpropagation..." [200 tokens]
2. doc1.txt (score: 0.76) - "Machine learning uses algorithms..." [180 tokens]
3. doc4.txt (score: 0.68) - "Training involves..." [150 tokens]
4. doc5.txt (score: 0.42) - "Vector databases store..." [120 tokens]

If budget = 550 tokens:
✅ Keep chunks 1, 2, 3 (530 tokens total)
❌ Drop chunk 4 (lowest relevance)
```

**Why this budget?**
- 550 tokens = ~2-3 document excerpts
- Enough for meaningful context
- Not so large that it dominates the window

**Similarity Threshold (0.3):**
- Cosine similarity ranges 0-1 (1 = identical)
- We convert distance to similarity: `score = 1 / (1 + distance)`
- Threshold 0.3 = moderate relevance minimum
- Below 0.3 = likely not helpful

---

### 5. Tool Outputs (855 tokens)

**Source:** Recent tool execution results  
**Selection Logic:** Most recent, prioritize successful over failed  
**Fallback:** Keep recent successful, drop old/failed  
**Priority:** MEDIUM-LOW
```python
# Example
Tool outputs (sorted by success, then timestamp):
1. web_search (SUCCESS, t=500) - "Found 3 results..." [200 tokens]
2. calculator (SUCCESS, t=400) - "Result: 42" [50 tokens]
3. web_search (SUCCESS, t=300) - "Retrieved..." [250 tokens]
4. database (FAILED, t=200) - "Connection error" [100 tokens]
5. web_search (SUCCESS, t=100) - "Old results..." [300 tokens]

If budget = 855 tokens:
✅ Keep outputs 1, 2, 3 (500 tokens total)
❌ Drop outputs 4, 5 (failed + old)
```

**Why this budget?**
- 855 tokens = largest allocation (recognizes tool importance)
- Tool outputs often contain valuable data
- Recent successful > old or failed

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key (free tier available)

### Setup Steps
```bash
# 1. Clone or download the project
cd context-window-aware-RAG

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Initialize vector store
python -m cli.main init
```

---

## Quick Start

### Basic Usage
```bash
# Ask a question
python -m cli.main query "How do neural networks learn?"

# Show budget breakdown
python -m cli.main query "What is RAG?" --show-budget

# Show assembled context
python -m cli.main query "Explain backpropagation" --show-context

# Demonstrate budget overflow
python -m cli.main demo-overflow
```

### Example Session
```bash
# Initialize system
python -m cli.main init

# Ask related questions (builds conversation memory)
python -m cli.main query "What is machine learning?"
python -m cli.main query "How does it relate to neural networks?"
python -m cli.main query "Can you summarize what we discussed?"

# View statistics
python -m cli.main stats

# Clear memory when done
python -m cli.main clear
```

---

## Prioritization & Truncation Rules

### When Budget is Exceeded

The system follows a strict priority order:
```
Priority Order (lowest to highest):
1. Memory         - Truncate FIRST  (least critical)
2. Tool Outputs   - Truncate SECOND
3. Retrieval      - Truncate THIRD
4. Goal           - Truncate FOURTH
5. Instructions   - NEVER truncate (most critical)
```

### Truncation Strategies by Section

#### 1. Instructions: NEVER TRUNCATE
```
If instructions exceed 255 tokens:
  → Log ERROR
  → This is a configuration issue
  → Instructions are included as-is
  → System behavior may be unpredictable
```

#### 2. Goal: Start + End Preservation
```python
def truncate_goal(goal: str, budget: int) -> str:
    """
    Keep first 40% and last 40%, remove middle 20%
    
    Rationale:
    - Start: Context setup ("I'm working on...")
    - End: Actual question ("What is best way to...")
    - Middle: Often verbose details
    """
    start_portion = goal[:int(len(goal) * 0.4)]
    end_portion = goal[int(len(goal) * 0.6):]
    
    return f"{start_portion} [...] {end_portion}"
```

**Example:**
```
Input (150 tokens):
"I'm building a neural network for image classification using PyTorch 
and TensorFlow. I've tried various architectures including ResNet, 
VGG, and custom CNNs. My dataset has 10,000 images across 5 classes. 
I'm getting 65% accuracy but want 90%+. I've tuned learning rates, 
batch sizes, and tried different optimizers. What is the best approach 
to improve accuracy?"

Output (90 tokens):
"I'm building a neural network for image classification using PyTorch 
and TensorFlow. I've tried various architectures [...] What is the 
best approach to improve accuracy?"

✅ Preserved: What user is doing + what they want
❌ Lost: Specific implementation details (recoverable from context)
```

#### 3. Memory: FIFO Queue
```python
def truncate_memory(memory_items: List[str], budget: int) -> str:
    """
    Keep most recent exchanges that fit within budget
    
    Rationale:
    - Recent context > old context
    - Conversation naturally evolves
    - Older exchanges become less relevant
    """
    selected = []
    total_tokens = 0
    
    # Start from newest, work backwards
    for item in reversed(memory_items):
        if total_tokens + count_tokens(item) <= budget:
            selected.insert(0, item)
            total_tokens += count_tokens(item)
        else:
            break
    
    return "\n".join(selected)
```

**Example:**

