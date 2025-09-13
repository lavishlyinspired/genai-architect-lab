#!/usr/bin/env python3
"""
Complete LangSmith Tracing Setup Guide
=====================================

This guide shows different ways to enable LangSmith tracing for your LangChain applications.
"""

import os
import uuid
from typing import Optional

# Core imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled, collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

# =============================================================================
# METHOD 1: GLOBAL ENVIRONMENT VARIABLES (Simplest approach)
# =============================================================================

def setup_environment_tracing():
    """
    Set up LangSmith tracing using environment variables.
    This is the easiest method for most applications.
    """
    print("🔧 Setting up LangSmith tracing with environment variables...")
    import os

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ["LANGSMITH_PROJECT"] = "pr-gripping-carrier-50"

    # Optional: Set OpenAI API key if using OpenAI
    os.environ["OPENAI_API_KEY"] = ""
    
    # Optional: Background callbacks (default: true)
    # Set to false for serverless environments to ensure traces complete
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"
    
    print("✅ Environment variables set. All LangChain operacstions will now be traced!")


def basic_traced_example():
    """
    Basic example with environment-based tracing.
    No additional code needed - tracing happens automatically!
    """
    print("\n🚀 Running basic traced example...")
    
    # Create a simple chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    output_parser = StrOutputParser()
    
    chain = prompt | model | output_parser
    
    # This will be automatically traced if environment variables are set
    result = chain.invoke({
        "question": "What is machine learning?",
        "context": "Machine learning is a subset of AI that enables computers to learn."
    })
    
    print(f"Result: {result}")
    print("🎯 Check your LangSmith project dashboard to see the trace!")


# =============================================================================
# METHOD 2: SELECTIVE TRACING WITH CALLBACKS
# =============================================================================

def selective_tracing_with_callbacks():
    """
    Trace only specific invocations using callback handlers.
    Useful when you don't want to trace everything globally.
    """
    print("\n🎯 Demonstrating selective tracing with callbacks...")
    
    # Create chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    
    # Method 2a: Using LangChainTracer directly
    tracer = LangChainTracer(project_name="selective-tracing-demo")
    
    result1 = chain.invoke(
        {"input": "Explain quantum computing"},
        config={"callbacks": [tracer]}
    )
    print(f"Traced result: {result1[:100]}...")
    
    # Method 2b: Using context manager (Python only)
    with tracing_v2_enabled(project_name="context-manager-demo"):
        result2 = chain.invoke({"input": "What is blockchain?"})
        print(f"Context traced result: {result2[:100]}...")
    
    # This won't be traced (assuming no global env vars)
    result3 = chain.invoke({"input": "Tell me about AI"})
    print("Untraced result: This won't appear in LangSmith")


# =============================================================================
# METHOD 3: PROGRAMMATIC CLIENT CONFIGURATION
# =============================================================================

def programmatic_tracing_setup():
    """
    Set up tracing programmatically without environment variables.
    Useful for environments where env vars can't be set.
    """
    print("\n💻 Setting up programmatic tracing...")
    
    # Create LangSmith client programmatically
    client = Client(
        api_key="your-langsmith-api-key",  # Get from secrets manager
        api_url="https://api.smith.langchain.com"  # Default endpoint
    )
    
    # Create tracer with custom client
    tracer = LangChainTracer(
        client=client,
        project_name="programmatic-tracing"
    )
    
    # Create chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data scientist."),
        ("user", "{query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    
    # Trace with custom client
    result = chain.invoke(
        {"query": "Explain the bias-variance tradeoff"},
        config={"callbacks": [tracer]}
    )
    
    print("✅ Programmatic tracing complete!")


# =============================================================================
# METHOD 4: ADVANCED TRACING FEATURES
# =============================================================================

def advanced_tracing_features():
    """
    Demonstrate advanced tracing features like metadata, tags, and custom names.
    """
    print("\n🔬 Demonstrating advanced tracing features...")
    
    # Create chain with custom configuration
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}."),
        ("user", "{question}")
    ])
    
    # Add tags and metadata to the model
    model = ChatOpenAI(model="gpt-4o-mini").with_config({
        "tags": ["gpt-4o-mini", "production"],
        "metadata": {
            "model_version": "2024-09",
            "environment": "production",
            "user_id": "user_123"
        }
    })
    
    output_parser = StrOutputParser()
    
    # Configure entire chain with additional metadata
    chain = (prompt | model | output_parser).with_config({
        "tags": ["qa-chain", "expert-domain"],
        "metadata": {"chain_type": "domain_expert_qa"}
    })
    
    # Invoke with runtime configuration
    result = chain.invoke(
        {
            "domain": "machine learning",
            "question": "What's the difference between supervised and unsupervised learning?"
        },
        config={
            "run_name": "ML_Expert_QA",  # Custom run name
            "tags": ["runtime-tag"],     # Additional runtime tags
            "metadata": {"session_id": "session_456"}  # Runtime metadata
        }
    )
    
    print(f"Advanced traced result: {result[:150]}...")
    print("🏷️  Check LangSmith - this trace has custom tags, metadata, and naming!")


def get_run_id_example():
    """
    Example of how to capture and use run IDs for linking traces.
    """
    print("\n🆔 Demonstrating run ID capture...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    
    # Method 1: Using collect_runs context manager
    with collect_runs() as cb:
        result = chain.invoke({"input": "What is the capital of France?"})
        run_id = cb.traced_runs[0].id
        print(f"🎯 Captured run ID: {run_id}")
    
    # Method 2: Setting custom run ID
    custom_run_id = str(uuid.uuid4())
    result = chain.invoke(
        {"input": "What is the capital of Spain?"},
        config={"run_id": custom_run_id}
    )
    print(f"🔧 Used custom run ID: {custom_run_id}")


# =============================================================================
# METHOD 5: SERVERLESS OPTIMIZATIONS
# =============================================================================

def serverless_tracing_setup():
    """
    Special considerations for serverless environments.
    """
    print("\n☁️ Serverless tracing setup...")
    
    # For serverless: disable background callbacks to ensure completion
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
    
    try:
        # Your LangChain code here
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a serverless assistant."),
            ("user", "{input}")
        ])
        model = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | model
        
        result = chain.invoke({"input": "Explain serverless computing"})
        print(f"Serverless result: {result[:100]}...")
        
    finally:
        # CRITICAL: Wait for all traces to be submitted before function ends
        print("⏳ Waiting for traces to complete...")
        wait_for_all_tracers()
        print("✅ All traces submitted!")


# =============================================================================
# COMPLETE EXAMPLE: ENHANCED RETRIEVAL SYSTEM WITH TRACING
# =============================================================================

class TracedRetrievalSystem:
    """
    Enhanced version of the original retrieval system with comprehensive tracing.
    """
    
    def __init__(self, project_name: str = "rag-system"):
        self.project_name = project_name
        
        # Set up tracing
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project_name
        
        # Initialize components with tracing metadata
        self.embeddings = OpenAIEmbeddings().with_config({
            "tags": ["embeddings", "openai"],
            "metadata": {"component": "embeddings"}
        })
        
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_config({
            "tags": ["llm", "reranking"],
            "metadata": {"component": "reranker", "temperature": 0}
        })
        
        print(f"✅ TracedRetrievalSystem initialized with project: {project_name}")
    
    def search_and_rerank_traced(self, query: str, documents: list):
        """
        Perform search and reranking with detailed tracing.
        """
        with tracing_v2_enabled(project_name=self.project_name):
            # This entire operation will be traced as a single run tree
            
            # Step 1: Document retrieval (would use FAISS in real implementation)
            print(f"🔍 Retrieving documents for: {query}")
            retrieved_docs = documents[:5]  # Simulate retrieval
            
            # Step 2: Reranking with traced LLM call
            rerank_prompt = ChatPromptTemplate.from_template("""
            Rank these documents by relevance to the query: {query}
            
            Documents:
            {documents}
            
            Return only numbers separated by commas (e.g., "3,1,2"):
            """).with_config({
                "tags": ["rerank-prompt"],
                "metadata": {"prompt_type": "reranking"}
            })
            
            chain = (rerank_prompt | self.llm).with_config({
                "run_name": f"Rerank_Query_{query[:30]}",
                "tags": ["reranking-chain"],
                "metadata": {
                    "query": query,
                    "num_documents": len(retrieved_docs),
                    "operation": "rerank"
                }
            })
            
            formatted_docs = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)])
            
            response = chain.invoke({
                "query": query,
                "documents": formatted_docs
            })
            
            print(f"🎯 Reranking complete. Check LangSmith project '{self.project_name}' for trace details!")
            
            return response


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Run all tracing examples.
    """
    print("🔬 LangSmith Tracing Setup Guide")
    print("=" * 50)
    
    # Note: In practice, you'd set these with real API keys
    print("⚠️  Before running:")
    print("1. Get your LangSmith API key from https://smith.langchain.com/")
    print("2. Replace 'your-langsmith-api-key-here' with your actual key")
    print("3. Set your OpenAI API key if using OpenAI models")
    print("4. Update the API keys in the code above")
    print()
    
    # Example 1: Environment-based tracing
    print("📋 Example 1: Environment-based tracing")
    setup_environment_tracing()
    basic_traced_example()  # Uncomment when API keys are set
    
    # Example 2: Selective tracing
    print("\n📋 Example 2: Selective tracing")
    selective_tracing_with_callbacks()  # Uncomment when API keys are set
    
    # Example 3: Advanced features
    print("\n📋 Example 3: Advanced tracing features")
    advanced_tracing_features()  # Uncomment when API keys are set
    
    # Example 4: Run ID capture
    print("\n📋 Example 4: Run ID capture")
    get_run_id_example()  # Uncomment when API keys are set
    
    # Example 5: Enhanced retrieval system
    print("\n📋 Example 5: Enhanced retrieval system with tracing")
    sample_docs = [
        "LangChain is a framework for developing applications powered by language models.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "RAG combines retrieval and generation for more accurate LLM responses.",
        "Vector databases store and search high-dimensional vectors efficiently.",
        "LangSmith provides observability and evaluation tools for LLM applications."
    ]
    
    system = TracedRetrievalSystem("enhanced-rag-demo")
    system.search_and_rerank_traced("What is RAG?", sample_docs)
    
    print("\n🎉 Tracing guide complete!")
    print("📊 Visit https://smith.langchain.com/ to view your traces")


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK REFERENCE
# =============================================================================

"""
QUICK SETUP CHECKLIST:
======================

1. Install required packages:
   pip install langchain langchain-openai langsmith

2. Set environment variables:
   export LANGSMITH_TRACING=true
   export LANGSMITH_API_KEY=your-api-key
   export LANGSMITH_PROJECT=your-project-name
   export OPENAI_API_KEY=your-openai-key  # if using OpenAI

3. Run your LangChain code - tracing happens automatically!

COMMON ENVIRONMENT VARIABLES:
============================
- LANGSMITH_TRACING=true              # Enable tracing
- LANGSMITH_API_KEY=ls_xxx            # Your LangSmith API key  
- LANGSMITH_PROJECT=my-project        # Project name (optional)
- LANGSMITH_ENDPOINT=https://...      # Custom endpoint (optional)
- LANGCHAIN_CALLBACKS_BACKGROUND=true # Background processing (default: true)

SERVERLESS CONSIDERATIONS:
=========================
- Set LANGCHAIN_CALLBACKS_BACKGROUND=false
- Always call wait_for_all_tracers() before function exit
- Consider using context managers for critical sections

USEFUL LANGSMITH FEATURES:
=========================
- Automatic trace visualization
- Performance monitoring  
- Error tracking and debugging
- A/B testing and evaluation
- Cost tracking across models
- Custom metadata and tagging
"""