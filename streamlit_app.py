import streamlit as st
import time
from ollama_client import OllamaClient
from typing import List, Dict, Callable
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from langchain.document_loaders import JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
from pathlib import Path
import tempfile
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class OllamaConnection:
    name: str
    url: str
    model: str
    client: Optional[OllamaClient] = None

def load_ollama_connections(yaml_path: str) -> List[OllamaConnection]:
    """Load Ollama server connections from YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        connections = []
        for conn in config.get('connections', []):
            try:
                client = OllamaClient(server_url=conn['url'])
                connection = OllamaConnection(
                    name=conn['name'],
                    url=conn['url'],
                    model=conn['model'],
                    client=client
                )
                connections.append(connection)
            except Exception as e:
                st.error(f"Failed to initialize connection to {conn['name']}: {str(e)}")
        
        return connections
    except Exception as e:
        st.error(f"Error loading Ollama connections: {str(e)}")
        return []

def initialize_session_state():
    """Initialize all session state variables."""
    if 'connections' not in st.session_state:
        try:
            st.session_state.connections = load_ollama_connections('ollama-connections.yaml')
        except Exception as e:
            st.error(f"Failed to load Ollama connections: {str(e)}")
            st.session_state.connections = []
    
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    
    if 'loading' not in st.session_state:
        st.session_state.loading = {}
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

# Initialize session state
initialize_session_state()

# Create a queue for thread-safe communication
response_queue = queue.Queue()

def metadata_func(record: dict, metadata: dict = None) -> dict:
    """Extract content and metadata properly from JSON records."""
    meta = record.get("metadata", {})
    return {
        "article_id": meta.get("article_id", ""),
        "title": meta.get("title", ""),
        "authors": meta.get("authors", ""),
        "classification": meta.get("classification", "")
    }

def load_research_articles(json_path: str):
    """Load research articles from JSON file."""
    try:
        loader = JSONLoader(
            file_path=json_path,
            jq_schema='.[]',
            content_key='text',
            metadata_func=lambda x: {
                'article_id': x.get('article_id', ''),
                'authors': x.get('authors', []),
                'title': x.get('title', ''),
                'year': x.get('year', ''),
                'journal': x.get('journal', ''),
                'source_file': os.path.basename(json_path)  # Add source file to metadata
            }
        )
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading research articles from {json_path}: {str(e)}")
        return None

def create_vectorstore(documents):
    """Create and persist vectorstore from documents."""
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./final_project_research_chroma_db"
        )
        
        # Persist the vectorstore
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files and combine their documents."""
    all_documents = []
    
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())
            
            try:
                # Load documents using JSONLoader with the new configuration
                loader = JSONLoader(
                    file_path=temp_path,
                    jq_schema=".",  # REQUIRED even if not using JQ
                    content_key="page_content",
                    metadata_func=metadata_func
                )
                documents = loader.load()
                
                if documents:
                    # Add source file to metadata for each document
                    for doc in documents:
                        doc.metadata["source_file"] = uploaded_file.name
                    all_documents.extend(documents)
                    st.sidebar.success(f"Successfully processed {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up the temporary file
                os.unlink(temp_path)
    
    return all_documents

def retrieve_relevant_docs(query: str, vectorstore, k: int = 3):
    """Retrieve relevant documents for the query."""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def format_context(docs):
    """Format retrieved documents into context."""
    context = ""
    for doc in docs:
        context += f"Source: {doc.metadata['source_file']}\n"
        context += f"Title: {doc.metadata['title']}\n"
        context += f"Authors: {doc.metadata['authors']}\n"
        context += f"Classification: {doc.metadata['classification']}\n"
        context += f"Content: {doc.page_content}\n\n"
    return context

def get_model_response(connection: OllamaConnection, prompt: str, context: str = "") -> str:
    """Get response from a specific model."""
    if not connection.client:
        return f"Error: No client available for {connection.name}"
    
    print(f"[DEBUG] Starting request for {connection.name}")
    try:
        # Augment prompt with context if available
        if context:
            augmented_prompt = f"""Context information:
{context}

Based on the above context, please answer the following question:
{prompt}

Please provide a detailed and accurate response based on the context provided."""
        else:
            augmented_prompt = prompt

        print(f"[DEBUG] Generating response for model:{connection.model} in connection: {connection.name}")
        response = connection.client.generate_response(model=connection.model, prompt=augmented_prompt)
        print(f"[DEBUG] Received response from {connection.name}: {response[:100]}...")
        return response
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[DEBUG] Error from {connection.name}: {error_msg}")
        return error_msg

def process_model_response(connection: OllamaConnection, prompt: str, context: str = ""):
    """Process model response in a separate thread and put result in queue."""
    print(f"[DEBUG] Processing response for {connection.name}")
    response = get_model_response(connection, prompt, context)
    print(f"[DEBUG] Queueing response for {connection.name}")
    response_queue.put((connection.name, response))

def update_responses_concurrent(prompt: str, selected_connections: List[OllamaConnection], context: str = ""):
    """Update responses for all models concurrently."""
    if not selected_connections:
        st.error("No Ollama connections available. Please check your configuration.")
        return

    print(f"[DEBUG] Starting concurrent updates for connections: {[c.name for c in selected_connections]}")
    
    # Reset loading states
    for connection in selected_connections:
        st.session_state.loading[connection.name] = True
        st.session_state.responses[connection.name] = "Loading..."
        print(f"[DEBUG] Set loading state for {connection.name}")

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=len(selected_connections)) as executor:
        # Submit all tasks
        print("[DEBUG] Submitting tasks to thread pool")
        futures = [executor.submit(process_model_response, connection, prompt, context) 
                  for connection in selected_connections]
        
        # Process results as they come in
        completed = 0
        print("[DEBUG] Starting to process results")
        while completed < len(selected_connections):
            try:
                name, response = response_queue.get(timeout=3)
                print(f"[DEBUG] Processing result for {name}")
                print(f"[DEBUG] Current state before update - Loading: {st.session_state.loading.get(name)}, Response: {st.session_state.responses.get(name)}")
                
                # Update state
                st.session_state.responses[name] = response
                st.session_state.loading[name] = False
                
                print(f"[DEBUG] State after update - Loading: {st.session_state.loading.get(name)}, Response: {st.session_state.responses.get(name)[:100]}...")
                completed += 1
                print(f"[DEBUG] Completed {completed}/{len(selected_connections)} connections")
                
                # Only rerun to force update UI after all models have completed
                if completed == len(selected_connections):
                    print("[DEBUG] All models completed, triggering rerun")
                    st.rerun()
            except queue.Empty:
                print("[DEBUG] Queue empty, waiting for more results...")
                continue

# Custom CSS for dark mode
st.markdown("""
<style>
    /* Dark mode theme */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Model frames */
    .model-frame {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #3D3D3D;
    }
    
    .model-header {
        color: #00B4D8;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #00B4D8;
    }
    
    .model-response {
        background-color: #363636;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        white-space: pre-wrap;
        color: #E0E0E0;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 180, 216, 0.3);
        border-radius: 50%;
        border-top-color: #00B4D8;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Input styling */
    .stTextArea textarea {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #3D3D3D;
        font-size: 16px;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #00B4D8;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    
    .stButton button:hover {
        background-color: #0096B7;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    
    /* Title and text colors */
    h1, h2, h3, p {
        color: #FFFFFF;
    }
    
    /* Footer styling */
    .footer {
        color: #888888;
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Biomedical RAG System")
st.write("Query research articles using multiple LLM models")

# Check if connections are available
if not st.session_state.connections:
    st.error("""
    No Ollama connections available. Please check:
    1. ollama-connections.yaml file exists and is properly configured
    2. All specified Ollama servers are running
    3. Server URLs are correct
    """)
    st.stop()

# Sidebar for model selection and document upload
st.sidebar.title("Configuration")

# Model selection
available_connections = st.session_state.connections
selected_connections = st.sidebar.multiselect(
    "Select models to compare",
    options=available_connections,
    format_func=lambda x: f"{x.name} ({x.model})",
    default=available_connections[:2] if len(available_connections) >= 2 else available_connections
)

# Document upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Research Articles (JSON)",
    type=['json'],
    accept_multiple_files=True
)

# Process uploaded files if there are new ones
if uploaded_files and uploaded_files != st.session_state.uploaded_files:
    st.session_state.uploaded_files = uploaded_files
    
    # Process all uploaded files
    all_documents = process_uploaded_files(uploaded_files)
    
    if all_documents:
        # Create or update vectorstore
        st.session_state.vectorstore = create_vectorstore(all_documents)
        st.sidebar.success(f"Successfully processed {len(uploaded_files)} files with {len(all_documents)} articles!")

# Display currently loaded files
if st.session_state.uploaded_files:
    st.sidebar.write("Currently loaded files:")
    for file in st.session_state.uploaded_files:
        st.sidebar.write(f"- {file.name}")

# Main content area with prompt and button in the same row
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.text_area("Enter your question:", height=100)
with col2:
    st.write("")  # Add some vertical spacing
    st.write("")  # Add some vertical spacing
    generate_button = st.button("Generate Responses", use_container_width=True)

# Create a container for all model responses
responses_container = st.container()

if generate_button:
    if not prompt:
        st.warning("Please enter a question")
    elif not selected_connections:
        st.warning("Please select at least one model")
    elif not st.session_state.vectorstore:
        st.warning("Please upload research articles first")
    else:
        print(f"[DEBUG] Starting response generation for connections: {[c.name for c in selected_connections]}")
        
        # Get relevant context if vectorstore is available
        context = ""
        if st.session_state.vectorstore:
            docs = retrieve_relevant_docs(prompt, st.session_state.vectorstore)
            context = format_context(docs)
            st.sidebar.write("Retrieved Context:", context)
        
        # Clear previous responses
        st.session_state.responses = {}
        st.session_state.processing = True
        
        # Update responses concurrently
        update_responses_concurrent(prompt, selected_connections, context)
        
        # Reset processing state
        st.session_state.processing = False
        print("[DEBUG] Completed all response generation")

# Display responses in separate frames
with responses_container:
    for connection in selected_connections:
        loading_indicator = '<div class="loading"></div>' if st.session_state.loading.get(connection.name, False) else ''
        response = st.session_state.responses.get(connection.name, "Waiting for response...")
        print(f"[DEBUG] Displaying response for {connection.name}: {response[:100]}...")
        st.markdown(f"""
        <div class="model-frame">
            <div class="model-header">
                {loading_indicator}{connection.name} ({connection.model})
            </div>
            <div class="model-response">
                {response}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Powered by Ollama, LangChain, and Streamlit</p>
</div>
""", unsafe_allow_html=True) 