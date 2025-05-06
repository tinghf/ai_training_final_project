# Ollama RAG System Architecture

```ascii
+----------------------------------------------------------------------------------------+
|                                    Local Client                                        |
|                                                                                        |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|  |  RAG Client    |     |  Document      |     |  Vector DB     |     |  Configuration | |
|  |  (Python)      |     |  Loader        |     |  (ChromaDB)    |     |  Files         | |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|         |                      |                        |                      |         |
|         v                      v                        v                      v         |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|  |  Local         |     |  Document      |     |  Document      |     |  Environment   | |
|  |  Documents     |     |  Processing    |     |  Indexing      |     |  Variables     | |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
+----------------------------------------------------------------------------------------+
                                    |
                                    |
                                    v
+----------------------------------------------------------------------------------------+
|                                    Remote Servers                                      |
|                                                                                        |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|  |  Ollama        |     |  Ollama        |     |  Ollama        |     |  Ollama        | |
|  |  Server 1      |     |  Server 2      |     |  Server 3      |     |  Server N      | |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|         |                      |                        |                      |         |
|         v                      v                        v                      v         |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
|  |  LLM Model 1   |     |  LLM Model 2   |     |  LLM Model 3   |     |  LLM Model N   | |
|  |  (e.g.,        |     |  (e.g.,        |     |  (e.g.,        |     |  (e.g.,        | |
|  |  llama2)       |     |  mistral)      |     |  codellama)    |     |  custom)       | |
|  +----------------+     +----------------+     +----------------+     +----------------+ |
+----------------------------------------------------------------------------------------+

Data Flow:
1. Local Documents → Document Loader → Vector DB (Indexing)
2. Query → RAG Client → Vector DB (Retrieval)
3. Retrieved Context + Query → RAG Client → Selected Ollama Server
4. Ollama Server → Response → RAG Client → User

Key Components:
- Local Client: Handles document processing, RAG operations, and server communication
- Vector DB: Stores document embeddings and enables semantic search
- Document Loader: Processes and chunks documents for indexing
- Ollama Servers: Host different LLM models for specialized tasks
- Configuration: Manages server URLs, model selection, and system settings

Security Layers:
- Nginx Reverse Proxy (not shown) for secure server access
- Document encryption in Vector DB
- Access control for document retrieval
- Rate limiting on server endpoints
``` 