```mermaid

graph LR
    subgraph Local Client Components
        A[Chat Client ]
        B[Document Processor]
        C[ChromaDB ]
    end

    subgraph Remote Servers
        D[Ollama Server 1]
        E[Ollama Server 2]
        F[Ollama Server N]
        D1[llama2]
        E1[mistral]
        F1[codellama]
    end

    subgraph Data Flow
        L[Local Knoledge Base] --> B
        B -- Load and Index --> C
        Q[Query] --> A
        A -- Query --> C
        C -- Retrieved Context --> A
        A --> D
        A --> E
        A --> F
        D-- Response -->A
        E-- Response -->A
        F-- Response -->A
        A --> U[User]
    end


    D --> D1
    E --> E1
    F --> F1
    
```

<!-- 
    subgraph Key Components
        KC1[Local Client: Handles document processing, RAG operations, and server communication]
        KC2[Vector DB: Stores document embeddings and enables semantic search]
        KC3[Document Loader: Processes and chunks documents for indexing]
        KC4[Ollama Servers: Host different LLM models for specialized tasks]
        KC5[Configuration: Manages server URLs, model selection, and system settings]
    end

    subgraph Security Layers
        SL1[Nginx Reverse Proxy - not shown - for secure server access]
        SL2[Document encryption in Vector DB]
        SL3[Access control for document retrieval]
        SL4[Rate limiting on server endpoints]
    end
 -->

