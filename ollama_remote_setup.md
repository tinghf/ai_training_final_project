# Setting up Ollama on Remote Server and Local Connection Guide

## Prerequisites
- A remote server (Linux-based) with root/sudo access
- Python 3.7+ installed on both remote and local machines
- Basic knowledge of SSH and terminal commands

## Step 1: Install Ollama on Remote Server

1. SSH into your remote server:
```bash
ssh username@your-server-ip
```

2. Install Ollama using the official installation script:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

3. Start the Ollama service:
```bash
sudo systemctl start ollama
```

4. Enable Ollama to start on boot:
```bash
sudo systemctl enable ollama
```

5. Verify the installation:
```bash
ollama --version
```

## Step 2: Pull and Run a Model on Remote Server

1. Pull a model (e.g., llama2):
```bash
ollama pull llama2
```

2. Start the model server:
```bash
ollama serve
```

3. In a new terminal, test the model:
```bash
ollama run llama2 "Hello, how are you?"
```

## Step 3: Configure Remote Server Security

1. Install and configure a reverse proxy (e.g., Nginx):
```bash
sudo apt update
sudo apt install nginx
```

2. Create an Nginx configuration file:
```bash
sudo nano /etc/nginx/sites-available/ollama
```

3. Add the following configuration:
```nginx
server {
    listen 80;
    server_name your-server-ip;

    location / {
        proxy_pass http://localhost:11434;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. Enable the configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Step 4: Set Up Local Python Environment

1. Create a new Python virtual environment:
```bash
python -m venv ollama-env
source ollama-env/bin/activate  # On Windows: ollama-env\Scripts\activate
```

2. Install required packages:
```bash
pip install requests langchain chromadb sentence-transformers python-dotenv
```

## Step 5: Set Up Local Vector Database and RAG

1. Create a directory for your documents and vector database:
```bash
mkdir -p documents vector_db
```

2. Create a document loader script (`document_loader.py`):
```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

def load_and_process_documents(documents_dir="documents", vector_db_dir="vector_db"):
    # Load documents from directory
    loader = DirectoryLoader(
        documents_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vector_db_dir
    )
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    load_and_process_documents()
```

3. Create a RAG client script (`rag_client.py`):
```python
import requests
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Dict

class RAGClient:
    def __init__(self, server_url="http://your-server-ip", vector_db_dir="vector_db"):
        self.server_url = server_url
        self.model = "llama2"
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=self.embeddings)

    def retrieve_relevant_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents from the vector database."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def generate_response(self, prompt: str, use_rag: bool = True) -> str:
        """Generate response using RAG if enabled."""
        if use_rag:
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(prompt)
            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            
            # Create RAG-enhanced prompt
            enhanced_prompt = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {prompt}

Answer:"""
        else:
            enhanced_prompt = prompt

        # Send request to Ollama
        endpoint = f"{self.server_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": enhanced_prompt,
            "stream": False
        }

        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama server: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the RAG client
    client = RAGClient()
    
    # Example with RAG
    prompt = "What are the key points about machine learning?"
    response = client.generate_response(prompt, use_rag=True)
    print(f"RAG Response: {response}")
    
    # Example without RAG
    response = client.generate_response(prompt, use_rag=False)
    print(f"Direct Response: {response}")
```

## Step 6: Using the RAG System

1. Prepare your documents:
   - Place your text documents in the `documents` directory
   - Documents should be in plain text format (.txt)

2. Load documents into the vector database:
```bash
python document_loader.py
```

3. Test the RAG system:
```bash
python rag_client.py
```

## Step 7: Test the Connection

1. Make sure your remote server's Ollama service is running
2. Run your local Python script:
```bash
python rag_client.py
```

## Troubleshooting

1. If you get connection errors:
   - Verify the server IP address in your Python script
   - Check if the Ollama service is running on the remote server
   - Ensure your firewall allows traffic on port 80 (or your configured port)

2. If you get permission errors:
   - Check if the Ollama service has proper permissions
   - Verify Nginx configuration and permissions

3. If the model doesn't respond:
   - Check if the model is properly pulled on the remote server
   - Verify the model name in your Python script matches the one on the server

4. If RAG responses are not relevant:
   - Check if documents are properly loaded into the vector database
   - Adjust the chunk size and overlap in the document loader
   - Try different embedding models
   - Verify the document format and content

## Security Considerations

1. Consider adding authentication to your Nginx configuration
2. Use HTTPS instead of HTTP for production environments
3. Implement rate limiting to prevent abuse
4. Regularly update both Ollama and your server's security patches
5. Consider encrypting sensitive documents in the vector database
6. Implement access control for document retrieval

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Python Requests Library](https://requests.readthedocs.io/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/) 