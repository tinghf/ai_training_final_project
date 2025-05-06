import requests
import json
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv

class OllamaClient:
    def __init__(self, server_url: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Get server URL from environment or use default
        self.server_url = server_url or os.getenv('OLLAMA_SERVER_URL', 'http://127.0.0.1:11434')
        self.default_model = os.getenv('OLLAMA_MODEL', 'llama2')
        # self.available_models: List[str] = ["llama2", "mistral", "llama3.2"]
        self.available_models: List[str] = []
        
        # Test connection and get available models
        self.test_connection()

    def test_connection(self):
        """Test the connection to the Ollama server and get available models."""
        try:
            response = requests.get(f"{self.server_url}/api/tags")
            response.raise_for_status()
            self.available_models = [model['name'] for model in response.json()['models']]
            print(f"Successfully connected to Ollama server at {self.server_url}")
            print(f"Available models: {self.available_models}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama server: {e}")
            print("\nTroubleshooting steps:")
            print("1. Make sure Ollama is installed and running")
            print("2. Check if the server URL is correct (default: http://127.0.0.1:11434)")
            print("3. Verify the port number (default: 11434)")
            print("4. Check if the models are downloaded (run: ollama pull <model_name>)")
            print("\nTo fix:")
            print("1. Start Ollama server: ollama serve")
            print("2. Pull the models: ollama pull llama2 && ollama pull mistral")
            print("3. Set environment variables:")
            print("   export OLLAMA_SERVER_URL=http://your-server:11434")
            print("   export OLLAMA_MODEL=llama2")
            raise

    def generate_response(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate response from the Ollama server using specified model."""
        print(f"[DEBUG] OllamaClient.generate_response for model: {model}")
        if model and model not in self.available_models:
            print(f"Model {model} not available. Using default model {self.default_model}")
            model = self.default_model
        elif not model:
            model = self.default_model

        endpoint = f"{self.server_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return None

    def get_model_info(self, model: str) -> Dict:
        """Get information about a specific model."""
        try:
            response = requests.get(f"{self.server_url}/api/show", json={"name": model})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting model info: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the client
        client = OllamaClient()
        
        # Example prompts with different models
        prompt = "What is the capital of France?"
        
        # Try with default model
        print("\nUsing default model:")
        response = client.generate_response(prompt)
        if response:
            print(f"Response: {response}")
        
        # Try with specific model if available
        if "mistral" in client.available_models:
            print("\nUsing Mistral model:")
            response = client.generate_response(prompt, model="mistral")
            if response:
                print(f"Response: {response}")
        
    except Exception as e:
        print(f"Initialization error: {e}") 