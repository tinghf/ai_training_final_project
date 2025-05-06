#!/bin/bash

# List of models to pull and serve
MODELS=("llama2" "mistral" "codellama")

# Pull each model
for model in "${MODELS[@]}"; do
    echo "Pulling model: $model"
    ollama pull $model
done

# Start the Ollama server
echo "Starting Ollama server..."
ollama serve 