#!/bin/bash
set -e

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done

# List of models to pull
models=("mistral:7b" "deepseek-r1:7b")

# Pull each model
for model in "${models[@]}"; do
    echo "Pulling $model..."
    ollama pull $model
done

# Keep container running
tail -f /dev/null
