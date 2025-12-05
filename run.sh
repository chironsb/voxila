#!/bin/bash
# Script to run Speech-to-RAG v2

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from env.example..."
    cp env.example .env
    echo "Please edit .env with your configuration."
fi

# Run the application
cd src
python main.py

