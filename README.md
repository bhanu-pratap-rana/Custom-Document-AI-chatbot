# Document Processing and AI Chatbot System

## Overview
This project implements a comprehensive document processing system with OCR capabilities, vector search, and an AI-powered chatbot interface.
The dataset is text image based in which you want to perform OCR, I used my personal dataset.

## Features
- Document OCR and text extraction
- Named Entity Recognition
- Text summarization using BART
- Vector search using Milvus
- SQL database integration
- Interactive chatbot interface

## Setup Instructions

### Prerequisites
- Python 3.8+
- MySQL Server
- Milvus Server
- Tesseract OCR

### Installation
1. Clone the repository
```bash
git clone <repository-url>

Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

Database Setup
mysql -u root -p
CREATE DATABASE documents_db;

Start Milvus Server
docker-compose up -d

Running the System
Process Documents
python document_processor.py

Generate Synthetic Dataset (Optional)
python ticket_generator.py

Launch Chatbot Interface
python main.py

Project Structure
project/
├── dataset/
src/              # Document images
├── document_processor.py # Main processing pipeline
├── dynamic_summarizer.py # Text summarization
├── bot.py               # Chatbot implementation
├── vector_search.py     # Vector search functionality
├── ticket_generator.py  # Synthetic data generation
└── main.py             # Application entry point

Usage
Place documents in the dataset folder
Run document processing
Access the chatbot interface at http://localhost:8000
