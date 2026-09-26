# Talk to PDF using RAG

A simple Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents.
The application extracts text from a PDF, converts it into embeddings, stores them in a vector database, retrieves 
relevant information based on user queries, and generates answers using Google's Gemini model.

## Features

* Upload and read PDF documents
* Extract text from PDFs
* Split text into chunks for efficient retrieval
* Generate embeddings using Hugging Face models
* Store embeddings in Chroma Vector Database
* Retrieve relevant document chunks using semantic search
* Generate answers using Google Gemini
* Interactive command-line chatbot experience

## Tech Stack

* Python
* LangChain
* Google Gemini
* Hugging Face Embeddings
* ChromaDB
* PyPDF
* Python Dotenv

## Project Workflow

```text
PDF Document
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
Retriever
      ↓
Gemini LLM
      ↓
Answer Generation
```

## Project Structure

```text
Talk_to_PDF/
│
├── pdfs/
│   └── sample.pdf
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/talk-to-pdf-rag.git
cd talk-to-pdf-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

## Running the Application

Place your PDF file inside the `pdfs` folder.

Example:

```text
pdfs/sample.pdf
```

Run:

```bash
python app.py
```

Enter your query when prompted:

```text
Enter your query: Which medicine is present at Desk M3?
```

Example Output:

```text
Model response:
Volini Gel is present at Desk M3.
```

## Example Queries

* What medicines are available in Desk A1?
* Is Dolo 650 present in the inventory?
* Which medicine is stored at Desk M3?
* What is the price of Paracetamol?
* List all medicines present in the document.
* Which medicine has the highest stock?

## Future Improvements

* PDF Upload UI
* Streamlit Web Interface
* Multi-PDF Support
* Persistent Chroma Database
* Chat History
* Source Citations
* Support for Excel and CSV Files
* Pharmacy Inventory Integration

## Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embeddings
* Semantic Search
* Prompt Engineering
* LangChain Pipelines
* Gemini Integration

## Author : Vivek Kumar Shaw

