# AI Loan Advisory Chatbot

An AI-powered Loan Advisory Chatbot built using a modular Retrieval-Augmented Generation (RAG) architecture. The chatbot answers loan-related questions using only official banking documents and provides source-backed responses to minimize hallucinations.

The system combines document processing, semantic search, intent classification, vector retrieval, and Large Language Models (LLMs) to deliver accurate and reliable financial information.

---

# 📌 Project Objective

The objective of this project is to develop an intelligent AI assistant that can:

- Answer loan-related queries in natural language.
- Retrieve relevant information from official banking documents.
- Generate accurate, context-aware responses using Retrieval-Augmented Generation (RAG).
- Reduce hallucinations by grounding responses in retrieved document chunks.
- Display source information for every response.
- Demonstrate a complete end-to-end RAG pipeline for financial question answering.

---

# 🚀 Key Features

- AI-powered Loan Advisory Chatbot
- Retrieval-Augmented Generation (RAG)
- GRU-based Intent Classification
- Semantic Document Retrieval using FAISS
- Sentence Transformer Embeddings
- Source-backed Responses
- Hallucination Reduction
- Intelligent Document Chunking
- Modular Notebook-based Architecture
- Support for Multiple Banks

---

# 🏦 Supported Banks

## State Bank of India (SBI)

- Home Loan
- Personal Loan
- Education Loan
- Gold Loan
- Vehicle Loan
- Loan Against Property

## HDFC Bank

- Home Loan
- Personal Loan
- Car Loan

---

# 🏗️ Project Architecture

```text
Official Banking PDFs
        │
        ▼
Notebook 1
PDF Processing Pipeline
        │
        ▼
Notebook 2
Intelligent Document Chunking
        │
        ▼
Notebook 3
Sentence Embeddings + FAISS
        │
        ▼
Notebook 4
GRU Intent Classifier
        │
        ▼
Notebook 5
Retrieval-Augmented Generation Pipeline
        │
        ▼
Streamlit Chatbot
        │
        ▼
Source-backed Response
```

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Development Environment

- Google Colab

## Deep Learning

- PyTorch
- GRU Neural Network

## NLP

- Sentence Transformers
- Transformers

## Vector Database

- FAISS

## Document Processing

- PyMuPDF
- pdfplumber

## Data Processing

- Pandas
- NumPy

## User Interface

- Streamlit

---

# 📁 Project Structure

```text
AI_Loan_Advisory_Chatbot/

│
├── data/
│   ├── SBI/
│   └── HDFC/
│
├── processed/
│   ├── cleaned_text/
│   ├── metadata/
│   ├── chunks/
│   └── embeddings/
│
├── vector_db/
│   └── faiss_index.bin
│
├── models/
│   ├── gru_intent_model.pth
│   ├── vocabulary.pkl
│   ├── label_encoder.pkl
│   └── tokenizer.pkl
│
├── notebooks/
│   ├── 01_PDF_Processing.ipynb
│   ├── 02_Document_Chunking.ipynb
│   ├── 03_Embeddings_FAISS.ipynb
│   ├── 04_GRU_Intent_Classifier.ipynb
│   ├── 05_RAG_Pipeline.ipynb
│   └── 06_Streamlit_App.ipynb
│
└── README.md
```

---

# ⚙️ Workflow

1. Collect official banking documents.
2. Extract text from PDFs.
3. Clean and preprocess document content.
4. Generate intelligent document chunks.
5. Create sentence embeddings.
6. Store embeddings in a FAISS vector database.
7. Predict the user's intent using a GRU classifier.
8. Retrieve the most relevant document chunks.
9. Generate grounded responses using retrieved context.
10. Display source attribution with every response.

---

# 📚 Dataset

The chatbot uses official banking documents collected from SBI and HDFC, including:

- Loan Policy Documents
- Interest Rate Documents
- Eligibility Guidelines
- Loan FAQs
- Terms and Conditions
- Processing Fee Documents
- Repayment Information
- Official Bank Circulars

---

# 🎯 Project Highlights

- Modular notebook-based implementation
- End-to-end Retrieval-Augmented Generation pipeline
- Semantic search using Sentence Transformers
- FAISS vector indexing
- Intent-aware retrieval
- Source-grounded responses
- Designed to minimize hallucinations
- Easily extensible to additional banks and loan products

---

# 🔮 Future Enhancements

- Multi-bank support
- Voice-based interaction
- Multi-language support
- Hybrid Retrieval (BM25 + Dense Retrieval)
- Cross-Encoder Re-ranking
- Personalized Loan Recommendations
- OCR support for scanned documents
- Deployment using Docker and cloud platforms

---

# 📄 License

This project has been developed for educational, internship, and research purposes.
