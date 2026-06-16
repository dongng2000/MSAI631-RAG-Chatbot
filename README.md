# MSAI631 RAG Chatbot

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot developed for MSAI 631.

The chatbot uses:

* Sentence Transformers for semantic search
* TinyLlama for response generation
* Gradio for the user interface
* Retrieval-Augmented Generation (RAG) architecture

## Features

* Semantic search
* Context-aware responses
* Suggested questions
* Fallback handling
* User guidance
* Conversational interface

## Architecture

User
↓
Gradio Interface
↓
Sentence Transformer Embeddings
↓
Semantic Search
↓
Retrieve Context
↓
TinyLlama
↓
Response

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:7860
