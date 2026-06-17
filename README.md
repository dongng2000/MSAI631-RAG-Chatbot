---

title: MSAI631 RAG Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
-------------

# MSAI631 RAG Chatbot

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot developed for MSAI 631.

The chatbot uses:

* Sentence Transformers for semantic search
* Microsoft Phi-4-mini-instruct (Azure AI Foundry)
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
Microsoft Phi-4-mini-instruct (Azure AI Foundry)
↓
Response

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```
