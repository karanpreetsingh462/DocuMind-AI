# 🧠 DocuMind AI - Smart Document Assistant
````markdown
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=facebook&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/AI-FF6B6B?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Document_Analysis-4ECDC4?style=for-the-badge&logo=file-text&logoColor=white"/>
</p>
````
## Project Overview

DocuMind AI is an intelligent document analysis platform that transforms your PDFs into interactive, conversational experiences. Built with cutting-edge AI technology, it enables users to upload documents and engage in natural language conversations to extract insights, analyze content, and answer complex questions about their materials.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# For Windows users
setup.bat

# For Unix/Linux/macOS users
./setup.sh

# Or run the Python setup script directly
python setup.py

```
### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv documind_env

# 2. Activate virtual environment
# Windows:
documind_env\Scripts\activate
# Unix/Linux/macOS:
source documind_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file and add your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 5. Run the application
streamlit run app.py
```

## 🌐 Live Demo

**Experience DocuMind AI:** [Smart Document Assistant](https://documind-ai.streamlit.app/)

* 📽️ [Deployment Video](https://drive.google.com/file/d/1OTtTHaZRkavjen6Cm4BN_j4PXN7yrKL4/view?usp=sharing)
* 🖼️ Screenshots:
  <div align="center">
    <img src="assets/1.png" alt="DocuMind AI Interface" width="80%"/>
    <p><em>Main Interface</em></p>
  </div>

## ✨ Key Features

### 🧠 AI-Powered Intelligence

* **Smart Document Analysis**: Advanced AI algorithms understand and interpret your content
* **Contextual Understanding**: Deep comprehension of document relationships and themes
* **Intelligent Insights**: Extract meaningful patterns and actionable information
* **Natural Language Processing**: Seamless conversation with your documents

### 🎨 Beautiful Modern Interface

* **Stunning Visual Design**: Clean and professional layout
* **Intuitive User Experience**: Easy navigation for all users
* **Responsive Layout**: Optimized for desktop and mobile
* **Smart Visual Feedback**: Real-time interaction and responses

### 🚀 Advanced Capabilities

* **Multi-Document Analysis**: Compare multiple documents simultaneously
* **Intelligent Retrieval**: FAISS-powered semantic search
* **Conversational Memory**: Maintains context across queries
* **Real-time Processing**: Lightning-fast AI responses with Groq
* **Source Attribution**: Transparent references to document locations
* **Customizable Intelligence**: Tune AI behavior and parameters
* **Multiple AI Models**: Choose between speed, precision, or depth

## 🔧 Architecture

The system implements a multi-stage RAG pipeline:

1. **Document Ingestion**: PDF parsing and text extraction
2. **Text Chunking**: Recursive character-based splitting
3. **Vector Embeddings**: HuggingFace sentence-transformers
4. **Vector Storage**: FAISS for efficient retrieval
5. **Query Processing**: LangChain conversational chain with memory
6. **Response Generation**: Groq-powered models for contextual answers

## ⚙️ Technical Implementation

### Core Components

* **Frontend**: Streamlit
* **Backend**: LangChain framework for orchestration
* **Vector Database**: FAISS
* **Language Models**: Groq API with multiple model options
* **Embeddings**: HuggingFace sentence-transformers

### Model Configuration

* **Primary Models**: Llama3-8B, Mixtral-8x7B, Llama3-70B
* **Embedding Model**: all-MiniLM-L6-v2
* **Vector Dimensions**: 384
* **Retrieval Strategy**: Similarity-based top-k

## ⚙️ Technology Stack

* **Groq API**: High-performance inference
* **Llama3 & Mixtral**: Advanced AI models
* **LangChain**: LLM orchestration framework
* **HuggingFace Transformers**: Pre-trained ecosystem
* **FAISS**: Vector similarity search
* **Streamlit + Python**: Web application and backend

## 🔩 Configuration Options

* **Temperature**: Controls response creativity
* **Top-k Retrieval**: Ensures context relevance
* **Chunk Size & Overlap**: Optimized for document continuity

## 📌 Use Cases

* **Research**: Paper analysis and summarization
* **Legal**: Contract and policy review
* **Technical**: API and manual querying
* **Education**: Interactive learning with textbooks
* **Business Intelligence**: Data-driven insights from reports

```
```