# 🧠 DocuMind AI - Smart Document Assistant
<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge"/></a>
  <a href="https://groq.com/"><img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge"/></a>
  <a href="https://faiss.ai/"><img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge"/></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/></a>
</p>

---

## 🌐 Live Demo & Deployment

- 🚀 **Try it here:** [DocuMind AI - Smart Document Assistant](https://docmind-ai-kp.streamlit.app)  
- 📽️ **Deployment Video:** [Watch](https://drive.google.com/file/d/1OTtTHaZRkavjen6Cm4BN_j4PXN7yrKL4/view?usp=sharing)  

<div align="center">
  <img src="assets/1.png" alt="DocuMind AI Interface" width="80%"/>
  <p><em>Main Interface</em></p>
</div>

---

## 📖 Project Overview

DocuMind AI is an **intelligent document analysis platform** that transforms your PDFs into **interactive, conversational experiences**.  
Upload documents and ask natural language questions to extract insights, analyze content, and explore knowledge in a human-like way.

---

## ✨ Key Features

### 🧠 AI-Powered Intelligence
- Smart Document Analysis with advanced AI algorithms  
- Contextual Understanding of relationships & themes  
- Actionable Insights and patterns  
- Natural Language Conversations with your files  

### 🎨 Modern Interface
- Clean & Professional Design  
- Intuitive Navigation  
- Responsive for desktop/mobile  
- Real-time Interaction & Feedback  

### 🚀 Advanced Capabilities
- Multi-Document Analysis  
- FAISS-Powered Semantic Search  
- Conversational Memory  
- Lightning-Fast AI with **Groq**  
- Source Attribution for transparency  
- Customizable AI Parameters  

---

## 🔧 Architecture

1. **Document Ingestion** → PDF parsing & text extraction  
2. **Text Chunking** → Recursive splitting for continuity  
3. **Vector Embeddings** → HuggingFace sentence-transformers  
4. **Vector Storage** → FAISS for efficient retrieval  
5. **Query Processing** → LangChain conversational chain  
6. **Response Generation** → Groq-powered contextual answers  

---

## ⚙️ Technology Stack

- **Frontend** → Streamlit  
- **Backend** → LangChain  
- **Vector DB** → FAISS  
- **Embeddings** → HuggingFace Transformers  
- **LLMs** → Groq API (Llama3, Mixtral, etc.)  
- **Language** → Python  

---

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh

# Or run directly
python setup.py
````

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv documind_env

# 2. Activate environment
# Windows:
documind_env\Scripts\activate
# Linux/Mac:
source documind_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API key
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run app
streamlit run app.py
```
---

## 📌 Use Cases

* **Research** → Summarize & analyze papers
* **Legal** → Review contracts/policies
* **Technical** → API docs, manuals
* **Education** → Interactive textbooks
* **Business Intelligence** → Extract insights from reports

---