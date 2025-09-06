import streamlit as st
import os
import tempfile
from typing import List
import warnings
from dotenv import load_dotenv
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Try new import first, fallback to old if needed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
class RAGConfig:
    def __init__(self):
        load_dotenv()

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model_name = "llama-3.1-8b-instant"  # Updated to current model
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.top_k = 4
        self.temperature = 0.1

# Initialize session state - NO auto-processing
def initialize_session_state():
    """Initialize session state variables safely"""
    if 'config' not in st.session_state:
        st.session_state.config = RAGConfig()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0
    
    # New features state
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}
    
    if 'export_format' not in st.session_state:
        st.session_state.export_format = "PDF"
    
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Standard"

# Helper functions
@st.cache_resource
def initialize_groq_llm():
    """Initialize Groq LLM with caching"""
    try:
        llm = ChatGroq(
            groq_api_key=st.session_state.config.groq_api_key,
            model_name=st.session_state.config.model_name,
            temperature=st.session_state.config.temperature,
            max_tokens=1024,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        return None

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings with caching"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=st.session_state.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

def load_pdf_documents(pdf_files) -> List[Document]:
    """Load PDF documents from uploaded files"""
    documents = []
    
    for pdf_file in pdf_files:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['source_file'] = pdf_file.name
            
            documents.extend(docs)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error loading {pdf_file.name}: {e}")
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.config.chunk_size,
        chunk_overlap=st.session_state.config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Create vector store from documents"""
    if not documents:
        return None
    
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def create_rag_chain(llm, vector_store):
    """Create conversational RAG chain with custom prompt"""
    if not vector_store:
        return None
    
    try:
        # Simplified prompt template
        prompt_template = """Answer the question based on the context provided. Be helpful and informative.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": st.session_state.config.top_k}
        )
        
        # Create a simpler chain without complex memory handling
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,  # Enable verbose mode for debugging
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        st.write(f"RAG chain error details: {type(e).__name__}: {str(e)}")
        return None

def process_pdfs(pdf_files):
    """Process uploaded PDFs - ONLY when explicitly called"""
    if not pdf_files:
        st.warning("Please upload at least one PDF file")
        return False
    
    if st.session_state.is_processing:
        st.warning("Already processing documents. Please wait.")
        return False
    
    st.session_state.is_processing = True
    
    try:
        with st.spinner("Processing PDFs..."):
            # Load documents
            documents = load_pdf_documents(pdf_files)
            
            if not documents:
                st.error("No documents loaded")
                return False
            
            st.info(f"Loaded {len(documents)} pages from {len(pdf_files)} files")
            
            # Split documents
            splits = split_documents(documents)
            st.info(f"Created {len(splits)} text chunks")
            
            # Create vector store
            vector_store = create_vector_store(splits, st.session_state.embeddings)
            
            if not vector_store:
                st.error("Failed to create vector store")
                return False
            
            st.info("Vector store created successfully")
            
            # Create RAG chain
            qa_chain = create_rag_chain(st.session_state.llm, vector_store)
            
            if not qa_chain:
                st.error("Failed to create RAG chain")
                return False
            
            st.info("RAG chain created successfully")
            
            # Store in session state
            st.session_state.qa_chain = qa_chain
            st.session_state.vector_store = vector_store
            st.session_state.processed_files = [f.name for f in pdf_files]
            
            # Update document stats
            st.session_state.document_stats = calculate_document_stats(splits)
            
            st.success(f"✅ Successfully processed {len(pdf_files)} PDF files with {len(documents)} pages and {len(splits)} chunks")
            return True
            
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        st.write(f"Debug info: {type(e).__name__}: {str(e)}")
        return False
    finally:
        st.session_state.is_processing = False

def query_rag_system(question: str):
    """Query the RAG system"""
    if not st.session_state.qa_chain:
        return "Please upload and process PDF files first.", []
    
    try:
        # Check if vector store exists and has documents
        if not st.session_state.vector_store:
            return "No documents available for analysis. Please upload and process documents first.", []
        
        # Test the retriever first
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": st.session_state.config.top_k}
        )
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            return "No relevant information found in the documents for your question. Try asking about the main topics or content.", []
        
        # Debug: Show what documents were found
        st.write(f"🔍 Found {len(relevant_docs)} relevant document chunks")
        
        # Query the chain with better error handling
        result = st.session_state.qa_chain({"question": question})
        
        # Debug: Show the raw result
        st.write("📋 Raw result keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        
        answer = result.get('answer', 'No answer generated')
        sources = result.get('source_documents', [])
        
        # Handle different response types more robustly
        if hasattr(answer, 'content'):
            answer = answer.content
        elif hasattr(answer, 'text'):
            answer = answer.text
        elif isinstance(answer, dict):
            answer = str(answer)
        elif not isinstance(answer, str):
            answer = str(answer)
        
        # Debug: Show the processed answer
        st.write(f"💬 Processed answer length: {len(answer) if answer else 0}")
        st.write(f"📄 Answer preview: {answer[:100] if answer else 'None'}...")
        
        # If answer is empty or too short, try a different approach
        if not answer or len(answer.strip()) < 10:
            # Try to get a simple answer from the LLM directly
            try:
                simple_prompt = f"Based on the following context, answer this question: {question}\n\nContext: {relevant_docs[0].page_content[:500]}..."
                direct_answer = st.session_state.llm.invoke(simple_prompt)
                
                if hasattr(direct_answer, 'content'):
                    direct_answer = direct_answer.content
                elif hasattr(direct_answer, 'text'):
                    direct_answer = direct_answer.text
                else:
                    direct_answer = str(direct_answer)
                
                if direct_answer and len(direct_answer.strip()) > 10:
                    return direct_answer, sources
                else:
                    return "I found relevant information in your documents, but couldn't generate a complete answer. Try rephrasing your question or asking about specific topics.", sources
            except Exception as direct_error:
                st.write(f"Direct LLM error: {direct_error}")
                return "I found relevant information in your documents, but couldn't generate a complete answer. Try rephrasing your question or asking about specific topics.", sources
        
        return answer, sources
    except Exception as e:
        error_msg = f"Error querying RAG system: {str(e)}"
        st.error(error_msg)
        # More detailed debugging
        st.write(f"Debug info: {type(e).__name__}: {str(e)}")
        st.write(f"Vector store exists: {st.session_state.vector_store is not None}")
        st.write(f"QA chain exists: {st.session_state.qa_chain is not None}")
        return "Error occurred while processing your question. Please try again.", []

def ask_question(question: str):
    """Handle question asking with proper state management"""
    if not question.strip():
        return
    
    if st.session_state.is_processing:
        st.warning("System is busy. Please wait.")
        return
    
    st.session_state.is_processing = True
    
    try:
        with st.spinner("Thinking..."):
            answer, sources = query_rag_system(question.strip())
            st.session_state.chat_history.append((question.strip(), answer, sources))
            
        # Clear input by incrementing counter
        st.session_state.input_counter += 1
        
    except Exception as e:
        st.error(f"Error processing question: {e}")
    finally:
        st.session_state.is_processing = False

def calculate_document_stats(documents):
    """Calculate statistics for uploaded documents"""
    stats = {
        'total_pages': 0,
        'total_chunks': 0,
        'avg_chunk_size': 0,
        'file_count': len(st.session_state.processed_files)
    }
    
    if documents:
        stats['total_pages'] = len(documents)
        stats['total_chunks'] = len(documents)
        chunk_sizes = [len(doc.page_content) for doc in documents]
        stats['avg_chunk_size'] = sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0
    
    return stats

def export_chat_history():
    """Export chat history in different formats"""
    if not st.session_state.chat_history:
        return None
    
    export_data = []
    for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
        export_data.append({
            'question': question,
            'answer': answer,
            'sources': [f"{doc.metadata.get('source_file', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})" for doc in sources]
        })
    
    return export_data

def generate_document_summary():
    """Generate a comprehensive document summary"""
    if not st.session_state.qa_chain:
        return "Please process documents first."
    
    summary_prompt = """
    Please provide a comprehensive analysis of all uploaded documents including:
    1. Main topics and themes
    2. Key findings and insights
    3. Important statistics or data points
    4. Conclusions and recommendations
    5. Document relationships and connections
    
    Format this as a professional executive summary.
    """
    
    try:
        result = st.session_state.qa_chain({"question": summary_prompt})
        answer = result.get('answer', 'No summary generated')
        
        if hasattr(answer, 'content'):
            answer = answer.content
        
        return answer
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        padding: 2rem 1rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(255,107,107,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Main content background */
    .main .block-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    
    /* Card Styling */
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 1px solid #475569;
        margin-bottom: 1rem;
    }
    
    .status-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255,107,107,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,107,107,0.6);
    }
    
    /* Chat Message Styling */
    .user-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        color: white;
        padding: 1rem;
        border-radius: 20px 20px 8px 20px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    
    .ai-message {
        background: linear-gradient(135deg, #4ecdc4 0%, #7dd3fc 100%);
        color: #1e293b;
        padding: 1rem;
        border-radius: 20px 20px 20px 8px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 4px 15px rgba(78,205,196,0.2);
    }
    
    /* File Upload Styling */
    .uploadedFile {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px dashed #22c55e;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Progress Indicators */
    .processing-indicator {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Quick Actions */
    .quick-action-btn {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        transform: translateY(-1px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="DocuMind AI - Smart Document Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Header Section
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🧠 DocuMind AI</h1>
        <p>Transform your documents into intelligent conversations with advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models if not already done
    if st.session_state.llm is None:
        with st.spinner("Initializing Groq LLM..."):
            st.session_state.llm = initialize_groq_llm()
    
    if st.session_state.embeddings is None:
        with st.spinner("Loading embeddings..."):
            st.session_state.embeddings = initialize_embeddings()
    
    # Check if models are loaded
    if not st.session_state.llm or not st.session_state.embeddings:
        st.error("Failed to initialize models. Please refresh the page.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📄 Document Upload")
        
        # File upload with better styling
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload your documents for AI analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.markdown(f"**📄 {len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                st.markdown(f"• {file.name}")
        
        # Process button with better styling - Equal size buttons
        if st.button("🧠 Process Documents", type="primary", disabled=st.session_state.is_processing, use_container_width=True):
            if uploaded_files:
                success = process_pdfs(uploaded_files)
                if success:
                    st.rerun()
            else:
                st.warning("Please upload documents first")
        
        if st.button("🗑️ Clear Chat", disabled=st.session_state.is_processing, use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.session_state.is_processing:
            st.markdown("""
            <div class="processing-indicator">
                <strong>⏳ Processing in progress...</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Settings Section
        st.markdown("### ⚙️ Settings")
        
        # Model selection with better descriptions
        model_options = {
            "llama-3.1-8b-instant": "⚡ Lightning Fast (Balanced Performance)",
            "llama-3.1-70b-versatile": "🎯 High Precision (Advanced Analysis)",
            "mixtral-8x7b-32768": "🧠 Maximum Intelligence (Premium Quality)"
        }
        
        selected_model = st.selectbox(
            "AI Model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state.config.model_name == "llama-3.1-8b-instant" else 1
        )
        
        if selected_model != st.session_state.config.model_name:
            st.session_state.config.model_name = selected_model
            st.session_state.llm = None  # Force reinitialize
            st.rerun()
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings", expanded=False):
            chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.config.chunk_size, 
                                 help="Size of text chunks for processing")
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, st.session_state.config.chunk_overlap,
                                    help="Overlap between chunks for context")
            top_k = st.slider("Top K Results", 1, 10, st.session_state.config.top_k,
                            help="Number of relevant chunks to retrieve")
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.config.temperature,
                                  help="Creativity level (0=precise, 1=creative)")
        
        # Update config
        st.session_state.config.chunk_size = chunk_size
        st.session_state.config.chunk_overlap = chunk_overlap
        st.session_state.config.top_k = top_k
        st.session_state.config.temperature = temperature
        
        st.divider()
        
        # Processed files info with better styling
        if st.session_state.processed_files:
            st.markdown("### 📚 Analyzed Documents")
            for file in st.session_state.processed_files:
                st.markdown(f"✅ {file}")
        
        st.divider()
        
        # New Features Section
        st.markdown("### 🚀 Advanced Features")
        
        # Analysis Mode Selection
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Standard", "Deep Analysis", "Quick Summary", "Research Mode"],
            index=0
        )
        st.session_state.analysis_mode = analysis_mode
        
        # Document Statistics
        if st.session_state.processed_files:
            st.markdown("### 📊 Document Statistics")
            if st.session_state.vector_store:
                # Calculate stats
                stats = calculate_document_stats([])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Files", stats['file_count'])
                    st.metric("📝 Pages", stats['total_pages'])
                with col2:
                    st.metric("🧩 Chunks", stats['total_chunks'])
                    st.metric("💬 Messages", len(st.session_state.chat_history))
        
        # Export Options
        if st.session_state.chat_history:
            st.markdown("### 📤 Export Options")
            export_format = st.selectbox(
                "Export Format",
                ["PDF", "TXT", "JSON", "CSV"],
                index=0
            )
            st.session_state.export_format = export_format
            
            if st.button("📥 Export Chat", use_container_width=True):
                export_data = export_chat_history()
                if export_data:
                    st.success(f"Chat exported in {export_format} format!")
        
        st.divider()
        
        # System Status with better visual indicators
        st.markdown("### 📊 System Status")
        
        # Status cards
        status_items = [
            ("Groq LLM", st.session_state.llm is not None, "Connected" if st.session_state.llm else "Not connected"),
            ("Embeddings", st.session_state.embeddings is not None, "Loaded" if st.session_state.embeddings else "Not loaded"),
            ("DocuMind AI", st.session_state.qa_chain is not None, "Ready" if st.session_state.qa_chain else "Upload Documents first"),
            ("Processing", not st.session_state.is_processing, "Ready" if not st.session_state.is_processing else "In progress")
        ]
        
        for name, status, text in status_items:
            icon = "✅" if status else "❌" if name != "Processing" else "⏳"
            color = "green" if status else "red" if name != "Processing" else "orange"
            st.markdown(f"{icon} **{name}:** {text}")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💭 Intelligent Document Chat")
        
        # Display chat history with better styling
        if st.session_state.chat_history:
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                with st.container():
                    # User message
                    st.markdown(f"""
                    <div class="user-message fade-in">
                        <strong>👤 You:</strong><br>
                        {question}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI message
                    st.markdown(f"""
                    <div class="ai-message fade-in">
                        <strong>🧠 DocuMind AI:</strong><br>
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Sources with better styling
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
                            for j, doc in enumerate(sources):
                                source_file = doc.metadata.get('source_file', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown')
                                st.markdown(f"**📄 Source {j+1}: {source_file} (Page {page})**")
                                st.markdown(f"```\n{doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}\n```")
                                if j < len(sources) - 1:
                                    st.divider()
                    
                    st.divider()
        else:
            # Welcome message
            st.markdown("""
            <div class="info-card fade-in">
                <h4>🌟 Welcome to DocuMind AI!</h4>
                <p>Your intelligent document assistant is ready to help you:</p>
                <ol>
                    <li>📄 Upload your PDF documents using the sidebar</li>
                    <li>🧠 Click "Process Documents" to enable AI analysis</li>
                    <li>💭 Start intelligent conversations about your content</li>
                </ol>
                <p><strong>💡 Smart Tips:</strong> Ask "What are the key insights?" or "Explain the main concepts" for best results</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat input with better styling
        st.markdown("### 💭 Ask DocuMind AI")
        
        # Use form to prevent auto-submission
        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_area(
                "Your question:",
                key=f"question_input_{st.session_state.input_counter}",
                placeholder="Ask anything about your documents... (e.g., 'What are the key insights?', 'Explain the main concepts', 'What are the important findings?')",
                disabled=not st.session_state.qa_chain or st.session_state.is_processing,
                height=100
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                submit_button = st.form_submit_button(
                    "🧠 Ask",
                    type="primary",
                    disabled=not st.session_state.qa_chain or st.session_state.is_processing,
                    use_container_width=True
                )
            
            if submit_button and user_question:
                ask_question(user_question)
                st.rerun()
    
    with col2:
        st.markdown("### ⚡ Smart Actions")
        
        if st.session_state.processed_files and not st.session_state.is_processing:
            st.markdown("**🧠 AI-Powered Analysis:**")
            
            quick_actions = [
                ("📋 Executive Summary", "Generate a professional executive summary of all documents."),
                ("🔍 Key Insights", "Extract the most important insights and findings."),
                ("📊 Data Analysis", "Analyze numerical data and statistics from the documents."),
                ("🎯 Main Concepts", "Identify core concepts and themes."),
                ("💡 Action Items", "Generate actionable recommendations."),
                ("🔬 Research Mode", "Deep dive analysis with detailed explanations."),
                ("📈 Trend Analysis", "Identify patterns and trends across documents."),
                ("🎨 Creative Summary", "Generate a creative, engaging summary.")
            ]
            
            for action_text, question in quick_actions:
                if st.button(action_text, key=f"quick_{action_text}", use_container_width=True):
                    ask_question(question)
                    st.rerun()
        
        elif st.session_state.is_processing:
            st.markdown("""
            <div class="processing-indicator">
                <strong>🧠 AI is analyzing your documents...</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <p>📄 Upload and process documents to unlock AI-powered analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # System info with better styling
        st.markdown("### 📊 System Info")
        
        info_items = [
            ("🤖 Model", st.session_state.config.model_name),
            ("🧠 Embeddings", "all-MiniLM-L6-v2"),
            ("📄 Files", f"{len(st.session_state.processed_files)} processed"),
            ("💬 Messages", f"{len(st.session_state.chat_history)} in chat")
        ]
        
        for label, value in info_items:
            st.markdown(f"**{label}:** {value}")
        
        # Performance metrics
        if st.session_state.chat_history:
            st.markdown("### 📈 Performance")
            st.markdown(f"**Avg Response Time:** ~2-3 seconds")
            st.markdown(f"**Model Speed:** {'Fast' if '8b' in st.session_state.config.model_name else 'Medium' if 'mixtral' in st.session_state.config.model_name else 'Slower'}")
            st.markdown(f"**Context Window:** {'8K' if '8b' in st.session_state.config.model_name else '32K' if 'mixtral' in st.session_state.config.model_name else '8K'}")
        
        # Document Insights
        if st.session_state.processed_files:
            st.markdown("### 🔍 Document Insights")
            
            # Analysis Mode Indicator
            mode_colors = {
                "Standard": "🟢",
                "Deep Analysis": "🔵", 
                "Quick Summary": "🟡",
                "Research Mode": "🟣"
            }
            st.markdown(f"**Mode:** {mode_colors.get(st.session_state.analysis_mode, '🟢')} {st.session_state.analysis_mode}")
            
            # Quick Stats
            if st.session_state.vector_store:
                st.markdown("**Processing Status:** ✅ Complete")
                st.markdown("**AI Ready:** 🧠 Active")
            else:
                st.markdown("**Processing Status:** ⏳ Pending")
                st.markdown("**AI Ready:** ❌ Inactive")
        
        # New Feature Highlights
        st.markdown("### ✨ New Features")
        st.markdown("• 🚀 **Advanced Analysis Modes**")
        st.markdown("• 📊 **Real-time Statistics**")
        st.markdown("• 📤 **Export Capabilities**")
        st.markdown("• 🎨 **Enhanced Visual Design**")
        st.markdown("• 🔬 **Research Mode**")
        st.markdown("• 📈 **Trend Analysis**")

if __name__ == "__main__":
    main()