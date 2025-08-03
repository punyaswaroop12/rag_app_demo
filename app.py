import streamlit as st
import os
from pathlib import Path
import tempfile
from rag_engine import RAGEngine

# Initialize RAG engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

# Page configuration
st.set_page_config(
    page_title="RAG Application with Azure",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Application with Azure OpenAI")
st.markdown("Upload documents and ask questions about them!")

# Initialize RAG engine
rag_engine = get_rag_engine()

# Sidebar for file upload and system info
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_paths.append(tmp_file.name)
                
                # Ingest documents
                try:
                    rag_engine.ingest_documents(temp_paths)
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                
                # Clean up temporary files
                for path in temp_paths:
                    os.unlink(path)
    
    st.divider()
    
    # System statistics
    st.header("📊 System Stats")
    stats = rag_engine.get_system_stats()
    st.json(stats)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input("Enter your question:", key="query_input")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        k_value = st.slider("Number of context chunks to retrieve", 1, 10, 5)
    
    if st.button("Ask Question", type="primary") and query:
        with st.spinner("Searching and generating response..."):
            try:
                result = rag_engine.query(query, k=k_value)
                
                st.subheader("🎯 Answer")
                st.write(result['answer'])
                
                st.subheader("📚 Sources")
                if result['sources']:
                    for source in result['sources']:
                        st.write(f"• {Path(source).name}")
                else:
                    st.write("No sources found")
                
                st.subheader("📈 Metadata")
                st.write(f"Context chunks used: {result['context_chunks']}")
                
            except Exception as e:
                st.error(f"Error processing query: {e}")

with col2:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
    2. **Process**: Click "Process Documents" to ingest them into the system
    3. **Ask Questions**: Enter your question in the text input
    4. **Get Answers**: The system will retrieve relevant context and generate an answer
    
    **Tips:**
    - Upload multiple related documents for better context
    - Ask specific questions for better results
    - Use the advanced options to control retrieval
    """)
    
    st.header("🔧 Configuration")
    st.markdown("""
    Make sure your `.env` file contains:
    ```
    AZURE_OPENAI_API_KEY=your_key
    AZURE_OPENAI_ENDPOINT=your_endpoint
    AZURE_OPENAI_API_VERSION=2023-12-01-preview
    EMBEDDING_MODEL_NAME=text-embedding-ada-002
    CHAT_MODEL_NAME=gpt-35-turbo
    ```
    """)

# Footer
st.divider()
st.markdown("Built with Azure OpenAI, FAISS, and Streamlit")
