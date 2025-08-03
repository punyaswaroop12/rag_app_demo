import os
from typing import List, Dict
import openai
from dotenv import load_dotenv
import logging
import streamlit as st

from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import FAISSVectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = FAISSVectorStore()
        
        # Initialize Azure OpenAI client for chat
        self.client = openai.AzureOpenAI(
            api_key=st.secrets["azure_openai"]["AZURE_OPENAI_API_KEY"],
            api_version=st.secrets["azure_openai"]["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["azure_openai"]["AZURE_OPENAI_ENDPOINT"],
        )
        self.chat_model = st.secrets['model_config']["CHAT_MODEL_NAME"]
    
    def ingest_documents(self, document_paths: List[str]):
        """Ingest documents into the RAG system"""
        logger.info("Starting document ingestion...")
        
        # Process documents
        chunks = self.document_processor.process_documents(document_paths)
        
        if not chunks:
            logger.error("No chunks extracted from documents")
            return
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        # Filter out empty embeddings
        valid_data = [(emb, chunk) for emb, chunk in zip(embeddings, chunks) if emb]
        
        if not valid_data:
            logger.error("No valid embeddings generated")
            return
        
        valid_embeddings, valid_chunks = zip(*valid_data)
        
        # Add to vector store
        self.vector_store.add_vectors(list(valid_embeddings), list(valid_chunks))
        self.vector_store.save_index()
        
        logger.info(f"Successfully ingested {len(valid_chunks)} chunks")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant context for a query"""
        query_embedding = self.embedding_service.get_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        results = self.vector_store.search(query_embedding, k)
        return [metadata for metadata, score in results]
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using retrieved context"""
        # Prepare context text
        context_text = "\n\n".join([
            f"Source: {ctx.get('source', 'Unknown')}\n{ctx['text']}" 
            for ctx in context
        ])
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Use the context below to answer the user's question. If the answer cannot be found in the context, 
say so clearly.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating the response."
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Complete RAG query: retrieve and generate"""
        logger.info(f"Processing query: {question}")
        
        # Retrieve context
        context = self.retrieve_context(question, k)
        
        if not context:
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': []
            }
        
        # Generate response
        answer = self.generate_response(question, context)
        
        # Extract sources
        sources = list(set([ctx.get('source', 'Unknown') for ctx in context]))
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_chunks': len(context)
        }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'vector_store_stats': self.vector_store.get_stats(),
            'models': {
                'embedding': self.embedding_service.model_name,
                'chat': self.chat_model
            }
        }
    