import os
import numpy as np
from typing import List
import openai
from dotenv import load_dotenv
import logging
import streamlit as st

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = openai.AzureOpenAI(
            api_key=st.secrets["azure_openai"]["AZURE_OPENAI_API_KEY"],
            api_version=st.secrets["azure_openai"]["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["azure_openai"]["AZURE_OPENAI_ENDPOINT"],
        )
        self.model_name = st.secrets["model_config"]["EMBEDDING_MODEL_NAME"]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    