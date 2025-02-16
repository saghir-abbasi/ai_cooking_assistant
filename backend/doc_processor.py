# backend/doc_processor.py
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional, List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time
load_dotenv()

class DocProcessor:
    """Handles document loading, splitting, and vector storage"""
    
    def __init__(self, embedding_model: str):
        # self.persist_dir = persist_dir
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=os.getenv("GOOGLE_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False
        )
    
    def load_and_split(self, uploaded_file) -> Optional[List]:
        """Load and split PDF document"""
        try:
            # Create temporary file with explicit cleanup
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            # Load and process document
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)
            return split_docs
        except Exception as e:
            raise ValueError(f"Document processing failed: {str(e)}")
        finally:
            # Clean up temporary file
            if 'temp_path' in locals():
                try:
                    os.remove(temp_path)
                except PermissionError:
                    pass  # Handle cases where file might still be open
    def create_vector_store(self, docs: Optional[List] = None):
        """Create or load Pinecone vector store"""
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        index = pc.Index("recipes-project")
        vector_store = PineconeVectorStore(embedding=self.embedding_model, index=index)   #create a vector store
        if docs:
            vector_store.delete(delete_all=True)  # Remove existing vectors (if required)
            vector_store.add_documents(docs)
            return vector_store.as_retriever()
        # Check if the index already contains vectors
        if index.describe_index_stats().get("total_vector_count", 0) > 0:
            return vector_store.as_retriever()
        raise ValueError("No documents provided and no existing vector store found")
    
    