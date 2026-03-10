import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

class TFIDFRetriever:
    def __init__(self, knowledge_file="knowledge.md"):
        self.knowledge_file = knowledge_file
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.docs = []
        self.tfidf_matrix = None
        self._initialize()

    def _initialize(self):
        """Loads and processes the knowledge base into chunks."""
        try:
            # Look for knowledge.md in the root of the backend folder
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, self.knowledge_file)
            
            if not os.path.exists(file_path):
                print(f"Warning: {self.knowledge_file} not found at {file_path}")
                self.docs = ["No knowledge base available."]
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split by headers (simple chunking strategy)
                self.docs = re.split(r'\n(?=# )', content)
                self.docs = [d.strip() for d in self.docs if d.strip()]
            
            if self.docs:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)
                print(f"TF-IDF Index built with {len(self.docs)} chunks.")
        except Exception as e:
            print(f"Error initializing TF-IDF Retriever: {e}")
            self.docs = ["Error loading knowledge base."]
            self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def get_relevant_documents(self, query: str, k: int = 3):
        """Retrieves top k relevant chunks for a query."""
        if self.tfidf_matrix is None or not self.docs:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        relevant_chunks = []
        for i in top_k_indices:
            if similarities[i] > 0: # Only return chunks with some similarity
                relevant_chunks.append(self.docs[i])
        
        return relevant_chunks

# Global retriever instance for preloading
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = TFIDFRetriever()
    return _retriever

def get_rag_chain(api_key: str = None):
    """Returns a function that performs the RAG logic."""
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    
    retriever = get_retriever()
    
    # Use provided api_key if available, otherwise it falls back to environment
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
        groq_api_key=api_key
    )
    
    template = """You are an HR Assistant for a company. Answer the question based ONLY on the following context.
    If you cannot find the answer in the context, say that you don't know based on the available data.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(docs)

    # Simplified RAG chain logic without heavy LangChain dependencies
    def chain(question: str):
        context_docs = retriever.get_relevant_documents(question)
        context_text = format_docs(context_docs)
        
        messages = prompt.format_messages(context=context_text, question=question)
        response = llm.invoke(messages)
        return response.content

    return chain
