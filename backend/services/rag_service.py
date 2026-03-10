import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load .env from project root
# (On Render, this won't matter as we use real Env Vars)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(base_dir, '.env'))

# Global variables for lazy loading
_chunks = None
_vectorizer = None
_tfidf_matrix = None

def get_knowledge_index():
    """Build a lightweight TF-IDF index from knowledge.md (uses ~5MB vs ~300MB for HuggingFace)"""
    global _chunks, _vectorizer, _tfidf_matrix
    if _chunks is None:
        print("Initializing RAG knowledge base with TF-IDF...")
        
        # Load knowledge file
        with open('knowledge.md', 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into chunks by section headers and paragraphs
        raw_chunks = []
        current_chunk = ""
        for line in text.split('\n'):
            current_chunk += line + "\n"
            # Split on section breaks or when chunk gets large enough
            if len(current_chunk) > 500 or (line.startswith('---') and len(current_chunk) > 100):
                if current_chunk.strip():
                    raw_chunks.append(current_chunk.strip())
                current_chunk = ""
        if current_chunk.strip():
            raw_chunks.append(current_chunk.strip())
        
        _chunks = raw_chunks
        
        # Build TF-IDF matrix
        _vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)  # unigrams + bigrams for better matching
        )
        _tfidf_matrix = _vectorizer.fit_transform(_chunks)
        print(f"RAG knowledge base ready: {len(_chunks)} chunks indexed.")
    
    return _chunks, _vectorizer, _tfidf_matrix


def ask_question(query: str, api_key: str = None):
    # Determine API key
    groq_api_key = api_key if api_key else os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        return "Error: No Groq API Key provided. Please enter your API Key in the sidebar."

    from langchain_groq import ChatGroq
    
    # Initialize LLM dynamically per request
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama-3.1-8b-instant'
        )
    except Exception as e:
        return f"Error connecting to Groq API. Please check your API key. Details: {e}"

    # Retrieve relevant chunks using TF-IDF similarity
    chunks, vectorizer, tfidf_matrix = get_knowledge_index()
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 3 most relevant chunks
    top_indices = np.argsort(similarities)[-3:][::-1]
    context = "\n".join([chunks[i] for i in top_indices if similarities[i] > 0.05])
    
    if not context:
        context = "\n".join([chunks[i] for i in top_indices])

    # prompt
    prompt = f"""You are an expert HR analytics assistant for an employee attrition prediction system.

Rules:
- Answer ONLY using the provided context. Never fabricate information.
- If the answer is not in the context, say: "This information is not available in the knowledge base."
- Be concise and direct — aim for under 150 words unless the question requires detailed explanation.
- Use bullet points for lists and comparisons.
- Include specific numbers, percentages, and statistics when available in the context.
- Structure multi-part answers with bold headings.
- Do not repeat the question in your answer.

Context:
{context}

Question: {query}

Answer:"""

    # Call LLM
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error invoking LLM: {e}"
