import os
from dotenv import load_dotenv

# Load .env from project root
# (On Render, this won't matter as we use real Env Vars)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(base_dir, '.env'))

# Global variables for lazy loading
_vectorstore = None
_embeddings = None
INDEX_PATH = "faiss_index"

def get_vectorstore():
    global _vectorstore, _embeddings
    if _vectorstore is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        # Check if local index exists
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(INDEX_PATH):
            print(f"Loading existing RAG index from {INDEX_PATH}...")
            _vectorstore = FAISS.load_local(
                INDEX_PATH, 
                _embeddings, 
                allow_dangerous_deserialization=True
            )
            print("RAG index loaded.")
        else:
            print("Index not found. Initializing RAG vectorstore from knowledge.md...")
            from langchain_text_splitters import CharacterTextSplitter
            from langchain_core.documents import Document
            
            # load knowledge
            with open('knowledge.md', 'r', encoding='utf-8') as f:
                text = f.read()

            # split into chunks
            splitter = CharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=70
            )

            chunks = splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            # create vector store
            _vectorstore = FAISS.from_documents(documents, _embeddings)
            
            # Save local index for next time
            _vectorstore.save_local(INDEX_PATH)
            print(f"RAG vectorstore initialized and saved to {INDEX_PATH}.")
            
    return _vectorstore

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

    # Retrieve relevant chunks from the lazy-loaded vectorstore
    vector_db = get_vectorstore()
    docs = vector_db.similarity_search(query, k=3)

    # Combine context
    context = "\n".join([doc.page_content for doc in docs])

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
