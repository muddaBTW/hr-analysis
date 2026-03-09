import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# load knowledge
with open('knowledge.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# split into chunks
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)
documents = [Document(page_content=chunk) for chunk in chunks]

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name='llama-3.1-8b-instant'
)

# RAG
def ask_question(query: str):

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=3)

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
    response = llm.invoke(prompt)

    return response.content
