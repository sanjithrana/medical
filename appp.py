#this is my appp.py file using flask with 8000 host
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-chatbot"
GROQ_API_KEY= os.getenv("GROQ_API_KEY")
#
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Set up embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to Pinecone vectorstore
vectorstore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embedding_model)

# LLM setup
llm = ChatGroq(model="llama3-8b-8192", temperature=0,api_key=GROQ_API_KEY)

# Prompt

prompt_template = PromptTemplate(
    template="""
    You are a medical assistant. Use the context below to answer the question.
    You are a knowledgeable medical assistant. Use the following context to answer the question as accurately and clearly as possible. 
    If the answer is not in the context, say you don't know — do not make up information.

    Context: {context}

    Question: {input}
    """,
    input_variables=["context", "input"]
)

# Create retrieval chain
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 3})
stuff_chain = create_stuff_documents_chain(llm, prompt_template)
retrieval_chain = create_retrieval_chain(retriever, stuff_chain)

# Routes
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/query", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."})
    
    try:
        response = retrieval_chain.invoke({"input": question})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    """app.run(debug=True, use_reloader=False)"""

    app.run(host="0.0.0.0", port=8000,debug=False, use_reloader=False)
