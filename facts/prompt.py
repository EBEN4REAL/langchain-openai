from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from redundant_filter_retriever import RedundantFilterRetriever
import os

load_dotenv()

# 5️⃣ Initialize your LLM
llm = ChatOpenAI(temperature=0.3)

# 1️⃣ Load your document
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "facts.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 2️⃣ Split into smaller chunks for better semantic search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,   # try 300–800 for small text files
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)

# 3️⃣ Create embeddings and build Chroma
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(
    documents=split_docs,   # ✅ use the chunks, not the full file
    embedding=embeddings,
    persist_directory="./facts/facts_chroma_db"
)

retriever = db.as_retriever()  # retrieve top 1 chunks

# Create your custom retriever
retriever = RedundantFilterRetriever(
    vectorstore=db,
    threshold=0.8,      # Similarity threshold for filtering
    k=5,                # Number of documents to return
    fetch_k=20,         # Number of candidates to fetch for MMR
    lambda_mult=0.5     # Balance between relevance (1.0) and diversity (0.0)
)

# 6️⃣ Build the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",  # The LLM reads each chunk individually and summarizes or answers the question for that chunk, The partial answers/summaries are merged and summarized again into a final answer.
    # chain_type="stuff",     # “Here are 3 text chunks → LLM, answer using them all.”
    retriever=retriever, 
    return_source_documents=True,  # optional, to see what context was used (This is optional but very useful for debugging and explainability.)
)

result = qa.invoke({"query": "What is an interesting fact about the English Language?"})


print(result["result"]) # the answer
print(len(result["source_documents"]))      # the chunks retrieved from your DB




