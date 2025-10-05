from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

load_dotenv()

# Load a single text file
current_dir = os.path.dirname(__file__)  # directory of this script
file_path = os.path.join(current_dir, "facts.txt")
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# print(f"Loaded {len(documents)} document(s)")
# print(f"Content preview: {documents[0].page_content[:100]}...")
# print(f"Metadata: {documents[0].metadata}\n")

def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        separators=["\n"]
    )
    
    chunks = text_splitter.split_documents(documents)
    # print(f"✅ Split into {len(chunks)} chunks")
    # print("CHUNKS => ", chunks)

    return chunks

docs = split_documents(documents)

# for i, doc in enumerate(docs[:1], 1):  # Print first 2 chunks as a sample
#     print(f"--- Chunk {i} ---\n{doc.page_content}\nMetadata: {doc.metadata}\n")
    

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # OpenAI's embedding model
)

emb = embeddings.embed_documents([doc.page_content for doc in docs])

# emb = embeddings.embed_query("What is a programming language?") 

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="emb",  # Name your collection
    persist_directory="./facts/facts_chroma_db"
)

query = "What is the meaning of sleep?"

results_with_scores = vector_store.similarity_search_with_score(query, k=3)

results = vector_store.similarity_search(query, k=2)

print("\n📊 Results:")
for i, doc in enumerate(results, 1):
    print("\n--- Result {i} ---")
    print(f"{doc.page_content}")

# print("\n📊 Results with similarity scores:")
# for i, (doc, score) in enumerate(results_with_scores, 1):
#     print(f"  {i}. Score: {score:.4f}")
#     print(f"     Content: {doc.page_content}")
#     print(f"     Category: {doc.metadata['source']}\n")