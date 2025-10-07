from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from typing import List
from pydantic import Field


class RedundantFilterRetriever(BaseRetriever):
    """
    Retriever that uses Max Marginal Relevance (MMR) search to ensure
    both relevance and diversity of retrieved chunks, and also filters
    out highly similar documents above a threshold.
    """
    
    # Declare all attributes as Pydantic fields
    vectorstore: Chroma = Field(description="Chroma vectorstore instance")
    embeddings: OpenAIEmbeddings = Field(default_factory=OpenAIEmbeddings)
    threshold: float = Field(default=0.8, description="Similarity threshold for filtering")
    k: int = Field(default=5, description="Number of documents to return")
    fetch_k: int = Field(default=20, description="Number of candidates to fetch for MMR")
    lambda_mult: float = Field(default=0.5, description="Balance between relevance and diversity")

    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like Chroma

    def _similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve top-k diverse documents using MMR and filter redundant ones.
        """
        # Step 1: Embed the query
        query_vector = self.embeddings.embed_query(query)

        # Step 2: Perform Max Marginal Relevance (MMR) search
        mmr_results = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=query_vector,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

        # Step 3: Redundancy filtering with cached embeddings
        filtered_docs = []
        filtered_embeddings = []
        
        for doc in mmr_results:
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            
            # Check if similar to any already filtered document
            is_redundant = False
            for existing_emb in filtered_embeddings:
                if self._similarity(doc_embedding, existing_emb) >= self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered_docs.append(doc)
                filtered_embeddings.append(doc_embedding)

        return filtered_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of _get_relevant_documents()."""
        # Embed the query asynchronously
        query_vector = await self.embeddings.aembed_query(query)
        
        # Perform MMR search
        mmr_results = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=query_vector,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult,
        )

        # Redundancy filtering with cached embeddings
        filtered_docs = []
        filtered_embeddings = []
        
        for doc in mmr_results:
            doc_embedding = await self.embeddings.aembed_query(doc.page_content)
            
            is_redundant = False
            for existing_emb in filtered_embeddings:
                if self._similarity(doc_embedding, existing_emb) >= self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered_docs.append(doc)
                filtered_embeddings.append(doc_embedding)

        return filtered_docs