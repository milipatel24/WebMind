
from langchain.embeddings.base import Embeddings
from typing import List, Sequence
from openai import OpenAI
import os

class GoogleEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper that calls Google's Gemini embeddings
    via the OpenAI-compatible client (per Google's OpenAI-compatible endpoint).
    """
    def __init__(self, api_key: str = None, model: str = "gemini-embedding-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batch embeddings call
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
