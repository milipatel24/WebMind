# WebMind (Google Gemini + LangChain + Streamlit)

This project demonstrates a simple retrieval-augmented generation (RAG) app that:
- Takes a webpage URL and a user question.
- Extracts and chunks the page text.
- Creates embeddings using **Google Gemini** (`gemini-embedding-001`).
- Builds an in-memory FAISS vector index and retrieves relevant chunks.
- Calls **Google Gemini** chat model (`gemini-2.5-flash`) to answer using retrieved context.
- UI built with Streamlit.
