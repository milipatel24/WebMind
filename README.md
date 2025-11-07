<<<<<<< HEAD
# URL → Q&A Chatbot (Google Gemini + LangChain + Streamlit)

This project demonstrates a simple retrieval-augmented generation (RAG) app that:
- Takes a webpage URL and a user question.
- Extracts and chunks the page text.
- Creates embeddings using **Google Gemini** (`gemini-embedding-001`).
- Builds an in-memory FAISS vector index and retrieves relevant chunks.
- Calls **Google Gemini** chat model (`gemini-2.5-flash`) to answer using retrieved context.
- UI built with Streamlit.

---

## Files created
- `app.py` - Streamlit application.
- `google_embeddings.py` - LangChain-compatible embeddings wrapper that calls Gemini embeddings.
- `requirements.txt` - Python dependencies.
- `.env.example` - Example environment file for your Gemini API key.
- `README.md` - This document.

## Setup and run (local)

1. Clone or unzip the project directory.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\\Scripts\\activate    # Windows (PowerShell)
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   # edit .env and set GEMINI_API_KEY=your_key_here
   ```
   You can get a Gemini API key from Google AI Studio or Google Cloud (Generative Language API).
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
6. In the web UI enter a webpage URL and your question, then click **Run**.

Notes & caveats:
- This example uses an in-memory FAISS index and is intended for demo / toy usage only.
- API usage may incur costs. Monitor your Google account.
- Adjust chunk sizes, model names, and retrieval parameters as needed.
=======
# WebMind
>>>>>>> c3e1974fde78a2b46d018a8dc6ef25c439d3229b
