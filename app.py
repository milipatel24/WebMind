
import os, streamlit as st, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from google_embeddings import GoogleEmbeddings
from openai import OpenAI
import tempfile
import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def extract_clean_text(url: str):
    """Fetch webpage content and extract clean text using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "aside", "svg"]):
            tag.decompose()

        # Extract visible text
        text = soup.get_text(separator=" ", strip=True)

        # Optional: remove long sequences of spaces
        text = " ".join(text.split())

        return text

    except Exception as e:
        st.error(f"Error fetching/cleaning URL: {e}")
        return ""




load_dotenv()

st.set_page_config(page_title="URL → Q&A (Gemini + LangChain)", layout="centered")

st.title("WebMind")
# st.markdown("""Enter a webpage URL and a question. The app will fetch the page, create embeddings from the page text using Google's **gemini-embedding-001** model, build a vector index, retrieve relevant passages, and ask Gemini to answer using those passages as context.""")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it to a .env file (see README).")
    st.stop()

url = st.text_input("Webpage URL", placeholder="https://example.com/article")
query = st.text_input("Your question about the page")

top_k = st.slider("How many passages to retrieve (k)", 1, 10, 3)

if st.button("Run") and url and query:
    with st.spinner("Fetching and cleaning webpage..."):
        text = extract_clean_text(url)

        if not text:
            st.error("Could not extract text from webpage. Please check the URL.")
            st.stop()

    #st.success("Page fetched — extracting and chunking text...")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages = splitter.split_text(text)
    docs = [Document(page_content=p, metadata={"source": url, "chunk": i}) for i,p in enumerate(pages)]

    #st.write(f"Extracted {len(docs)} chunks. Building embeddings & vector index... (may take a few seconds)")

    # Create LangChain-compatible embedding object that calls Gemini embeddings
    embedder = GoogleEmbeddings(api_key=GEMINI_API_KEY, model="gemini-embedding-001")

    # Build FAISS vectorstore (in-memory). FAISS will call our embedder when creating the index.
    try:
        vectorstore = FAISS.from_documents(docs, embedder)
    except Exception as e:
        st.error(f"Error building vectorstore: {e}")
        st.stop()

    #st.success("Vector index built. Retrieving top-k relevant passages...")
    results = vectorstore.similarity_search(query, k=top_k)
    context_text = "\n\n---\n\n".join([f"Source chunk {r.metadata.get('chunk','?')}:\\n{r.page_content[:1500]}" for r in results])

    #st.subheader("Retrieved context (truncated)")
    for r in results:
        clean_text = r.page_content[:600].replace('\n', ' ')
        #st.markdown(f"**Chunk {r.metadata.get('chunk','?')}** — {clean_text}...")


    #st.info("Sending context and question to Gemini for a final answer...")

    # Build prompt for Gemini
    system = "You are an assistant that answers questions based only on the provided context. If the answer is not in the context, say you don't know."
    user_message = f"""Context:
    {context_text}

    Question: {query}

    Instructions: Answer concisely and cite the chunk numbers (e.g., 'chunk 2') when you reference the context."""

    client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    try:
        completion = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=800
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Gemini chat completions: {e}")
        st.stop()

    st.subheader("Answer from WebMind:")
    st.write(answer)

    #st.markdown("---")
    #st.write("**Raw retrieved chunks (for debugging)**")
    #for i, r in enumerate(results):
        #st.write(f"Chunk {i} (metadata: {r.metadata}):\n{r.page_content[:2000]}")