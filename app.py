import os
import io
import numpy as np
import streamlit as st
from typing import List, Tuple
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided document context to answer the user's queries. "
    "If the answer is not explicitly supported by the document, say you are not sure based on the document, "
    "and optionally provide a general best-effort answer. Be concise. When helpful, quote small snippets."
)


def get_openai_client(api_key: str = "") -> OpenAI:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        st.error("PyPDF2 is required to read PDFs. Please install it: pip install PyPDF2")
        return ""
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts)
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
        return ""


def clean_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(
    text: str,
    chunk_size_words: int = 800,
    overlap_words: int = 150
) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    # Batched embeddings for efficiency on large docs
    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(model=model, input=batch)
        embeddings.extend([d.embedding for d in res.data])
    return embeddings


def normalize_embeddings(embeds: List[List[float]]) -> np.ndarray:
    arr = np.array(embeds, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def retrieve_context(
    client: OpenAI,
    query: str,
    chunks: List[str],
    norm_embeds: np.ndarray,
    top_k: int = 5,
    embed_model: str = "text-embedding-3-small",
) -> Tuple[str, List[Tuple[int, float]]]:
    q_emb = client.embeddings.create(model=embed_model, input=[query]).data[0].embedding
    q = np.array(q_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = norm_embeds @ q
    top_idx = np.argsort(-sims)[:top_k]
    ordered = [(int(i), float(sims[i])) for i in top_idx]
    context_parts = []
    for i, _ in ordered:
        context_parts.append(chunks[i])
    context = "\n\n".join(context_parts)
    return context, ordered


def build_messages(system_prompt: str, context: str, history: List[dict], user_query: str) -> List[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    if context.strip():
        msgs.append({"role": "system", "content": f"Document context:\n{context}"})
    # Append truncated history to keep prompt size reasonable
    # We keep last 6 turns (12 messages) by default
    max_history = 12
    trimmed_history = history[-max_history:] if len(history) > max_history else history
    msgs.extend(trimmed_history)
    msgs.append({"role": "user", "content": user_query})
    return msgs


def init_session_state():
    defaults = {
        "messages": [],
        "doc_text": "",
        "doc_chunks": [],
        "doc_name": None,
        "norm_embeddings": None,
        "ready": False,
        "top_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_document(client: OpenAI, uploaded_file):
    name = uploaded_file.name
    text = extract_text_from_pdf(uploaded_file)
    text = clean_text(text)
    if not text:
        st.session_state.ready = False
        st.error("No text extracted from the PDF.")
        return
    chunks = chunk_text(text)
    if not chunks:
        st.session_state.ready = False
        st.error("Failed to split document into chunks.")
        return
    embeds = embed_texts(client, chunks)
    norm_embeds = normalize_embeddings(embeds)
    st.session_state.doc_name = name
    st.session_state.doc_text = text
    st.session_state.doc_chunks = chunks
    st.session_state.norm_embeddings = norm_embeds
    st.session_state.ready = True
    st.session_state.messages = []
    st.success(f"Processed document: {name} ({len(chunks)} chunks).")


def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
    init_session_state()

    with st.sidebar:
        st.title("📄 PDF Chatbot")
        api_key = st.text_input("OpenAI API Key", type="password", help="Will use environment variable OPENAI_API_KEY if left blank.")
        model = st.selectbox("Chat Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=5)
        st.divider()
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button("Process PDF", use_container_width=True, type="primary", disabled=uploaded_file is None):
                client = get_openai_client(api_key)
                process_document(client, uploaded_file)
        with col_sb2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []

        if st.session_state.ready:
            st.caption(f"Loaded: {st.session_state.doc_name} | Chunks: {len(st.session_state.doc_chunks)}")

    st.header("Chat with your PDF")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        if not (api_key or os.environ.get("OPENAI_API_KEY")):
            st.error("Please enter your OpenAI API key in the sidebar.")
            return
        if not st.session_state.ready:
            st.error("Please upload and process a PDF first.")
            return

        client = get_openai_client(api_key)

        # Retrieve relevant context
        context, top_hits = retrieve_context(
            client=client,
            query=user_input,
            chunks=st.session_state.doc_chunks,
            norm_embeds=st.session_state.norm_embeddings,
            top_k=top_k,
        )

        # Prepare messages with context
        messages = build_messages(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            history=st.session_state.messages,
            user_query=user_input
        )

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call OpenAI chat completion
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4" if model == "gpt-4" else "gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.2,
                    )
                    assistant_reply = response.choices[0].message.content
                except Exception as e:
                    assistant_reply = f"Error from OpenAI API: {e}"

            st.write(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            st.session_state.top_sources = top_hits

    # Sources panel
    with st.expander("Show retrieved sources"):
        if st.session_state.ready and st.session_state.top_sources:
            for idx, score in st.session_state.top_sources:
                st.markdown(f"Chunk {idx} (similarity: {score:.3f})")
                st.write(st.session_state.doc_chunks[idx][:1200] + ("..." if len(st.session_state.doc_chunks[idx]) > 1200 else ""))
                st.divider()
        elif st.session_state.ready:
            st.caption("Ask a question to see the most relevant chunks.")

    # Footer
    st.caption("Your API key is never stored. It is used only to call the OpenAI API from this app.")


if __name__ == "__main__":
    main()