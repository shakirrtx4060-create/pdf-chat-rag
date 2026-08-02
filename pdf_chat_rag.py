"""
Simple RAG Website — Upload PDFs and Chat with them
Run with:  streamlit run pdf_chat_rag.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat (RAG)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with your PDFs")
st.caption("Upload PDF files → ask questions → get answers grounded in your documents")

# ──────────────────────────────────────────────
# Sidebar — settings + upload
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com",
    )

    model_name = st.selectbox(
        "LLM Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
    )

    top_k = st.slider("Documents to retrieve (top-k)", 2, 8, 4)

    st.divider()
    st.header("📤 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once",
    )

    process_btn = st.button("Process PDFs", type="primary", use_container_width=True)

    if st.button("Clear everything", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    """Load embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def process_pdfs(files) -> Chroma | None:
    """Load, split, embed and store uploaded PDFs."""
    if not files:
        return None

    all_docs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            # Save uploaded file to disk so PyPDFLoader can read it
            path = Path(tmpdir) / f.name
            path.write_bytes(f.getvalue())

            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = f.name
            all_docs.extend(docs)

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_chat",
    )
    return vectorstore


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')} | page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


# ──────────────────────────────────────────────
# Process uploaded files
# ──────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Processing PDFs (loading → chunking → embedding)…"):
            vs = process_pdfs(uploaded_files)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.messages = []  # reset chat
                st.sidebar.success(f"Processed {len(uploaded_files)} file(s) successfully!")
            else:
                st.sidebar.error("Could not extract text from the PDFs.")


# ──────────────────────────────────────────────
# Main area — status + chat
# ──────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process PDFs** to start chatting.")
    st.stop()

# Show which files are loaded
st.success(
    f"Ready — {len(st.session_state.file_names)} file(s) loaded: "
    + ", ".join(st.session_state.file_names)
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs…"):
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )

                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_name,
                    temperature=0.1,
                    max_tokens=1024,
                )

                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a helpful assistant. Answer the question using ONLY the context below. "
                            "If the context does not contain enough information, say so clearly. "
                            "When possible, mention the source file and page number.\n\n"
                            "Context:\n{context}",
                        ),
                        ("human", "{question}"),
                    ]
                )

                chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(prompt)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
