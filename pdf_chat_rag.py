"""
Simple RAG Website — Chat with PDFs stored in a repo folder
Run with:  streamlit run pdf_chat_rag.py

PDFs are read automatically from the "catalog_pdf/" folder next to this
script (i.e. inside your GitHub repo) — no upload widget needed.
"""

import os
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
# Config
# ──────────────────────────────────────────────
CATALOG_DIR = Path(__file__).parent / "catalog_pdf"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat (RAG)",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Chat with our PDF Catalog")
st.caption("Ask questions → get answers grounded in the documents in our catalog")

# ──────────────────────────────────────────────
# Sidebar — settings only (no upload)
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
            "qwen/qwen3.6-27b",
        ],
        index=0,
    )

    top_k = st.slider("Documents to retrieve (top-k)", 2, 8, 4)

    st.divider()
    st.header("📚 Catalog")

    if CATALOG_DIR.exists():
        pdf_files_in_folder = sorted(CATALOG_DIR.glob("*.pdf"))
        if pdf_files_in_folder:
            st.caption(f"{len(pdf_files_in_folder)} PDF(s) found:")
            for p in pdf_files_in_folder:
                st.caption(f"• {p.name}")
        else:
            st.warning(f"No PDFs found in '{CATALOG_DIR.name}/'.")
    else:
        st.error(f"Folder '{CATALOG_DIR.name}/' not found next to the app.")

    refresh_btn = st.button(
        "🔄 Rebuild index from catalog", type="primary", use_container_width=True
    )

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
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


def _catalog_signature(folder: Path) -> tuple:
    """A cheap fingerprint of the folder's contents (names + mtimes),
    used to decide whether the vectorstore needs rebuilding."""
    if not folder.exists():
        return ()
    return tuple(
        sorted((p.name, p.stat().st_mtime) for p in folder.glob("*.pdf"))
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(folder_str: str, signature: tuple):
    """Load, split, embed and store all PDFs found in `folder_str`.

    `signature` is only used as a cache key — when the folder's contents
    change (files added/removed/modified), Streamlit will detect the new
    signature and rebuild automatically.
    """
    folder = Path(folder_str)
    pdf_paths = sorted(folder.glob("*.pdf"))
    if not pdf_paths:
        return None, []

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = path.name
        all_docs.extend(docs)

    if not all_docs:
        return None, []

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
    return vectorstore, [p.name for p in pdf_paths]


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')} | page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


# ──────────────────────────────────────────────
# Build / refresh the index from the catalog folder
# ──────────────────────────────────────────────
if refresh_btn:
    build_vectorstore.clear()  # force a rebuild even if signature is unchanged
    st.session_state.messages = []

signature = _catalog_signature(CATALOG_DIR)

with st.spinner("Loading catalog PDFs (this runs once, then is cached)…"):
    vectorstore, loaded_files = build_vectorstore(str(CATALOG_DIR), signature)

if vectorstore is None:
    st.info(
        f"No PDFs available yet. Add PDF files to the **{CATALOG_DIR.name}/** "
        f"folder in the repo and redeploy (or click 'Rebuild index from catalog')."
    )
    st.stop()

st.session_state.vectorstore = vectorstore
st.session_state.file_names = loaded_files

# ──────────────────────────────────────────────
# Main area — status + chat
# ──────────────────────────────────────────────
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
if prompt := st.chat_input("Ask a question about the catalog…"):
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
