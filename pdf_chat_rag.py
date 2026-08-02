import os
import streamlit as st
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs_from_folder(folder_path: str) -> Chroma | None:
    """Load, split, embed and store PDFs from a fixed folder."""
    all_docs = []
    pdf_dir = Path(folder_path)

    for path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = path.name
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

def get_llm(api_key: str, model_name: str):
    return ChatGroq(groq_api_key=api_key, model_name=model_name)

def get_qa_chain(llm, vectorstore, top_k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question.
        If you don't know the answer, say you don't know.
        Keep the answer concise and clear.

        Context: {context}
        Question: {question}
        Answer:"""
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(page_title="📚 PDF Chat RAG", layout="wide")
st.title("📚 PDF Chat RAG")

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

# ──────────────────────────────────────────────
# Load PDFs automatically from catalog_pdfs
# ──────────────────────────────────────────────

folder_path = "catalog_pdfs"
vs = process_pdfs_from_folder(folder_path)

if vs:
    st.session_state.vectorstore = vs
    st.session_state.file_names = [p.name for p in Path(folder_path).glob("*.pdf")]
    st.success(f"Loaded {len(st.session_state.file_names)} PDFs: {', '.join(st.session_state.file_names)}")
else:
    st.error("No PDFs found in catalog_pdfs folder.")

# ──────────────────────────────────────────────
# Chat Interface with history
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if api_key and "vectorstore" in st.session_state:
    llm = get_llm(api_key, model_name)
    qa_chain = get_qa_chain(llm, st.session_state.vectorstore, top_k)

    if query := st.chat_input("Ask a question about the PDFs:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        try:
            answer = qa_chain.run(query)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {e}")
