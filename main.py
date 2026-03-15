import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5

DOCUMENT_LIST = [
    ("Companies Act 2013", "Corporate governance and company formation"),
    ("CGST Act 2017", "Goods and Services Tax legislation"),
    ("CGST Rules 2017", "GST compliance procedures and filing rules"),
    ("Code on Wages 2019", "Labour and wage regulations"),
    ("Startup India Action Plan", "Government initiatives supporting startups"),
]

SAMPLE_QUERIES = [
    "What documents are required to register a private limited company?",
    "What is the GST registration threshold for businesses?",
    "What penalties apply for late GST filing?",
    "What are the minimum wage requirements under the Code on Wages?",
    "What benefits are provided under the Startup India initiative?",
]

# --- Page config ---
st.set_page_config(
    page_title="Legal Compliance Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Early error checks ---
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()

if not os.path.exists(FAISS_INDEX_PATH):
    st.error("FAISS index not found. Run ingestion first.")
    st.stop()

# --- Cached resource loading ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource(show_spinner="Loading legal document index...")
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )

try:
    vectorstore = load_vectorstore()
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()


# --- FALLBACK RAG FUNCTION ---
def generate_rag_answer(query: str):

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=TOP_K)

    retrieved_docs = []
    relevant_docs = []

    for doc, score in docs_with_scores:

        retrieved_docs.append(doc)

        # Lower score = better match
        if score < SIMILARITY_THRESHOLD:
            relevant_docs.append(doc)

    system_prompt = (
        "You are an expert legal compliance assistant for Indian startups and small businesses. "
        "Answer using provided legal document context when available."
    )

    # -------- RAG CASE --------
    if relevant_docs:

        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        user_prompt = (
            f"Context from official legal documents:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a structured answer using only the context."
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        return response.content, retrieved_docs

    # -------- FALLBACK CASE --------
    else:

        fallback_prompt = (
            f"No relevant legal documents were found.\n\n"
            f"Question: {query}\n\n"
            "Answer using your general legal knowledge."
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=fallback_prompt),
        ])

        return response.content, retrieved_docs


# --- Source citations renderer ---
def render_sources(docs):
    if not docs:
        return
    with st.expander(f"Sources — {len(docs)} document chunks retrieved", expanded=False):
        for i, doc in enumerate(docs):
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = doc.metadata.get("page", "N/A")
            st.markdown(f"**{i + 1}. {source_file}  |  Page {page_num}**")
            preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                preview += "..."
            st.caption(preview)
            if i < len(docs) - 1:
                st.divider()


# --- Sidebar ---
with st.sidebar:
    st.title("⚖️ Legal Assistant")
    st.markdown("---")

    st.subheader("About")
    st.markdown(
        "This assistant answers startup and business compliance questions "
        "using official Indian legal documents."
    )
    st.markdown("---")

    st.subheader("Document Library")
    for name, desc in DOCUMENT_LIST:
        st.markdown(f"**{name}**  \n{desc}")

    st.markdown("---")

    st.subheader("Sample Queries")
    st.caption("Click any question to ask it:")

    for query in SAMPLE_QUERIES:
        if st.button(query, use_container_width=True):
            st.session_state["prefill_query"] = query

    st.markdown("---")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# --- Main chat UI ---
st.title("AI-Powered Startup Legal Compliance Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "prefill_query" not in st.session_state:
    st.session_state["prefill_query"] = ""

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])

prefill = st.session_state.pop("prefill_query", "")

prompt = st.chat_input("Ask a legal compliance question...")

active_prompt = prompt or (prefill if prefill else None)

if active_prompt:

    st.session_state["messages"].append({"role": "user", "content": active_prompt})

    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching legal documents..."):

            try:
                answer, source_docs = generate_rag_answer(active_prompt)
            except RuntimeError as e:
                st.error(str(e))
                answer = None
                source_docs = []

        if answer:
            st.markdown(answer)
            render_sources(source_docs)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": source_docs,
            })