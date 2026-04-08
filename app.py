import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "startup_legal_compliance"
VECTORSTORE_CONFIG_PATH = os.path.join(CHROMA_DB_PATH, "rag_config.json")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSIONS = 768
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

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

st.set_page_config(
    page_title="Legal Compliance Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    st.error("GROQ_API_KEY not found or not set.")
    st.info(
        "Get a Groq API key, then add it to the `.env` file: "
        "GROQ_API_KEY=your_key_here"
    )
    st.stop()

if not os.path.exists(CHROMA_DB_PATH):
    st.error("Chroma DB not found. You need to build it first.")
    st.info("Run the ingestion script once before launching the app:")
    st.code("python rag_load_to_emb.py", language="bash")
    st.stop()

if not os.path.exists(VECTORSTORE_CONFIG_PATH):
    st.error("Chroma DB config not found. Please rebuild the database.")
    st.info("Run `python rag_load_to_emb.py` to rebuild the persisted Chroma store.")
    st.stop()

with open(VECTORSTORE_CONFIG_PATH, "r", encoding="utf-8") as config_file:
    vectorstore_config = json.load(config_file)

expected_config = {
    "collection_name": CHROMA_COLLECTION_NAME,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimensions": EMBEDDING_DIMENSIONS,
}

for key, expected_value in expected_config.items():
    if vectorstore_config.get(key) != expected_value:
        st.error("Chroma DB was built with a different embedding configuration.")
        st.info("Run `python rag_load_to_emb.py` to rebuild it for the current models.")
        st.stop()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    model_kwargs = {}
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading legal document index...")
def load_vectorstore():
    embeddings = load_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
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


def generate_rag_answer(query: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(query)

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are an expert legal compliance assistant for Indian startups and small businesses. "
        "You have deep knowledge of Indian corporate law, taxation, labour law, and startup regulations. "
        "Answer questions based on the provided legal document context. "
        "Be precise, cite specific sections or acts when possible, and structure your answers clearly. "
        "Use numbered lists or bullet points for multi-step processes. "
        "If the provided context does not contain enough information to answer fully, say so explicitly."
    )

    user_prompt = (
        f"Context from official legal documents:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a comprehensive, well-structured answer based on the context above."
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return response.content, docs
    except Exception as e:
        err = str(e)
        err_lower = err.lower()
        if "resource_exhausted" in err_lower or "429" in err_lower or "quota" in err_lower:
            raise RuntimeError(
                "Groq API quota exceeded for this key. Please wait and try again."
            ) from e
        if "api_key_invalid" in err_lower or "invalid api key" in err_lower or "400" in err_lower:
            raise RuntimeError(
                "Invalid API key. Please check your GROQ_API_KEY in the .env file."
            ) from e
        raise RuntimeError(f"Answer generation failed: {err[:300]}") from e


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


with st.sidebar:
    st.title("⚖️ Legal Assistant")
    st.markdown("---")

    st.subheader("About")
    st.markdown(
        "This assistant answers startup and business compliance questions "
        "using official Indian legal documents. All answers are grounded in the source PDFs below."
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

    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state["messages"] = []
        st.rerun()


st.title("AI-Powered Startup Legal Compliance Assistant")
st.caption(
    "Ask any question about company registration, GST, wages, or startup schemes. "
    "Answers are retrieved from official Indian legal documents."
)

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
prompt = st.chat_input("Ask a legal compliance question...", key="chat_input")
active_prompt = prompt or (prefill if prefill else None)

if active_prompt:
    st.session_state["messages"].append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents and generating answer..."):
            try:
                answer, source_docs = generate_rag_answer(active_prompt)
            except RuntimeError as e:
                st.error(str(e))
                answer = None
                source_docs = []

        if answer:
            st.markdown(answer)
            render_sources(source_docs)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_docs,
                }
            )
