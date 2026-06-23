import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Attempt to load Google GenAI SDK (with fallback compatibility)
try:
    from google import genai
    from google.genai import types
    NEW_SDK_AVAILABLE = True
except ImportError:
    import google.generativeai as genai
    NEW_SDK_AVAILABLE = False

# ==========================================
# 🎨 STREAMLIT PAGE LAYOUT & CONFIG
# ==========================================
st.set_page_config(
    page_title="CFPB Consumer Complaint RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .source-block {
        border-left: 3px solid #10b981;
        background-color: #f0fdf4;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 REUSABLE CORE RAG FUNCTIONS
# ==========================================
@st.cache_resource(show_spinner="🧠 Loading Semantic Embedding Model...")
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="📥 Indexing persistent FAISS Vector Store...")
def get_vector_store(_embedding_model):
    vector_store_dir = "vector_store"
    index_path = os.path.join(vector_store_dir, "faiss_index")
    if not os.path.exists(index_path):
        st.error("❌ FAISS Index not found.")
        st.stop()
    return FAISS.load_local(index_path, _embedding_model, allow_dangerous_deserialization=True)

embedding_model = get_embedding_model()
vector_store = get_vector_store(embedding_model)

# ==========================================
# 🛠️ SIDEBAR - SYSTEM PARAMETERS & KEYS
# ==========================================
st.sidebar.title("🛡️ System Configurations")

# Clear Conversation Button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

api_key_input = st.sidebar.text_input("🔑 Gemini API Key", type="password")
active_api_key = api_key_input if api_key_input else os.environ.get("GEMINI_API_KEY", "")

# Retrieval Settings
top_k = st.sidebar.slider("Top-k Similarity Chunks", 1, 8, 4)
similarity_threshold = st.sidebar.slider("Max Distance Threshold (L2)", 0.0, 2.0, 1.5)

# ==========================================
# 💬 INTERACTIVE CHAT PORTAL
# ==========================================
st.title("🏛️ RAG-Based Consumer Complaint Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 Retrieved Context & Source Verification"):
                for src in msg["sources"]:
                    st.markdown(f"<div class='source-block'><strong>ID:</strong> {src['complaint_id']} | <strong>Score:</strong> {src['score']:.4f}<br/><em>{src['text']}</em></div>", unsafe_allow_html=True)

user_query = st.chat_input("Ask about financial disputes...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("🔍 Retrieving..."):
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(user_query, k=top_k)
        valid_sources = [{"product": d.metadata.get("product"), "complaint_id": d.metadata.get("complaint_id"), "score": s, "text": d.page_content} for d, s in retrieved_docs_with_scores if s <= similarity_threshold]
        context_blocks = [s['text'] for s in valid_sources]
        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    with st.chat_message("assistant"):
        response_container = st.empty()
        try:
            if NEW_SDK_AVAILABLE:
                client = genai.Client(api_key=active_api_key)
                response = client.models.generate_content(model="gemini-2.5-flash", contents=f"Context: {context_str}\nQuery: {user_query}")
                answer = response.text
            else:
                genai.configure(api_key=active_api_key)
                answer = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Context: {context_str}\nQuery: {user_query}").text
            
            response_container.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")
            answer = "Generation failed."
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": valid_sources})