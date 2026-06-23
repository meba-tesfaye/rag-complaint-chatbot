import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# STREAMING_CHUNK: Setting up SDK imports and robust fallbacks...
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

# STREAMING_CHUNK: Injecting custom CSS styles for clean visual cards...
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
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
@st.cache_resource(show_spinner="🧠 Loading Semantic Embedding Model (all-MiniLM-L6-v2)...")
def get_embedding_model():
    """Caches the embedding model to avoid reloading on every interaction."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STREAMING_CHUNK: Caching FAISS Vector Index safely...
@st.cache_resource(show_spinner="📥 Indexing persistent FAISS Vector Store...")
def get_vector_store(_embedding_model):
    """Loads and caches the persistent FAISS vector index."""
    vector_store_dir = "vector_store"
    index_path = os.path.join(vector_store_dir, "faiss_index")
    if not os.path.exists(index_path):
        st.error(f"❌ FAISS Index not found at `{index_path}`. Please run `python src/indexing.py` first to compile database.")
        st.stop()
    return FAISS.load_local(index_path, _embedding_model, allow_dangerous_deserialization=True)

# Initialize models
embedding_model = get_embedding_model()
vector_store = get_vector_store(embedding_model)

# ==========================================
# 🛠️ SIDEBAR - SYSTEM PARAMETERS & KEYS
# ==========================================
st.sidebar.title("🛡️ System Configurations")

# STREAMING_CHUNK: Constructing secure input fields...
env_api_key = os.environ.get("GEMINI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "🔑 Gemini API Key", 
    value=env_api_key if env_api_key != "your_actual_api_key_here" else "", 
    type="password",
    help="Provide your Google Gemini API key. If empty, the system looks for the GEMINI_API_KEY environment variable."
)

active_api_key = api_key_input if api_key_input else env_api_key

# Parameter configurations
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Retrieval Settings")
top_k = st.sidebar.slider("Top-k Similarity Chunks", min_value=1, max_value=8, value=4, step=1)
similarity_threshold = st.sidebar.slider("Max Distance Threshold (L2)", min_value=0.0, max_value=2.0, value=1.5, step=0.1, help="L2 distance metric. Lower values require closer matching.")

# STREAMING_CHUNK: Setting generation parameters...
st.sidebar.subheader("⚙️ LLM Generation Settings")
model_selection = st.sidebar.selectbox("Gemini Engine", ["gemini-2.5-flash", "gemini-1.5-flash"])
temperature = st.sidebar.slider("Generation Temp (Creativity)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

# Application Information Card
st.sidebar.markdown("---")
st.sidebar.info("""
**RAG System Overview:**
* **Corpus Density:** 12,000 complaints / 36,222 semantic nodes.
* **Embeddings:** Dense 384-dimensional space.
* **Validation Layer:** Enforces factual compliance.
""")

# ==========================================
# 💬 INTERACTIVE CHAT PORTAL
# ==========================================
st.title("🏛️ RAG-Based Consumer Complaint Chatbot")
st.markdown("Query the historical CFPB dispute database to draft factually grounded support resolutions.")

# Inform users if API Key is missing
if not active_api_key or active_api_key == "your_actual_api_key_here":
    st.warning("⚠️ Please provide a valid **Gemini API Key** in the sidebar to initiate generation.")

# STREAMING_CHUNK: Initializing user chat state...
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display prior chat records
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 Retrieved Context & Source Verification"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-block">
                        <strong>Product Class:</strong> {src['product']} | 
                        <strong>Complaint ID:</strong> {src['complaint_id']} | 
                        <strong>Distance Score:</strong> {src['score']:.4f}<br/>
                        <em>"{src['text']}"</em>
                    </div>
                    """, unsafe_allow_html=True)

# STREAMING_CHUNK: Capturing user message and running similarity...
user_query = st.chat_input("Ask about fee disputes, loan terms, unauthorized transfers, or bank disputes...")

if user_query:
    # 1. Show user message instantly
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # 2. Perform FAISS Retrieval Core
    with st.spinner("🔍 Retrieving semantically similar precedent cases..."):
        # Retrieve chunks with distance metrics
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(user_query, k=top_k)
        
        # Filter based on the chosen threshold
        valid_sources = []
        context_blocks = []
        for doc, score in retrieved_docs_with_scores:
            if score <= similarity_threshold:
                context_blocks.append(doc.page_content)
                valid_sources.append({
                    "product": doc.metadata.get("product", "Unknown"),
                    "complaint_id": doc.metadata.get("complaint_id", "Unknown"),
                    "score": score,
                    "text": doc.page_content
                })

    # STREAMING_CHUNK: Formatting grounded prompt template...
    if active_api_key and active_api_key != "your_actual_api_key_here":
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Combine Context
            if context_blocks:
                context_str = "\n\n---\n\n".join(context_blocks)
            else:
                context_str = "No semantically matching historical records found."

            # Construct Prompt Template
            system_instruction = (
                "You are an expert, highly empathetic, and objective financial services compliance assistant. "
                "Your objective is to help customer support teams draft professional and regulation-compliant responses to consumer grievances.\n\n"
                "CRITICAL INSTRUCTION FOR TRUTHFULNESS:\n"
                "1. Base your answer STRICTLY on the facts presented in the provided context (historical complaints).\n"
                "2. If the retrieved context does not provide sufficient, explicit evidence to answer the query, "
                "state EXACTLY: 'I cannot find the answer in the retrieved historical complaints.'\n"
                "3. Never hallucinate, extrapolate, or invent bank guidelines, specific policies, interest rates, or consumer resolutions."
            )
            
            user_prompt = f"""
### Context (Historical Precedents):
{context_str}

### Consumer Query:
{user_query}

Based on the precedents above, please draft an empathetic and compliant support resolution.
"""

            # STREAMING_CHUNK: Connecting to Google GenAI API layer...
            try:
                if NEW_SDK_AVAILABLE:
                    client = genai.Client(api_key=active_api_key)
                    config = types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=1024
                    )
                    response = client.models.generate_content(
                        model=model_selection,
                        contents=user_prompt,
                        config=config
                    )
                    answer = response.text
                else:
                    genai.configure(api_key=active_api_key)
                    model = genai.GenerativeModel(
                        model_name="gemini-1.5-flash",
                        system_instruction=system_instruction,
                        generation_config={"temperature": temperature}
                    )
                    response = model.generate_content(user_prompt)
                    answer = response.text

                # Render response cleanly
                response_container.write(answer)
                
            except Exception as e:
                st.error(f"Generation Engine Error: {e}")
                answer = f"Error during synthesis: {e}"
                response_container.write(answer)

            # Store generation metadata in chat log
            msg_payload = {
                "role": "assistant",
                "content": answer,
                "sources": valid_sources
            }
            st.session_state.chat_history.append(msg_payload)
            
            # STREAMING_CHUNK: Displaying source verification metrics...
            if valid_sources:
                with st.expander("🔍 Retrieved Context & Source Verification"):
                    for src in valid_sources:
                        st.markdown(f"""
                        <div class="source-block">
                            <strong>Product Class:</strong> {src['product']} | 
                            <strong>Complaint ID:</strong> {src['complaint_id']} | 
                            <strong>Distance Score:</strong> {src['score']:.4f}<br/>
                            <em>"{src['text']}"</em>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("⚠️ Note: No historical sources matched below the distance threshold limit.")
    else:
        # Prompt user to configure key if offline
        st.info("📢 Precedents successfully searched! Provide your API key in the sidebar to generate a response.")
        if valid_sources:
            with st.expander("🔍 Checked Matching Precedents (Ground Truth Chunks)"):
                for src in valid_sources:
                    st.markdown(f"""
                    <div class="source-block">
                        <strong>Product:</strong> {src['product']} | ID: {src['complaint_id']} | Distance: {src['score']:.4f}<br/>
                        <em>"{src['text']}"</em>
                    </div>
                    """, unsafe_allow_html=True)