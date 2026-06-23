🏛️ RAG-Based Consumer Complaint Chatbot

A highly grounded Retrieval-Augmented Generation (RAG) assistant designed to synthesize factually accurate financial support resolutions from historical Consumer Financial Protection Bureau (CFPB) data.

🚀 Overview

Modern financial support operations face high volumes of grievances. This system mitigates LLM hallucinations by injecting verified historical precedents directly into the generation context, ensuring that every chatbot response is grounded in real, audited data.

🛠️ Architecture

The system employs a multi-stage pipeline:

Preprocessing: Normalization and cleaning of ~700k public narratives.

Indexing: Stratified sampling into a 384-dimensional FAISS vector space.

Retrieval: L2-distance-based semantic search to identify the most relevant precedent chunks.

Generation: Secure synthesis using the Gemini-2.5-Flash model, strictly constrained by system instructions to avoid out-of-bounds information.

⚙️ Project Structure

app.py: The main interactive Streamlit web interface.

src/query_engine.py: The core RAG retrieval and generation engine.

src/: Data processing and indexing pipeline scripts.

vector_store/: Persistent FAISS index repository.

📋 Installation

Clone the repository and install dependencies:

pip install -r requirements.txt


Ensure you have your Google Gemini API Key ready.

🚀 How to Run

CLI Query Interface

python src/query_engine.py "your financial question here"


Interactive Web Application

streamlit run app.py


Open the local address provided by Streamlit in your browser, paste your API key into the sidebar, and start chatting.

🛡️ Trust & Auditability

Every resolution drafted by this system includes an Audit Panel that lists the exact historical complaints (Product Class, Complaint ID, and Similarity Score) used to derive the answer, ensuring total transparency and factual verification.