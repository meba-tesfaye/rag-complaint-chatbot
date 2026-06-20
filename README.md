RAG-Based Consumer Complaint Chatbot 🚀

An end-to-end Retrieval-Augmented Generation (RAG) system built to clean, vectorize, index, and query the Consumer Financial Protection Bureau (CFPB) data repository. This system uses semantic vector spaces to retrieve highly localized context from complex financial complaints.

📊 Pipeline Metrics & Analysis

1. Preprocessing & Exploratory Data Analysis (Task 1)

Grand Total Rows Examined: 9,609,797 raw complaints

Complaints WITHOUT Public Narratives: 6,629,041 rows (69%)

Complaints WITH Public Narratives: 2,980,756 rows (31%)

Target Category Isolation: Extracted 703,505 narrative-backed rows matching our targeted product verticals: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

2. Stratified Sampling & Vector Store compilation (Task 2)

Stratified Sample Size: 12,000 distinct complaint narratives.

Proportional Product Distribution (Sample):

🏦 Personal Loan: 4,440 records (37.0%)

💳 Credit Card: 3,230 records (26.9%)

💰 Savings Account: 2,647 records (22.1%)

💸 Money Transfer: 1,683 records (14.0%)

Text Chunking Approach: RecursiveCharacterTextSplitter configured at a chunk_size of 500 characters and a chunk_overlap of 50 characters to preserve structural paragraph continuity.

Total Nodes Extracted: 36,222 unique text chunks.

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 generating dense 384-dimensional vector embeddings.

Database Layer: Local persistent FAISS Vector Index bound with tracking metadata schema (complaint_id, product).

📂 Project Structure

rag-complaint-chatbot/
├── data/
│   ├── raw/                 # Contains the raw complaints.csv (Git Ignored)
│   └── processed/           # Contains cleaned filtered_complaints.csv (Git Ignored)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Memory-efficient streaming preprocessor & text sanitizer
│   ├── indexing.py          # Stratified sampler, recursive chunker, and FAISS build script
│   └── eda_metrics.py       # High-speed data streaming calculation module
├── vector_store/
│   └── faiss_index/         # Persisted local vector database binary indices
├── .gitignore               # Excludes virtual environments and multi-gigabyte datasets
├── README.md                # System engineering guide
└── requirements.txt         # Project dependencies


🛠️ Setup & Reproduction Instructions

1. Build Environment & Install Dependencies

Ensure you are in the project's root folder and have your Python virtual environment active:

# Set up virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt


2. Run the Pipelines In Sequence

Make sure your raw CFPB dataset is downloaded and saved inside data/raw/complaints.csv. Then, run the processing sequence:

# Run 1: Preprocess raw dataset and extract target records
python src/preprocessing.py

# Run 2: Verify raw dataset metrics
python src/eda_metrics.py

# Run 3: Sample, chunk, embed, and compile the FAISS index database
python src/indexing.py


🔬 Search Verification Test

To confirm that your FAISS store retrieves semantic context correctly, run a test search directly against your local index database using your search test module.

# Verify similarity queries against the generated vector index
python src/search_test.py