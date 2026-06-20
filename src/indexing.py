import os
import pandas as pd
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_indexing_pipeline(processed_data_path, vector_store_dir):
    print("🚀 Starting Chunking, Embedding, and Indexing Pipeline...")
    
    # 1. Load Preprocessed Data
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data file not found at {processed_data_path}. Run preprocessing first.")
        
    df = pd.read_csv(processed_data_path)
    print(f"📥 Loaded {len(df):,} cleaned complaints.")
    
    # 2. Stratified Sampling (Targeting ~12,000 records)
    sample_size = 12000
    if len(df) > sample_size:
        print(f"⚖️ Performing stratified sampling to select {sample_size:,} records...")
        # Stratify by the 'Product' column to keep distributions identical
        _, df_sample = train_test_split(
            df, 
            test_size=sample_size / len(df), 
            stratify=df['Product'], 
            random_state=42
        )
    else:
        df_sample = df.copy()
        
    print("📊 Sample Product Distribution:")
    print(df_sample['Product'].value_counts())
    
    # 3. Setup Chunking Strategy
    # Using chunk_size=500 characters (~100 words) with 50 character overlap
    print("✂️ Splitting narratives into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chunks = []
    metadatas = []
    
    for _, row in df_sample.iterrows():
        narrative = str(row['cleaned_narrative'])
        doc_chunks = text_splitter.split_text(narrative)
        
        for chunk in doc_chunks:
            chunks.append(chunk)
            # Retain critical metadata trace for verification later
            metadatas.append({
                "complaint_id": str(row.get('Complaint ID', '')),
                "product": str(row.get('Product', ''))
            })
            
    print(f"🧩 Created {len(chunks):,} total text chunks from {len(df_sample):,} narratives.")
    
    # 4. Initialize Embedding Model
    print("🧠 Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 5. Generate Vector Store and Persist Local Index
    print("⚡ Generating embeddings and building FAISS vector store (this will take a few minutes)...")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embedding_model, metadatas=metadatas)
    
    print(f"💾 Saving FAISS index to disk directory: {vector_store_dir}")
    os.makedirs(vector_store_dir, exist_ok=True)
    vector_store.save_local(os.path.join(vector_store_dir, "faiss_index"))
    print("✅ Indexing pipeline completed successfully!")

if __name__ == "__main__":
    PROCESSED_INPUT = "data/processed/filtered_complaints.csv"
    VECTOR_STORE_DIR = "vector_store"
    
    run_indexing_pipeline(PROCESSED_INPUT, VECTOR_STORE_DIR)