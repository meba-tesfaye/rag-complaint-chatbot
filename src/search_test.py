import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def run_verification_test(query_text, k=3):
    print("🔬 Running Vector Store Verification Test...")
    
    # Define local vector store directory path
    vector_store_dir = "vector_store"
    index_path = os.path.join(vector_store_dir, "faiss_index")
    
    if not os.path.exists(index_path):
        print(f"❌ Error: Could not find FAISS index files at {index_path}.")
        print("Please ensure you run 'python src/indexing.py' first to build the index.")
        return

    # Initialize the same embedding model used during index compilation
    print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the persistent FAISS index safely
    print("📥 Loading local FAISS index from disk...")
    try:
        vector_store = FAISS.load_local(
            index_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("✅ FAISS index loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return
        
    # Perform similarity search
    print(f"🔍 Performing semantic similarity search for query: '{query_text}'...\n")
    results = vector_store.similarity_search_with_score(query_text, k=k)
    
    print(f"🎯 Retrieved Top {k} Matching Results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Product Class: {doc.metadata.get('product')}")
        print(f"    Complaint ID: {doc.metadata.get('complaint_id')}")
        print(f"    Similarity Distance Score: {score:.4f}")
        print(f"    Content Excerpt:\n    \"{doc.page_content[:250]}...\"")
        print("-" * 50)

if __name__ == "__main__":
    # Test query designed to hit credit card or banking categories
    test_query = "unauthorized fees charged to account without my signature or consent"
    run_verification_test(test_query, k=2)