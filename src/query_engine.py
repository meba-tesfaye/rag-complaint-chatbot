import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from google import genai
    from google.genai import types
    NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    NEW_SDK = False

class ComplaintRAGEngine:
    def __init__(self, vector_store_dir="vector_store", model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        
        print("🧠 Loading semantic embedding engine...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        index_path = os.path.join(vector_store_dir, "faiss_index")
        print(f"📥 Loading FAISS index store from {index_path}...")
        try:
            self.vector_store = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ Vector index loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load index database: {e}")
            sys.exit(1)

    def query(self, user_query, k=3, threshold=1.5):
        """Executes a fully grounded retrieval-augmented generation loop."""
        results = self.vector_store.similarity_search_with_score(user_query, k=k)
        
        valid_contexts = []
        sources = []
        for doc, score in results:
            if score <= threshold:
                valid_contexts.append(doc.page_content)
                sources.append({
                    "complaint_id": doc.metadata.get("complaint_id"),
                    "product": doc.metadata.get("product"),
                    "score": score
                })
        
        context_str = "\n\n---\n\n".join(valid_contexts) if valid_contexts else "No relevant context found."
        
        system_instruction = (
            "You are a compliant financial services RAG assistant.\n"
            "Ground your answer STRICTLY in the provided historical consumer complaints context.\n"
            "If the context does not contain enough evidence, output: "
            "'I cannot find the answer in the retrieved historical complaints.'"
        )
        
        prompt_content = f"""
Context:
{context_str}

Query:
{user_query}
"""
        
        if not self.api_key:
            return "⚠️ Generation bypassed: GEMINI_API_KEY environment variable is missing.", sources

        try:
            if NEW_SDK:
                client = genai.Client(api_key=self.api_key)
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_content,
                    config=config
                )
                answer = response.text
            else:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_instruction,
                    generation_config={"temperature": 0.1}
                )
                response = model.generate_content(prompt_content)
                answer = response.text
                
            return answer, sources
        except Exception as e:
            return f"❌ Error during generation layer execution: {e}", sources

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = "unauthorized fees on my credit card without my authorization"
        
    engine = ComplaintRAGEngine()
    print(f"\n🔍 Querying: '{query_text}'")
    answer, sources = engine.query(query_text)
    
    print("\n💬 Grounded RAG Response:")
    print(answer)
    print("\n🎯 Audited Precedent Nodes:")
    for src in sources:
        print(f" - ID: {src['complaint_id']} | Product: {src['product']} | Score (L2): {src['score']:.4f}")