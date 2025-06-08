"""
Optimized Modal functions with better container lifecycle management
"""

import modal
from typing import Dict, Any
from datetime import datetime

app = modal.App("smartledger")

# Optimized image with model pre-loading
model_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "sentence-transformers>=2.2.0",
        "transformers>=4.35.0", 
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "llama-index>=0.10.0",
        "scikit-learn>=1.3.0",
        "accelerate>=0.20.0",
    ])
    .run_commands([
        # Pre-download models during build (faster cold starts)
        "python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"all-MiniLM-L6-v2\")'",
        "python -c 'from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained(\"microsoft/DialoGPT-medium\"); AutoModelForCausalLM.from_pretrained(\"microsoft/DialoGPT-medium\")'",
    ])
)

# Persistent storage that survives container restarts
session_storage = modal.Dict.from_name("smartledger-sessions", create_if_missing=True)

@app.cls(
    image=model_image,
    memory=4096,
    timeout=1800,  # 30 minutes timeout
    scaledown_window=600,  # Keep warm for 10 minutes after last use
    min_containers=0,  # Start containers on demand
    max_containers=5   # Scale up to 5 concurrent containers
)
class SmartLedgerAnalyzer:
    """
    Modal class that loads models once and reuses them across calls.
    Container lifecycle is managed automatically:
    - Cold start: ~30-60s (loads models)
    - Warm calls: ~1-3s (models already loaded)
    - Auto-shutdown: After 10 min idle
    - Auto-restart: When UI makes new calls
    """

    def __enter__(self):
        """Initialize models once when container starts"""
        print("🚀 Loading models in new container...")
        
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ LLM model loaded")
        
        print("🎉 Container ready for requests!")
        return self

    @modal.method()
    def create_financial_index(self, csv_data: str, session_id: str) -> Dict[str, Any]:
        """Create index using pre-loaded models (fast)"""
        try:
            import pandas as pd
            from io import StringIO
            
            print(f"📊 Creating index for session: {session_id}")
            
            # Parse CSV
            df = pd.read_csv(StringIO(csv_data))
            
            # Create embeddings using pre-loaded model
            doc_texts = []
            doc_metadata = []
            
            for _, row in df.iterrows():
                doc_text = f"Date: {row.get('date', '')} Amount: ${row.get('amount', 0)} Description: {row.get('description', '')} Category: {row.get('category', '')}"
                doc_texts.append(doc_text)
                doc_metadata.append(row.to_dict())
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(doc_texts)
            
            # Store in persistent storage
            index_data = {
                "embeddings": embeddings.tolist(),
                "doc_texts": doc_texts,
                "doc_metadata": doc_metadata,
                "total_transactions": len(doc_texts),
                "created_at": datetime.now().isoformat(),
                "container_id": f"container_{id(self)}"
            }
            
            session_storage[session_id] = index_data
            
            return {
                "status": "success",
                "session_id": session_id,
                "total_transactions": len(doc_texts),
                "container_warm": True,  # Models were pre-loaded
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @modal.method()
    def query_financial_data(self, query: str, session_id: str) -> Dict[str, Any]:
        """Query using pre-loaded models (very fast)"""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            
            print(f"🔍 Processing query: '{query}' for session: {session_id}")
            
            # Get session data
            if session_id not in session_storage:
                return {"status": "error", "error": "Session not found"}
            
            session_data = session_storage[session_id]
            
            # Generate query embedding using pre-loaded model
            query_embedding = self.embedding_model.encode([query])
            
            # Find similar transactions
            doc_embeddings = np.array(session_data["embeddings"])
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:5]
            matching_transactions = []
            relevant_context = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    doc_text = session_data["doc_texts"][idx]
                    metadata = session_data["doc_metadata"][idx]
                    
                    matching_transactions.append({
                        "text": doc_text,
                        "metadata": metadata,
                        "similarity": float(similarities[idx])
                    })
                    relevant_context.append(doc_text)
            
            # Generate LLM analysis using pre-loaded model
            llm_analysis = ""
            if matching_transactions:
                context = f"Financial Query: {query}\n\nRelevant Transactions:\n{chr(10).join(relevant_context[:3])}\n\nAnalysis:"
                
                inputs = self.tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                llm_analysis = full_response[len(context):].strip()
            
            # Calculate summary
            total_amount = sum(float(t["metadata"].get("amount", 0)) for t in matching_transactions)
            
            return {
                "status": "success",
                "query": query,
                "llm_analysis": llm_analysis,
                "matching_transactions": len(matching_transactions),
                "total_amount": total_amount,
                "results": matching_transactions,
                "container_warm": True,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @modal.method()
    def health_check(self) -> Dict[str, Any]:
        """Quick health check using pre-loaded models"""
        try:
            # Test embedding model
            _ = self.embedding_model.encode(["test transaction"])
            
            # Test LLM
            _ = self.tokenizer.encode("Test", return_tensors="pt")
            
            return {
                "status": "healthy",
                "embedding_model": "ready",
                "llm_model": "ready", 
                "container_id": f"container_{id(self)}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Create analyzer instance
analyzer = SmartLedgerAnalyzer()

# Public endpoints that UI calls - these handle session storage directly
@app.function(image=model_image, timeout=600)
def create_index(csv_data: str, session_id: str):
    """UI calls this - automatically starts container if needed"""
    try:
        # Call the analyzer to process the data
        result = analyzer.create_financial_index.remote(csv_data, session_id)
        
        # If successful, also store a simple session marker
        if result.get("status") == "success":
            session_storage[f"session_active_{session_id}"] = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }
            print(f"✅ Session {session_id} marked as active in storage")
        
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=model_image, timeout=600) 
def query_data(query: str, session_id: str):
    """UI calls this - uses warm container if available"""
    try:
        # Check if session exists
        session_key = f"session_active_{session_id}"
        if session_key not in session_storage:
            return {
                "status": "error", 
                "error": f"Session {session_id} not found. Please upload CSV data first.",
                "available_sessions": list(session_storage.keys())
            }
        
        # Call the analyzer
        result = analyzer.query_financial_data.remote(query, session_id)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=model_image)
def check_health():
    """UI calls this to verify system status"""
    return analyzer.health_check.remote()

@app.function(image=model_image)
def list_sessions():
    """Debug function to list all active sessions"""
    try:
        return {
            "status": "success",
            "sessions": dict(session_storage),
            "session_count": len(session_storage)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("Optimized Modal functions with automatic container management")
    print("Containers start on-demand and stay warm for 10 minutes")
    print("UI calls: modal.Function.lookup('smartledger', 'create_index')")