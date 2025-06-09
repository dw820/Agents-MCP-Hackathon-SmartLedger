"""
Optimized Modal functions with better container lifecycle management
"""

import modal
from typing import Dict, Any
from datetime import datetime
import requests
import base64
import json
import os

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

# Vision model configuration (now using Hyperbolic endpoint)
# Inject HYPERBOLIC_API_KEY as a Modal secret for all relevant functions
HYPERBOLIC_API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
HYPERBOLIC_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Global variables to cache models
_embedding_model = None
_tokenizer = None
_llm_model = None

def _load_models():
    """Load models once and cache globally"""
    global _embedding_model, _tokenizer, _llm_model
    
    if _embedding_model is None:
        print("🚀 Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
    
    if _tokenizer is None or _llm_model is None:
        print("🚀 Loading LLM model...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        _tokenizer.pad_token = _tokenizer.eos_token
        _llm_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        print("✅ LLM model loaded")
    
    return _embedding_model, _tokenizer, _llm_model

@app.function(
    image=model_image,
    memory=4096,
    timeout=1800,  # 30 minutes timeout
    min_containers=1  # Keep one container warm
)
def create_index(csv_data: str, session_id: str) -> Dict[str, Any]:
    """Create index using cached models"""
    try:
        import pandas as pd
        from io import StringIO
        
        # Load models
        embedding_model, tokenizer, llm_model = _load_models()
        
        print(f"📊 Creating index for session: {session_id}")
        
        # Parse CSV
        df = pd.read_csv(StringIO(csv_data))
        
        # Create embeddings
        doc_texts = []
        doc_metadata = []
        
        for _, row in df.iterrows():
            doc_text = f"Date: {row.get('date', '')} Amount: ${row.get('amount', 0)} Description: {row.get('description', '')} Category: {row.get('category', '')}"
            doc_texts.append(doc_text)
            doc_metadata.append(row.to_dict())
        
        # Generate embeddings
        print(f"🔗 Generating embeddings for {len(doc_texts)} transactions...")
        embeddings = embedding_model.encode(doc_texts)
        
        # Store in persistent storage
        index_data = {
            "embeddings": embeddings.tolist(),
            "doc_texts": doc_texts,
            "doc_metadata": doc_metadata,
            "total_transactions": len(doc_texts),
            "created_at": datetime.now().isoformat()
        }
        
        session_storage[session_id] = index_data
        session_storage[f"session_active_{session_id}"] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        print(f"✅ Index created for {len(doc_texts)} transactions")
        
        return {
            "status": "success",
            "session_id": session_id,
            "total_transactions": len(doc_texts),
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return {"status": "error", "error": str(e)}

@app.function(
    image=model_image,
    memory=4096,
    timeout=600,
    min_containers=1  # Keep one container warm
)
def query_data(query: str, session_id: str) -> Dict[str, Any]:
    """Query using cached models"""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        import torch
        
        # Load models
        embedding_model, tokenizer, llm_model = _load_models()
        
        print(f"🔍 Processing query: '{query}' for session: {session_id}")
        
        # Get session data
        if session_id not in session_storage:
            return {"status": "error", "error": "Session not found"}
        
        session_data = session_storage[session_id]
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        
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
        
        # Generate LLM analysis
        llm_analysis = ""
        if matching_transactions:
            context = f"Financial Query: {query}\n\nRelevant Transactions:\n{chr(10).join(relevant_context[:3])}\n\nAnalysis:"
            
            inputs = tokenizer.encode(context, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = llm_model.generate(
                    inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return {"status": "error", "error": str(e)}

@app.function(image=model_image)
def check_health() -> Dict[str, Any]:
    """Quick health check"""
    try:
        # Load models
        embedding_model, tokenizer, llm_model = _load_models()
        
        # Test embedding model
        _ = embedding_model.encode(["test transaction"])
        
        # Test LLM
        _ = tokenizer.encode("Test", return_tensors="pt")
        
        return {
            "status": "healthy",
            "embedding_model": "ready",
            "llm_model": "ready", 
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.function(
    image=model_image,
    memory=2048,
    timeout=300,
    secrets=[modal.Secret.from_name("HYPERBOLIC_API_KEY")],
)
def process_image_transactions(image_data, session_id: str, filename: str = "uploaded_image") -> Dict[str, Any]:
    """
    Process image to extract transaction data using Llama 3.2 Vision Instruct
    
    Args:
        image_data: File object or base64 encoded string
        session_id: Session ID for storing extracted transactions
        filename: Original filename for reference
        
    Returns:
        Dictionary containing extracted transactions and metadata
    """
    try:
        print(f"📷 Processing image for session: {session_id}")
        print(f"🔍 Debug - image_data type: {type(image_data)}")
        print(f"🔍 Debug - image_data dir: {dir(image_data)}")
        print(f"🔍 Debug - hasattr read: {hasattr(image_data, 'read')}")
        
        # Handle file object vs base64 string
        if hasattr(image_data, 'read'):
            # It's a file object, read and encode
            print("📖 Reading file object...")
            image_bytes = image_data.read()
            print(f"📏 Read {len(image_bytes)} bytes")
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            print(f"🔤 Encoded to base64 length: {len(image_b64)}")
        elif isinstance(image_data, str):
            # It's already base64 encoded
            print("📝 Using string as base64...")
            image_b64 = image_data
        else:
            # It's raw bytes
            print(f"🔢 Converting bytes to base64... type: {type(image_data)}")
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create optimized prompt for transaction extraction
        prompt = """You are a financial document analyzer. Extract transaction data from this bank statement, receipt, or financial document.

For each transaction you find, return ONLY a valid JSON array with this exact format:

[
  {
    "date": "YYYY-MM-DD",
    "amount": 123.45,
    "vendor": "Vendor Name",
    "description": "Transaction description",
    "type": "debit"
  }
]

Rules:
- Extract ALL transactions visible in the image
- Use negative amounts for debits/expenses, positive for credits/income  
- Parse dates to YYYY-MM-DD format
- Clean vendor names (remove extra spaces, standardize)
- Include meaningful descriptions
- Return ONLY the JSON array, no other text
- If no transactions found, return: []

Analyze the image and extract all transaction data:"""
        
        # Call Hyperbolic vision model endpoint
        hyperbolic_api_key = os.environ["HYPERBOLIC_API_KEY"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hyperbolic_api_key}",
        }
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }],
            "model": HYPERBOLIC_MODEL_NAME,
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.001,
        }
        print("🧠 Calling Hyperbolic Vision model...")
        print(f"Hyperbolic API URL: {HYPERBOLIC_API_URL}")
        response = requests.post(
            HYPERBOLIC_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        if response.status_code != 200:
            return {
                "status": "error",
                "error": f"Vision model API failed: {response.status_code} - {response.text}",
                "transactions": [],
                "total_transactions": 0
            }
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        # Parse JSON response
        try:
            # Find JSON array in response
            response_text = response_text.strip()
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"⚠️ No JSON array found in response")
                transactions = []
            else:
                json_str = response_text[start_idx:end_idx]
                transactions = json.loads(json_str)
                
                # Clean and validate transactions
                cleaned_transactions = []
                for txn in transactions:
                    if isinstance(txn, dict) and "amount" in txn and "vendor" in txn:
                        try:
                            txn["amount"] = float(txn["amount"])
                            txn["source"] = "image_extraction"
                            txn["extracted_at"] = datetime.now().isoformat()
                            cleaned_transactions.append(txn)
                        except (ValueError, TypeError):
                            print(f"⚠️ Invalid transaction amount: {txn}")
                
                transactions = cleaned_transactions
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            transactions = []
        
        # Store extracted transactions for this session
        image_session_key = f"{session_id}_image_transactions"
        session_storage[image_session_key] = {
            "transactions": transactions,
            "total_transactions": len(transactions),
            "filename": filename,
            "extracted_at": datetime.now().isoformat(),
            "raw_response": response_text[:500] if len(response_text) > 500 else response_text
        }
        
        print(f"✅ Extracted {len(transactions)} transactions from image")
        
        return {
            "status": "success",
            "transactions": transactions,
            "total_transactions": len(transactions),
            "filename": filename,
            "extracted_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return {
            "status": "error",
            "error": str(e),
            "transactions": [],
            "total_transactions": 0,
            "filename": filename,
            "extracted_at": datetime.now().isoformat()
        }

@app.function(
    image=model_image,
    memory=2048,
    timeout=300
)
def reconcile_transactions(session_id: str) -> Dict[str, Any]:
    """
    Reconcile image-extracted transactions with CSV ledger entries
    
    Args:
        session_id: Session ID containing both CSV and image transaction data
        
    Returns:
        Dictionary containing matches, unmatched transactions, and confidence scores
    """
    try:
        print(f"🔄 Reconciling transactions for session: {session_id}")
        
        # Get CSV transactions
        if session_id not in session_storage:
            return {
                "status": "error",
                "error": "No CSV data found for session. Please upload and analyze CSV first."
            }
        
        csv_data = session_storage[session_id]
        csv_transactions = csv_data.get("doc_metadata", [])
        
        # Get image transactions
        image_session_key = f"{session_id}_image_transactions"
        if image_session_key not in session_storage:
            return {
                "status": "error", 
                "error": "No image data found for session. Please upload and process an image first."
            }
        
        image_data = session_storage[image_session_key]
        image_transactions = image_data.get("transactions", [])
        
        if not csv_transactions or not image_transactions:
            return {
                "status": "error",
                "error": f"Insufficient data: {len(csv_transactions)} CSV transactions, {len(image_transactions)} image transactions"
            }
        
        print(f"Reconciling {len(image_transactions)} image transactions with {len(csv_transactions)} CSV transactions")
        
        # Simple reconciliation algorithm
        matches = []
        matched_csv_indices = set()
        matched_image_indices = set()
        
        for i, image_txn in enumerate(image_transactions):
            best_match = None
            best_score = 0
            best_csv_idx = -1
            
            for j, csv_txn in enumerate(csv_transactions):
                if j in matched_csv_indices:
                    continue
                
                score = 0
                reasons = []
                discrepancies = []
                
                # Amount matching (40% weight)
                try:
                    image_amount = float(image_txn.get("amount", 0))
                    csv_amount = float(csv_txn.get("amount", 0))
                    amount_diff = abs(image_amount - csv_amount)
                    
                    if amount_diff <= 0.01:  # Exact match
                        score += 0.4
                        reasons.append(f"Exact amount match: ${csv_amount}")
                    elif amount_diff <= max(0.01, abs(csv_amount) * 0.02):  # 2% tolerance
                        score += 0.35
                        reasons.append(f"Close amount match: ${csv_amount} ≈ ${image_amount}")
                    elif amount_diff <= abs(csv_amount) * 0.05:  # 5% tolerance
                        score += 0.2
                        reasons.append(f"Approximate amount match: ${csv_amount} ≈ ${image_amount}")
                    else:
                        discrepancies.append(f"Amount difference: ${csv_amount} vs ${image_amount}")
                        
                except:
                    discrepancies.append("Invalid amount data")
                
                # Date matching (25% weight) - simplified
                try:
                    from datetime import datetime
                    
                    csv_date_str = str(csv_txn.get("date", ""))
                    image_date_str = str(image_txn.get("date", ""))
                    
                    if csv_date_str[:10] == image_date_str[:10]:  # Same date (YYYY-MM-DD)
                        score += 0.25
                        reasons.append(f"Exact date match: {csv_date_str[:10]}")
                    elif abs(len(csv_date_str) - len(image_date_str)) <= 3:  # Close dates
                        score += 0.15
                        reasons.append(f"Similar dates: {csv_date_str[:10]} ≈ {image_date_str[:10]}")
                    else:
                        discrepancies.append(f"Date difference: {csv_date_str[:10]} vs {image_date_str[:10]}")
                        
                except:
                    discrepancies.append("Invalid date data")
                
                # Vendor matching (25% weight) - simplified
                try:
                    csv_vendor = str(csv_txn.get("vendor", "")).upper().strip()
                    image_vendor = str(image_txn.get("vendor", "")).upper().strip()
                    
                    if csv_vendor == image_vendor:
                        score += 0.25
                        reasons.append(f"Exact vendor match: {csv_vendor}")
                    elif csv_vendor in image_vendor or image_vendor in csv_vendor:
                        score += 0.2
                        reasons.append(f"Partial vendor match: {csv_vendor} ≈ {image_vendor}")
                    elif len(set(csv_vendor.split()).intersection(set(image_vendor.split()))) > 0:
                        score += 0.1
                        reasons.append(f"Vendor keyword match: {csv_vendor} ≈ {image_vendor}")
                    else:
                        discrepancies.append(f"Vendor difference: {csv_vendor} vs {image_vendor}")
                        
                except:
                    discrepancies.append("Invalid vendor data")
                
                # Description matching (10% weight) - simplified
                try:
                    csv_desc = str(csv_txn.get("description", "")).upper().strip()
                    image_desc = str(image_txn.get("description", "")).upper().strip()
                    
                    if csv_desc and image_desc:
                        if csv_desc == image_desc:
                            score += 0.1
                            reasons.append(f"Description match: {csv_desc}")
                        elif csv_desc in image_desc or image_desc in csv_desc:
                            score += 0.05
                            reasons.append(f"Partial description match")
                        
                except:
                    pass
                
                # Store best match for this image transaction
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_match = {
                        "csv_transaction": csv_txn,
                        "image_transaction": image_txn,
                        "confidence_score": round(score, 3),
                        "match_reasons": reasons,
                        "discrepancies": discrepancies,
                        "match_type": "high" if score >= 0.85 else "medium" if score >= 0.65 else "low"
                    }
                    best_csv_idx = j
            
            # Add best match if found
            if best_match:
                matches.append(best_match)
                matched_csv_indices.add(best_csv_idx)
                matched_image_indices.add(i)
        
        # Calculate unmatched transactions
        unmatched_csv = [csv_transactions[i] for i in range(len(csv_transactions)) if i not in matched_csv_indices]
        unmatched_image = [image_transactions[i] for i in range(len(image_transactions)) if i not in matched_image_indices]
        
        # Categorize matches
        high_confidence = [m for m in matches if m["confidence_score"] >= 0.85]
        medium_confidence = [m for m in matches if 0.65 <= m["confidence_score"] < 0.85]
        low_confidence = [m for m in matches if m["confidence_score"] < 0.65]
        
        # Calculate summary
        match_rate = len(matches) / len(image_transactions) if image_transactions else 0
        total_image_amount = sum(float(txn.get("amount", 0)) for txn in image_transactions)
        total_matched_amount = sum(float(m["image_transaction"].get("amount", 0)) for m in matches)
        
        result = {
            "status": "success",
            "summary": {
                "total_image_transactions": len(image_transactions),
                "total_csv_transactions": len(csv_transactions),
                "total_matches": len(matches),
                "match_rate": round(match_rate * 100, 1),
                "high_confidence_matches": len(high_confidence),
                "medium_confidence_matches": len(medium_confidence),
                "low_confidence_matches": len(low_confidence),
                "unmatched_image_transactions": len(unmatched_image),
                "unmatched_csv_transactions": len(unmatched_csv),
                "total_image_amount": round(total_image_amount, 2),
                "total_matched_amount": round(total_matched_amount, 2),
                "reconciliation_percentage": round((total_matched_amount / total_image_amount * 100) if total_image_amount else 0, 1)
            },
            "matches": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence
            },
            "unmatched": {
                "image_transactions": unmatched_image,
                "csv_transactions": unmatched_csv
            },
            "reconciled_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        # Store reconciliation results
        reconciliation_key = f"{session_id}_reconciliation"
        session_storage[reconciliation_key] = result
        
        print(f"✅ Reconciliation complete: {len(matches)} matches found ({match_rate*100:.1f}% match rate)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during reconciliation: {e}")
        return {
            "status": "error",
            "error": str(e),
            "summary": {},
            "matches": {"high_confidence": [], "medium_confidence": [], "low_confidence": []},
            "unmatched": {"image_transactions": [], "csv_transactions": []},
            "reconciled_at": datetime.now().isoformat()
        }

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
    print("SmartLedger Modal functions with image processing and reconciliation")
    print("Features: CSV indexing, image processing, transaction reconciliation")
    print("UI calls: modal.Function.lookup('smartledger', 'create_index')")