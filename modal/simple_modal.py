"""
Simplified Modal functions that work without heavy model dependencies
"""

import modal
from typing import Dict, Any
from datetime import datetime
import pandas as pd
from io import StringIO

app = modal.App("smartledger-simple")

# Lightweight image for basic processing
simple_image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "pandas>=2.0.0",
    "numpy>=1.24.0"
])

# Session storage
session_storage = modal.Dict.from_name("smartledger-simple-sessions", create_if_missing=True)

@app.function(image=simple_image, timeout=300)
def create_index(csv_data: str, session_id: str) -> Dict[str, Any]:
    """Create a simple financial index without LLM models"""
    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))
        
        # Basic data processing
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Create simple data structure
        transactions = []
        for _, row in df.iterrows():
            transaction = {
                "date": str(row.get('date', '')),
                "description": str(row.get('description', row.get('vendor', 'Unknown'))),
                "amount": float(row.get('amount', 0)),
                "category": str(row.get('category', 'Uncategorized'))
            }
            transactions.append(transaction)
        
        # Store session data
        session_data = {
            "transactions": transactions,
            "total_transactions": len(transactions),
            "total_amount": sum(t["amount"] for t in transactions),
            "created_at": datetime.now().isoformat(),
            "session_id": session_id
        }
        
        session_storage[session_id] = session_data
        
        return {
            "status": "success",
            "session_id": session_id,
            "total_transactions": len(transactions),
            "total_amount": session_data["total_amount"],
            "created_at": session_data["created_at"]
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=simple_image, timeout=300)
def query_data(query: str, session_id: str) -> Dict[str, Any]:
    """Query financial data using simple keyword matching"""
    try:
        # Check if session exists
        if session_id not in session_storage:
            available_sessions = list(session_storage.keys())
            return {
                "status": "error",
                "error": f"Session {session_id} not found",
                "available_sessions": available_sessions[:5]  # Show first 5
            }
        
        session_data = session_storage[session_id]
        transactions = session_data["transactions"]
        
        # Simple keyword-based search
        query_lower = query.lower()
        keywords = query_lower.split()
        
        matching_transactions = []
        for transaction in transactions:
            # Check if any keyword matches description or category
            text_to_search = f"{transaction['description']} {transaction['category']}".lower()
            if any(keyword in text_to_search for keyword in keywords):
                matching_transactions.append(transaction)
        
        # Basic analysis
        if matching_transactions:
            total_amount = sum(t["amount"] for t in matching_transactions)
            categories = list(set(t["category"] for t in matching_transactions))
            
            # Simple response generation
            if "food" in query_lower or "restaurant" in query_lower:
                analysis = f"Found {len(matching_transactions)} food-related transactions totaling ${total_amount:.2f}"
            elif "total" in query_lower or "spend" in query_lower:
                analysis = f"Total spending: ${total_amount:.2f} across {len(matching_transactions)} transactions"
            else:
                analysis = f"Found {len(matching_transactions)} matching transactions in categories: {', '.join(categories)}"
        else:
            total_amount = 0
            categories = []
            analysis = "No transactions found matching your query"
        
        return {
            "status": "success",
            "query": query,
            "llm_analysis": analysis,
            "matching_transactions": len(matching_transactions),
            "total_amount": total_amount,
            "categories": categories,
            "results": matching_transactions[:5],  # Top 5 results
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=simple_image)
def list_sessions() -> Dict[str, Any]:
    """List all active sessions"""
    try:
        sessions = {}
        for session_id in session_storage.keys():
            session_data = session_storage[session_id]
            sessions[session_id] = {
                "total_transactions": session_data.get("total_transactions", 0),
                "created_at": session_data.get("created_at", "unknown")
            }
        
        return {
            "status": "success",
            "sessions": sessions,
            "session_count": len(sessions)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.function(image=simple_image)
def health_check() -> Dict[str, Any]:
    """Simple health check"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "Simple Modal functions ready"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("Simple Modal functions for immediate testing")
    print("Run: modal deploy simple_modal.py --name smartledger-simple")