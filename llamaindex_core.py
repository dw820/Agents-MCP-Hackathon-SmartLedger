from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class TransactionIndex:
    def __init__(self):
        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding()
        Settings.llm = OpenAI(temperature=0.1)
        
        self.index = None
        self.transactions_data = []
        self.query_engine = None

    def build_index(self, ledger_df: pd.DataFrame):
        """Build vector index from uploaded ledger CSV"""
        documents = []
        self.transactions_data = []

        for _, row in ledger_df.iterrows():
            # Create searchable document from each ledger entry
            doc_text = self._create_document_text(row)
            
            # Store transaction data
            transaction_data = {
                'date': str(row.get('date', '')),
                'vendor': str(row.get('vendor', '')),
                'amount': float(row.get('amount', 0)),
                'category': str(row.get('category', 'Unknown')),
                'description': str(row.get('description', ''))
            }
            self.transactions_data.append(transaction_data)

            doc = Document(
                text=doc_text,
                metadata=transaction_data
            )
            documents.append(doc)

        if documents:
            # Create vector index
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )

    def _create_document_text(self, row) -> str:
        """Create searchable text from transaction row"""
        date = row.get('date', '')
        vendor = row.get('vendor', '')
        amount = row.get('amount', 0)
        category = row.get('category', 'Unknown')
        description = row.get('description', '')
        
        return f"Date: {date}, Vendor: {vendor}, Amount: ${amount:.2f}, Category: {category}, Description: {description}"

    def reconcile_transactions(self, new_transactions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Reconcile new transactions against indexed ledger"""
        matched = []
        unmatched = []

        for trans in new_transactions:
            # Skip transactions with errors
            if 'error' in trans:
                unmatched.append(trans)
                continue

            # Try exact matching first
            exact_match = self._find_exact_match(trans)
            if exact_match:
                matched_trans = {**trans, 'ledger_match': exact_match, 'confidence': 1.0}
                matched.append(matched_trans)
                continue

            # Try semantic matching via LlamaIndex
            if self.index:
                similar_match = self._find_semantic_match(trans)
                if similar_match:
                    matched.append(similar_match)
                else:
                    unmatched.append(trans)
            else:
                unmatched.append(trans)

        return matched, unmatched

    def _find_exact_match(self, transaction: Dict) -> Dict:
        """Find exact amount and date match in ledger (with tolerance)"""
        trans_amount = transaction.get('amount')
        trans_date = transaction.get('date')
        
        if not trans_amount or not trans_date:
            return None

        try:
            trans_date_obj = datetime.strptime(trans_date, '%Y-%m-%d')
        except (ValueError, TypeError):
            return None

        for ledger_entry in self.transactions_data:
            # Check amount match (within $2.00 tolerance)
            ledger_amount = ledger_entry.get('amount', 0)
            amount_diff = abs(ledger_amount - trans_amount)
            
            if amount_diff <= 2.0:
                # Check date match (within 5 days)
                try:
                    ledger_date_obj = datetime.strptime(ledger_entry['date'], '%Y-%m-%d')
                    date_diff = abs((ledger_date_obj - trans_date_obj).days)
                    
                    if date_diff <= 5:
                        return ledger_entry
                except (ValueError, TypeError):
                    continue

        return None

    def _find_semantic_match(self, transaction: Dict) -> Dict:
        """Find semantically similar transactions using vector search"""
        if not self.query_engine:
            return None

        vendor = transaction.get('vendor', '')
        amount = transaction.get('amount', 0)
        description = transaction.get('description', '')
        
        # Create search query
        query = f"Vendor similar to '{vendor}' with amount around ${amount:.2f}"
        if description:
            query += f" for {description}"

        try:
            response = self.query_engine.query(query)
            
            # Check if we have source nodes with good similarity
            if hasattr(response, 'source_nodes') and response.source_nodes:
                best_node = response.source_nodes[0]
                
                # Use a similarity threshold
                if best_node.score > 0.75:  # Adjust threshold as needed
                    suggested_category = best_node.metadata.get('category', 'Unknown')
                    
                    return {
                        **transaction,
                        'suggested_category': suggested_category,
                        'confidence': min(best_node.score, 0.9),  # Cap confidence at 0.9 for semantic matches
                        'similar_transaction': best_node.metadata
                    }
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            
        return None

    def query_natural_language(self, query: str) -> str:
        """Handle natural language queries about transactions"""
        if not self.query_engine:
            return "No transaction data indexed yet. Please upload and process documents first."

        try:
            response = self.query_engine.query(query)
            return str(response)
        except Exception as e:
            return f"Query failed: {str(e)}"

    def get_category_suggestions(self, vendor: str) -> List[str]:
        """Get category suggestions based on similar vendors"""
        if not self.index:
            return []

        query = f"Vendors similar to '{vendor}'"
        try:
            response = self.query_engine.query(query)
            
            categories = set()
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:3]:  # Top 3 matches
                    category = node.metadata.get('category')
                    if category and category != 'Unknown':
                        categories.add(category)
            
            return list(categories)
        except:
            return []

    def get_spending_summary(self, category: str = None, days: int = 30) -> Dict:
        """Get spending summary for specified period"""
        if not self.transactions_data:
            return {"error": "No transaction data available"}

        cutoff_date = datetime.now() - timedelta(days=days)
        total_amount = 0
        transaction_count = 0
        categories = {}

        for trans in self.transactions_data:
            try:
                trans_date = datetime.strptime(trans['date'], '%Y-%m-%d')
                if trans_date >= cutoff_date:
                    amount = trans.get('amount', 0)
                    trans_category = trans.get('category', 'Unknown')
                    
                    if category is None or trans_category.lower() == category.lower():
                        total_amount += amount
                        transaction_count += 1
                        
                        if trans_category in categories:
                            categories[trans_category] += amount
                        else:
                            categories[trans_category] = amount
            except (ValueError, TypeError):
                continue

        return {
            "total_amount": total_amount,
            "transaction_count": transaction_count,
            "categories": categories,
            "period_days": days
        }

# Global instance for the application
transaction_index = TransactionIndex()

def reconcile_with_llamaindex(transactions: List[Dict], ledger_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Main reconciliation function called from app.py"""
    try:
        # Build index from ledger
        transaction_index.build_index(ledger_df)

        # Reconcile transactions
        matched, unmatched = transaction_index.reconcile_transactions(transactions)

        return matched, unmatched
    
    except Exception as e:
        error_transaction = {"error": f"Reconciliation failed: {str(e)}"}
        return [], [error_transaction]

def query_transactions_mcp(query: str) -> str:
    """MCP query function for natural language queries"""
    return transaction_index.query_natural_language(query)

def get_spending_analytics(category: str = None, days: int = 30) -> Dict:
    """Get spending analytics for MCP queries"""
    return transaction_index.get_spending_summary(category, days)

# Utility functions for common queries
def format_currency(amount: float) -> str:
    """Format currency for display"""
    return f"${amount:,.2f}"

def parse_date_flexible(date_str: str) -> datetime:
    """Parse date string with multiple format support"""
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y']
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")