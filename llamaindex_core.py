"""
LlamaIndex core module for SmartLedger
Handles document indexing and intelligent querying of financial data
"""

import pandas as pd
from typing import List, Dict
import json

try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from modal_llama_integration import create_modal_llm, create_modal_embedding
except ImportError:
    # Graceful fallback if LlamaIndex not installed
    Document = None
    VectorStoreIndex = None
    Settings = None
    create_modal_llm = None
    create_modal_embedding = None

class LedgerIndexer:
    """
    Handles indexing and querying of financial ledger data using LlamaIndex
    """
    
    def __init__(self, use_modal_llm: bool = True):
        """
        Initialize the LedgerIndexer
        
        Args:
            use_modal_llm: Whether to use Modal-deployed LLM (True) or OpenAI directly (False)
        """
        self.index = None
        self.df = None
        self.use_modal_llm = use_modal_llm
        
        # Initialize LlamaIndex settings
        if Settings is not None:
            if use_modal_llm and create_modal_llm is not None:
                # Use Modal-hosted models
                try:
                    Settings.llm = create_modal_llm(temperature=0.1, max_new_tokens=512)
                    Settings.embed_model = create_modal_embedding()
                    print("✅ Using Modal-hosted models")
                except Exception as e:
                    print(f"⚠️ Failed to initialize Modal models: {e}")
                    print("💡 Run 'modal deploy modal_functions.py' to enable Modal models")
                    Settings.llm = None
                    Settings.embed_model = None
            else:
                print("⚠️ Modal integration not available - LlamaIndex indexing disabled")
                Settings.llm = None
                Settings.embed_model = None
    
    def create_monthly_summaries(self, df: pd.DataFrame) -> List[Document]:
        """
        Create monthly summary documents optimized for anomaly detection
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            List of LlamaIndex Documents containing monthly summaries
        """
        if Document is None:
            raise ImportError("LlamaIndex not available")
            
        documents = []
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Group by month
        monthly_groups = df.groupby('year_month')
        
        for period, month_data in monthly_groups:
            # Calculate monthly statistics
            total_amount = month_data['amount'].sum()
            transaction_count = len(month_data)
            avg_transaction = month_data['amount'].mean()
            unique_vendors = month_data['vendor'].nunique()
            
            # Category breakdown
            category_breakdown = {}
            if 'category' in month_data.columns:
                category_breakdown = month_data.groupby('category')['amount'].sum().to_dict()
            
            # Vendor breakdown (top 10)
            vendor_breakdown = month_data.groupby('vendor')['amount'].sum().nlargest(10).to_dict()
            
            # Unusual patterns detection data
            large_transactions = month_data[month_data['amount'] > month_data['amount'].quantile(0.95)]
            frequent_vendors = month_data['vendor'].value_counts().head(5).to_dict()
            
            # Create document content
            content = f"""
FINANCIAL SUMMARY FOR {period}

OVERVIEW:
- Total Spending: ${total_amount:,.2f}
- Transaction Count: {transaction_count}
- Average Transaction: ${avg_transaction:.2f}
- Unique Vendors: {unique_vendors}
- Date Range: {month_data['date'].min().strftime('%Y-%m-%d')} to {month_data['date'].max().strftime('%Y-%m-%d')}

CATEGORY BREAKDOWN:
{json.dumps(category_breakdown, indent=2)}

TOP VENDORS BY SPENDING:
{json.dumps(vendor_breakdown, indent=2)}

FREQUENT VENDORS (by transaction count):
{json.dumps(frequent_vendors, indent=2)}

LARGE TRANSACTIONS (95th percentile):
{large_transactions[['date', 'vendor', 'amount', 'category']].to_string(index=False)}

DAILY SPENDING PATTERN:
{month_data.groupby(month_data['date'].dt.day)['amount'].sum().to_dict()}
"""
            
            # Create metadata for better querying
            metadata = {
                "period": str(period),
                "year": period.year,
                "month": period.month,
                "total_amount": float(total_amount),
                "transaction_count": int(transaction_count),
                "avg_transaction": float(avg_transaction),
                "unique_vendors": int(unique_vendors),
                "top_category": max(category_breakdown.items(), key=lambda x: x[1])[0] if category_breakdown else "Unknown",
                "document_type": "monthly_summary"
            }
            
            doc = Document(
                text=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def create_transaction_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Create individual transaction documents for granular analysis
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            List of LlamaIndex Documents for individual transactions
        """
        if Document is None:
            raise ImportError("LlamaIndex not available")
            
        documents = []
        
        for idx, row in df.iterrows():
            content = f"""
TRANSACTION RECORD

Date: {row['date']}
Vendor: {row['vendor']}
Amount: ${row['amount']:.2f}
Category: {row.get('category', 'Uncategorized')}
Description: {row.get('description', 'No description')}

Context:
- Day of week: {pd.to_datetime(row['date']).strftime('%A')}
- Month: {pd.to_datetime(row['date']).strftime('%B %Y')}
"""
            
            metadata = {
                "transaction_id": str(idx),
                "date": str(row['date']),
                "vendor": str(row['vendor']),
                "amount": float(row['amount']),
                "category": str(row.get('category', 'Uncategorized')),
                "document_type": "transaction"
            }
            
            doc = Document(
                text=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def index_ledger_data(self, df: pd.DataFrame, include_transactions: bool = False) -> bool:
        """
        Index the ledger data using LlamaIndex
        
        Args:
            df: DataFrame containing ledger data
            include_transactions: Whether to include individual transactions
            
        Returns:
            True if indexing successful, False otherwise
        """
        try:
            if VectorStoreIndex is None:
                print("❌ LlamaIndex not available - install with: pip install llama-index")
                return False
            
            if Settings.llm is None or Settings.embed_model is None:
                print("❌ Modal models not configured - deploy with: modal deploy modal_functions.py")
                return False
            
            self.df = df.copy()
            documents = []
            
            # Create monthly summary documents (primary for anomaly detection)
            monthly_docs = self.create_monthly_summaries(df)
            documents.extend(monthly_docs)
            
            # Optionally include individual transactions
            if include_transactions:
                transaction_docs = self.create_transaction_documents(df)
                documents.extend(transaction_docs)
            
            print(f"📄 Created {len(documents)} documents for indexing")
            
            # Create the index
            self.index = VectorStoreIndex.from_documents(documents)
            
            print("✅ Successfully indexed financial data")
            return True
            
        except Exception as e:
            print(f"❌ Error indexing data: {e}")
            return False
    
    def query_anomalies(self, query: str = None) -> str:
        """
        Query for anomalies in the financial data
        
        Args:
            query: Custom query string, defaults to anomaly detection
            
        Returns:
            LLM response about anomalies found
        """
        if self.index is None:
            return "❌ No data indexed. Please upload and analyze a CSV file first, then ensure Modal models are deployed."
        
        if Settings.llm is None:
            return "❌ Modal LLM not available. Please deploy Modal functions: modal deploy modal_functions.py"
        
        if query is None:
            query = """
            Analyze this financial data for anomalies and unusual patterns. Look for:
            1. Month-over-month spending increases or decreases > 20%
            2. Unusual vendor patterns or new large expenses
            3. Category spending that deviates from normal patterns
            4. Suspicious transaction amounts or frequencies
            5. Seasonal anomalies or unexpected spikes
            
            Provide specific examples with amounts and dates where possible.
            Focus on actionable insights for business expense management.
            """
        
        try:
            print("🔍 Querying indexed data for anomalies...")
            query_engine = self.index.as_query_engine(
                response_mode="tree_summarize",
                verbose=False
            )
            response = query_engine.query(query)
            return str(response)
            
        except Exception as e:
            return f"❌ Error querying data: {e}\n💡 Ensure Modal models are deployed and accessible."
    
    def query_insights(self, question: str) -> str:
        """
        Query the indexed data for specific insights
        
        Args:
            question: Natural language question about the financial data
            
        Returns:
            LLM response with insights
        """
        if self.index is None:
            return "❌ No data indexed. Please upload and analyze a CSV file first, then ensure Modal models are deployed."
        
        if Settings.llm is None:
            return "❌ Modal LLM not available. Please deploy Modal functions: modal deploy modal_functions.py"
        
        try:
            print(f"💬 Answering question: {question}")
            query_engine = self.index.as_query_engine(
                response_mode="compact",
                verbose=False
            )
            response = query_engine.query(question)
            return str(response)
            
        except Exception as e:
            return f"❌ Error querying data: {e}\n💡 Ensure Modal models are deployed and accessible."
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "No index created"}
        
        try:
            return {
                "status": "Index ready",
                "document_count": len(self.index.docstore.docs),
                "data_rows": len(self.df) if self.df is not None else 0,
                "date_range": {
                    "start": self.df['date'].min().strftime('%Y-%m-%d') if self.df is not None else None,
                    "end": self.df['date'].max().strftime('%Y-%m-%d') if self.df is not None else None
                } if self.df is not None else None
            }
        except Exception as e:
            return {"status": f"Error getting stats: {e}"}

# Global indexer instance
_indexer = None

def get_indexer() -> LedgerIndexer:
    """Get or create the global indexer instance"""
    global _indexer
    if _indexer is None:
        _indexer = LedgerIndexer()
    return _indexer

def index_dataframe(df: pd.DataFrame) -> bool:
    """Convenience function to index a DataFrame"""
    indexer = get_indexer()
    return indexer.index_ledger_data(df)

def query_financial_anomalies(custom_query: str = None) -> str:
    """Convenience function to query for anomalies"""
    indexer = get_indexer()
    return indexer.query_anomalies(custom_query)

def query_financial_insights(question: str) -> str:
    """Convenience function to query for insights"""
    indexer = get_indexer()
    return indexer.query_insights(question)