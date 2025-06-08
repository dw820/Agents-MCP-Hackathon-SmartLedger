import gradio as gr
import pandas as pd
from utils.ledger_analysis import handle_csv_upload, load_sample_data, analyze_ledger_data
from typing import Dict
import io
import uuid
from datetime import datetime

# Setup Modal client
try:
    import modal
    # Try simple functions first, fallback to complex if needed
    try:
        modal_app = modal.App.lookup("smartledger-simple", create_if_missing=False)
        modal_create_index = modal.Function.lookup("smartledger-simple", "create_index")
        modal_query_data = modal.Function.lookup("smartledger-simple", "query_data") 
        modal_check_health = modal.Function.lookup("smartledger-simple", "health_check")
        modal_list_sessions = modal.Function.lookup("smartledger-simple", "list_sessions")
        print("✅ Using simple Modal functions")
    except:
        # Fallback to complex functions
        modal_app = modal.App.lookup("smartledger", create_if_missing=False)
        modal_create_index = modal.Function.lookup("smartledger", "create_index")
        modal_query_data = modal.Function.lookup("smartledger", "query_data") 
        modal_check_health = modal.Function.lookup("smartledger", "check_health")
        modal_list_sessions = modal.Function.lookup("smartledger", "list_sessions")
        print("✅ Using complex Modal functions")
    
    modal_available = True
    print("✅ Modal client connected successfully")
except Exception as e:
    print(f"⚠️ Modal not available: {e}")
    modal_available = False
    modal_create_index = None
    modal_query_data = None
    modal_check_health = None
    modal_list_sessions = None

# Global session management
current_session_id = None

def analyze_ledger_from_csv(csv_content: str) -> Dict:
    """
    Analyze a ledger CSV using Modal serverless functions for LLM-powered insights.
    
    This function processes CSV data and creates an intelligent financial index using
    Modal's serverless compute with embedding models and LLMs.
    
    Args:
        csv_content: CSV content as string with required columns: date,vendor,amount
                    Optional columns: category,description
        
    Returns:
        Dictionary containing analysis results, statistics, and LLM indexing status
    """
    global current_session_id
    
    try:
        # Parse CSV content for basic analysis
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Process the data
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Handle different column names for vendor/description
        if 'vendor' not in df.columns and 'description' in df.columns:
            df['vendor'] = df['description']
        elif 'vendor' not in df.columns and 'description' not in df.columns:
            df['vendor'] = 'Unknown'
            
        df = df.dropna(subset=['date', 'amount'])
        
        if df.empty:
            return {"error": "No valid transactions found"}
        
        # Generate basic analysis
        analysis_text = analyze_ledger_data(df)
        
        # Create structured response
        total_amount = float(df['amount'].sum())
        transaction_count = len(df)
        avg_amount = float(df['amount'].mean())
        
        # Top vendors
        top_vendors = df['vendor'].value_counts().head(5).to_dict()
        
        # Categories if available
        categories = {}
        if 'category' in df.columns:
            categories = df.groupby('category')['amount'].sum().to_dict()
            categories = {k: float(v) for k, v in categories.items()}
        
        # Create Modal index for LLM analysis
        modal_indexing_status = "Modal not available"
        llm_ready = False
        
        if modal_available and modal_create_index:
            try:
                # Generate unique session ID
                current_session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Standardize column names for Modal
                modal_df = df.copy()
                if 'vendor' in modal_df.columns:
                    modal_df['description'] = modal_df['vendor']
                if 'category' not in modal_df.columns:
                    modal_df['category'] = 'Uncategorized'
                
                # Convert to CSV for Modal
                modal_csv = modal_df.to_csv(index=False)
                
                # Call Modal function to create index
                print(f"🚀 Creating Modal index for session: {current_session_id}")
                modal_result = modal_create_index.remote(modal_csv, current_session_id)
                
                if modal_result.get("status") == "success":
                    modal_indexing_status = f"✅ Modal LLM index created (Session: {current_session_id[:12]}...)"
                    llm_ready = True
                else:
                    modal_indexing_status = f"❌ Modal indexing failed: {modal_result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                modal_indexing_status = f"❌ Modal indexing error: {str(e)}"
        
        return {
            "status": "success",
            "summary": {
                "total_transactions": transaction_count,
                "total_amount": total_amount,
                "average_transaction": avg_amount,
                "date_range": {
                    "start": df['date'].min().strftime('%Y-%m-%d'),
                    "end": df['date'].max().strftime('%Y-%m-%d')
                }
            },
            "top_vendors": top_vendors,
            "categories": categories,
            "analysis_text": analysis_text,
            "indexing_status": modal_indexing_status,
            "llm_ready": llm_ready,
            "session_id": current_session_id,
            "modal_available": modal_available
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def get_spending_by_category(csv_content: str, category: str = "") -> Dict:
    """
    Get spending breakdown by category or filter by specific category.
    
    Analyzes financial data to provide category-based spending insights.
    If no category is specified, returns all categories with totals.
    
    Args:
        csv_content: CSV content as string with transaction data
        category: Optional specific category name to filter results
        
    Returns:
        Spending information organized by category with totals and transaction counts
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        if 'category' not in df.columns:
            return {"error": "No category column found in the data"}
        
        if category:
            # Filter by specific category
            category_data = df[df['category'].str.contains(category, case=False, na=False)]
            total = float(category_data['amount'].sum())
            count = len(category_data)
            
            return {
                "category": category,
                "total_amount": total,
                "transaction_count": count,
                "transactions": category_data[['date', 'vendor', 'amount', 'description']].to_dict('records')
            }
        else:
            # All categories
            category_totals = df.groupby('category')['amount'].agg(['sum', 'count']).round(2)
            
            result = {}
            for cat, row in category_totals.iterrows():
                result[cat] = {
                    "total_amount": float(row['sum']),
                    "transaction_count": int(row['count'])
                }
            
            return {"categories": result}
            
    except Exception as e:
        return {"error": f"Category analysis failed: {str(e)}"}

def get_vendor_analysis(csv_content: str, vendor: str = "") -> Dict:
    """
    Analyze spending patterns by vendor with detailed transaction breakdowns.
    
    Provides comprehensive vendor spending analysis including total amounts,
    transaction frequencies, and average spending per vendor.
    
    Args:
        csv_content: CSV content as string containing transaction data
        vendor: Optional specific vendor name to analyze in detail
        
    Returns:
        Vendor spending analysis with totals, averages, and transaction details
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['vendor', 'amount'])
        
        if vendor:
            # Specific vendor analysis
            vendor_data = df[df['vendor'].str.contains(vendor, case=False, na=False)]
            if vendor_data.empty:
                return {"error": f"No transactions found for vendor: {vendor}"}
            
            total = float(vendor_data['amount'].sum())
            count = len(vendor_data)
            avg = float(vendor_data['amount'].mean())
            
            return {
                "vendor": vendor,
                "total_amount": total,
                "transaction_count": count,
                "average_amount": avg,
                "transactions": vendor_data[['date', 'amount', 'category', 'description']].to_dict('records')
            }
        else:
            # All vendors summary
            vendor_stats = df.groupby('vendor')['amount'].agg(['sum', 'count', 'mean']).round(2)
            vendor_stats = vendor_stats.sort_values('sum', ascending=False)
            
            result = {}
            for vendor_name, row in vendor_stats.iterrows():
                result[vendor_name] = {
                    "total_amount": float(row['sum']),
                    "transaction_count": int(row['count']),
                    "average_amount": float(row['mean'])
                }
            
            return {"vendors": result}
            
    except Exception as e:
        return {"error": f"Vendor analysis failed: {str(e)}"}

def detect_financial_anomalies(custom_query: str = "") -> Dict:
    """
    Detect anomalies using Modal's LLM-powered analysis.
    
    Performs intelligent anomaly detection on financial transactions using Modal's
    serverless LLM functions to identify spending spikes, unusual patterns, and suspicious transactions.
    
    Args:
        custom_query: Optional custom analysis query for specific anomaly detection
        
    Returns:
        Dictionary containing anomaly analysis results and recommendations
    """
    global current_session_id
    
    try:
        if not current_session_id:
            return {"error": "No financial data indexed. Please upload and analyze a CSV file first."}
        
        if not modal_available or not modal_query_data:
            return {"error": "Modal LLM functions not available. Using basic analysis."}
        
        # Use custom query if provided, otherwise default anomaly detection query
        if custom_query.strip():
            query = custom_query
            analysis_type = "custom"
        else:
            query = "Detect any unusual spending patterns, anomalies, or suspicious transactions in this financial data. Look for spending spikes, unusual vendors, or irregular amounts."
            analysis_type = "comprehensive"
        
        # Call Modal function for anomaly detection
        print(f"🔍 Running anomaly detection for session: {current_session_id}")
        modal_result = modal_query_data.remote(query, current_session_id)
        
        if modal_result.get("status") == "success":
            return {
                "status": "success",
                "anomaly_analysis": modal_result.get("llm_analysis", "No analysis available"),
                "analysis_type": analysis_type,
                "matching_transactions": modal_result.get("matching_transactions", 0),
                "total_amount": modal_result.get("total_amount", 0),
                "session_id": current_session_id
            }
        else:
            return {"error": f"Modal anomaly detection failed: {modal_result.get('error', 'Unknown error')}"}
        
    except Exception as e:
        return {"error": f"Anomaly detection failed: {str(e)}"}

def query_financial_data(question: str) -> Dict:
    """
    Query financial data using Modal's LLM-powered analysis.
    
    Ask natural language questions about spending patterns, vendor analysis, budget trends,
    or any other financial insights. Modal's serverless LLM will analyze the data to provide answers.
    
    Args:
        question: Natural language question about the financial data
        
    Returns:
        Dictionary containing the answer and supporting analysis
    """
    global current_session_id
    
    try:
        if not question.strip():
            return {"error": "Please provide a question about your financial data"}
        
        if not current_session_id:
            return {"error": "No financial data indexed. Please upload and analyze a CSV file first."}
            
        if not modal_available or not modal_query_data:
            return {"error": "Modal LLM functions not available. Please ensure Modal is deployed."}
        
        # Debug: Check available sessions
        if modal_list_sessions:
            try:
                sessions_info = modal_list_sessions.remote()
                print(f"📝 Available sessions: {sessions_info}")
            except Exception as e:
                print(f"⚠️ Could not list sessions: {e}")
        
        # Call Modal function for intelligent query processing
        print(f"💬 Processing query for session: {current_session_id}")
        print(f"Question: {question}")
        
        modal_result = modal_query_data.remote(question, current_session_id)
        
        if modal_result.get("status") == "success":
            return {
                "status": "success", 
                "question": question,
                "insights": modal_result.get("llm_analysis", "No insights available"),
                "matching_transactions": modal_result.get("matching_transactions", 0),
                "total_amount": modal_result.get("total_amount", 0),
                "results": modal_result.get("results", []),
                "session_id": current_session_id,
                "processed_at": modal_result.get("processed_at")
            }
        else:
            return {"error": f"Modal query failed: {modal_result.get('error', 'Unknown error')}"}
        
    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

with gr.Blocks(title="SmartLedger - Financial Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 SmartLedger - Smart Business Accounting")
        gr.Markdown("Upload your accounting ledger CSV file to analyze transactions, spending patterns, and get financial insights.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload & Analyze")
                
                csv_file = gr.File(
                    label="Upload CSV Ledger",
                    file_types=[".csv"],
                    value=None
                )
                
                gr.Markdown("*Required columns: date, vendor, amount*\n*Optional: category, description*")
                
                analyze_btn = gr.Button("Analyze Ledger", variant="primary", size="lg")
                
                with gr.Accordion("📋 CSV Format Guide", open=False):
                    gr.Textbox(
                        value="""date,vendor,amount,category,description
2024-01-15,Coffee Shop,4.50,Meals,Morning coffee
2024-01-16,Gas Station,45.00,Vehicle,Fuel
2024-01-17,Office Depot,23.99,Supplies,Paper""",
                        label="Expected CSV Format",
                        interactive=False,
                        lines=4,
                        max_lines=4
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                    value="Upload a CSV file to begin analysis"
                )
                
                ledger_dataframe = gr.Dataframe(
                    label="Transaction Data",
                    interactive=False,
                    wrap=True,
                    value=None
                )
                
                analysis_text = gr.Textbox(
                    label="Financial Insights", 
                    interactive=False,
                    lines=15,
                    value=""
                )
        
        # LLM Analysis Section
        with gr.Row():
            gr.Markdown("## 🤖 AI-Powered Analysis")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Anomaly Detection")
                anomaly_btn = gr.Button("Detect Anomalies", variant="primary")
                
                gr.Markdown("### 💬 Ask Questions")
                question_input = gr.Textbox(
                    label="Ask about your data",
                    placeholder="e.g., What are my highest spending categories this month?",
                    lines=2
                )
                query_btn = gr.Button("Get Insights", variant="secondary")
            
            with gr.Column(scale=2):
                llm_results = gr.Textbox(
                    label="AI Analysis Results",
                    interactive=False,
                    lines=10,
                    value="Upload and analyze a CSV file to enable AI-powered insights"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Quick Test")
                sample_btn = gr.Button("Load Sample Data", variant="secondary")
                
            with gr.Column():
                gr.Markdown("### 🔗 Modal LLM Integration")
                if modal_available:
                    gr.Markdown("✅ **Modal Status:** Connected\n🚀 **AI Models:** Embedding + LLM ready\n📡 **Functions:** `create_index`, `query_data`, `check_health`")
                else:
                    gr.Markdown("❌ **Modal Status:** Not available\n⚠️ **Fallback:** Basic analysis only")
                    
                gr.Markdown("**Available MCP Tools:**\n- `analyze_ledger_from_csv`\n- `get_spending_by_category`\n- `get_vendor_analysis`\n- `detect_financial_anomalies`\n- `query_financial_data`")
        
        # Enhanced CSV handler with Modal integration
        def enhanced_csv_upload(csv_file):
            """Enhanced CSV upload handler with Modal LLM indexing"""
            # First do the standard analysis
            status, df, analysis = handle_csv_upload(csv_file)
            
            # If successful, also index with Modal
            if df is not None:
                try:
                    # Convert dataframe to CSV for Modal analysis
                    csv_content = df.to_csv(index=False)
                    modal_result = analyze_ledger_from_csv(csv_content)
                    
                    if modal_result.get("status") == "success":
                        modal_status = modal_result.get("indexing_status", "Unknown status")
                        status += f"\n{modal_status}"
                        
                        # Add Modal session info
                        if modal_result.get("session_id"):
                            status += f"\nSession ID: {modal_result['session_id'][:20]}..."
                            
                        # Enhance analysis with Modal insights
                        if modal_result.get("llm_ready"):
                            analysis += f"\n\n🤖 AI Analysis Ready:\n• {modal_result['summary']['total_transactions']} transactions indexed\n• Modal LLM functions available for intelligent queries\n• Ask questions in the AI Analysis section below"
                    else:
                        status += f"\n⚠️ Modal analysis failed: {modal_result.get('error', 'Unknown error')}"
                        
                except Exception as e:
                    status += f"\n⚠️ Modal analysis error: {str(e)}"
            
            return status, df, analysis
        
        # Event handlers
        analyze_btn.click(
            fn=enhanced_csv_upload,
            inputs=[csv_file],
            outputs=[status_text, ledger_dataframe, analysis_text]
        )
        
        # Auto-analyze when file is uploaded
        csv_file.change(
            fn=enhanced_csv_upload,
            inputs=[csv_file],
            outputs=[status_text, ledger_dataframe, analysis_text]
        )
        
        # Enhanced sample data handler with Modal indexing
        def enhanced_sample_data():
            """Enhanced sample data loader with Modal LLM indexing"""
            status, df, analysis = load_sample_data()
            
            # Also index the sample data with Modal
            if df is not None:
                try:
                    # Convert dataframe to CSV for Modal analysis
                    csv_content = df.to_csv(index=False)
                    modal_result = analyze_ledger_from_csv(csv_content)
                    
                    if modal_result.get("status") == "success":
                        modal_status = modal_result.get("indexing_status", "Unknown status")
                        status += f"\n{modal_status}"
                        
                        # Add Modal session info
                        if modal_result.get("session_id"):
                            status += f"\nSession ID: {modal_result['session_id'][:20]}..."
                            
                        # Enhance analysis with Modal insights
                        if modal_result.get("llm_ready"):
                            analysis += f"\n\n🤖 AI Analysis Ready:\n• {modal_result['summary']['total_transactions']} transactions indexed\n• Modal LLM functions available for intelligent queries\n• Try asking: 'What are my highest spending categories?'"
                    else:
                        status += f"\n⚠️ Modal analysis failed: {modal_result.get('error', 'Unknown error')}"
                        
                except Exception as e:
                    status += f"\n⚠️ Modal analysis error: {str(e)}"
            
            return status, df, analysis
        
        # Sample data handler
        sample_btn.click(
            fn=enhanced_sample_data,
            outputs=[status_text, ledger_dataframe, analysis_text]
        )
        
        # LLM Analysis handlers
        def run_anomaly_detection():
            """Run anomaly detection and return formatted results"""
            result = detect_financial_anomalies()
            if result.get("status") == "success":
                return f"🚨 ANOMALY DETECTION RESULTS:\n\n{result['anomaly_analysis']}"
            else:
                return f"❌ {result.get('error', 'Unknown error')}"
        
        def run_query(question):
            """Run financial query and return formatted results"""
            if not question.strip():
                return "Please enter a question about your financial data."
            
            result = query_financial_data(question)
            if result.get("status") == "success":
                return f"💡 QUESTION: {result['question']}\n\n📊 INSIGHTS:\n{result['insights']}"
            else:
                return f"❌ {result.get('error', 'Unknown error')}"
        
        anomaly_btn.click(
            fn=run_anomaly_detection,
            outputs=[llm_results]
        )
        
        query_btn.click(
            fn=run_query,
            inputs=[question_input],
            outputs=[llm_results]
        )
        
        # Auto-query on Enter
        question_input.submit(
            fn=run_query,
            inputs=[question_input],
            outputs=[llm_results]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True
    )