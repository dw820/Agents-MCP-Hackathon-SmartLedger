import gradio as gr
import pandas as pd
from utils.ledger_analysis import handle_csv_upload, analyze_ledger_data
from typing import Dict
import io
import uuid
from datetime import datetime

# Setup Modal client
try:
    import os
    import modal
    
    # Configure Modal authentication from environment variables
    if "MODAL_TOKEN_ID" in os.environ and "MODAL_TOKEN_SECRET" in os.environ:
        modal.config.config["token_id"] = os.environ["MODAL_TOKEN_ID"]
        modal.config.config["token_secret"] = os.environ["MODAL_TOKEN_SECRET"]
    
    import modal
    modal_app = modal.App.lookup("smartledger", create_if_missing=False)
    modal_create_index = modal.Function.from_name("smartledger", "create_index")
    modal_query_data = modal.Function.from_name("smartledger", "query_data") 
    modal_check_health = modal.Function.from_name("smartledger", "check_health")
    modal_list_sessions = modal.Function.from_name("smartledger", "list_sessions")
    modal_process_image = modal.Function.from_name("smartledger", "process_image_transactions")
    modal_reconcile = modal.Function.from_name("smartledger", "reconcile_transactions")
    modal_available = True
    print("✅ Using Modal functions with AI models and image processing")
except Exception as e:
    print(f"⚠️ Modal not available: {e}")
    modal_available = False
    modal_create_index = None
    modal_query_data = None
    modal_check_health = None
    modal_list_sessions = None
    modal_process_image = None
    modal_reconcile = None

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


def process_image_and_reconcile(image_file, csv_file) -> Dict:
    """
    Process both image and CSV files, then reconcile transactions
    
    Args:
        image_file: Uploaded image file (bank statement, receipt)
        csv_file: Uploaded CSV ledger file
        
    Returns:
        Dictionary containing reconciliation results and analysis
    """
    global current_session_id
    
    try:
        if not modal_available:
            return {"error": "Modal functions not available"}
        
        if not image_file or not csv_file:
            return {"error": "Please upload both an image and CSV file"}
        
        # Generate session ID
        current_session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process CSV first
        print("📊 Processing CSV file...")
        try:
            df = pd.read_csv(csv_file.name)
            csv_content = df.to_csv(index=False)
            
            csv_result = modal_create_index.remote(csv_content, current_session_id)
            if csv_result.get("status") != "success":
                return {"error": f"CSV processing failed: {csv_result.get('error', 'Unknown error')}"}
            
            print(f"✅ CSV processed: {csv_result.get('total_transactions', 0)} transactions indexed")
            
        except Exception as e:
            return {"error": f"CSV processing error: {str(e)}"}
        
        # Process image
        print("📷 Processing image file...")
        try:
            import base64
            
            # Read image file and encode to base64
            print(f"type(image_file) in process_image_and_reconcile: {type(image_file)}")
            with open(image_file.name, "rb") as f:
                image_bytes = f.read()
            print(f"type(image_bytes) in process_image_and_reconcile: {type(image_bytes)}")
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            print(f"type(image_base64) in process_image_and_reconcile: {type(image_base64)}")
            
            image_result = modal_process_image.remote(
                image_base64, 
                current_session_id, 
                image_file.name
            )
            
            if image_result.get("status") != "success":
                return {"error": f"Image processing failed: {image_result.get('error', 'Unknown error')}"}
            
            print(f"✅ Image processed: {image_result.get('total_transactions', 0)} transactions extracted")
            
        except Exception as e:
            return {"error": f"Image processing error: {str(e)}"}
        
        # Reconcile transactions
        print("🔄 Reconciling transactions...")
        try:
            reconcile_result = modal_reconcile.remote(current_session_id)
            
            if reconcile_result.get("status") != "success":
                return {"error": f"Reconciliation failed: {reconcile_result.get('error', 'Unknown error')}"}
            
            print(f"✅ Reconciliation complete: {reconcile_result['summary']['total_matches']} matches found")
            
            return {
                "status": "success",
                "session_id": current_session_id,
                "csv_transactions": csv_result.get("total_transactions", 0),
                "image_transactions": image_result.get("total_transactions", 0),
                "reconciliation": reconcile_result,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Reconciliation error: {str(e)}"}
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

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
            return {"error": "No financial data indexed. Please upload and analyze files first."}
            
        if not modal_available or not modal_query_data:
            return {"error": "Modal LLM functions not available. Please ensure Modal is deployed."}
        
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

with gr.Blocks(title="SmartLedger - Transaction Reconciliation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📊 SmartLedger - Smart Business Accounting Reconciliation")
        gr.Markdown("Upload both your CSV ledger and bank statement/receipt image to automatically reconcile transactions with AI-powered confidence scoring.")
        
        # Dual Upload Section
        gr.Markdown("## 📁 Upload Files for Reconciliation")
        
        with gr.Row():
            with gr.Column():
                csv_file = gr.File(
                    label="📊 Upload CSV Ledger",
                    file_types=[".csv"],
                    value=None
                )
                gr.Markdown("*Required columns: date, vendor, amount*\n*Optional: category, description*")
                
            with gr.Column():
                image_file = gr.File(
                    label="📷 Upload Bank Statement/Receipt Image",
                    file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                    value=None
                )
                print(f"type(image_file): {type(image_file)}")
                gr.Markdown("*Supports: Bank statements, receipts, invoices*\n*Formats: JPG, PNG, PDF*")
        
        reconcile_btn = gr.Button("🔄 Process & Reconcile Transactions", variant="primary", size="lg")
        
        with gr.Accordion("📋 File Format Guide", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**CSV Format:**")
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
                with gr.Column():
                    gr.Markdown("**Image Requirements:**")
                    gr.Markdown("""
• Clear, readable text
• Bank statements or receipts
• Transaction details visible
• Date, amount, vendor information
• JPG, PNG, or PDF format
                    """)
        
        # Reconciliation Results Section
        gr.Markdown("## 📊 Reconciliation Results")
        
        reconciliation_status = gr.Textbox(
            label="Processing Status",
            interactive=False,
            lines=8,
            value="Upload both CSV and image files to begin reconciliation"
        )
        
        # Results Tabs
        with gr.Tabs():
            with gr.TabItem("🎯 Match Summary"):
                summary_dataframe = gr.Dataframe(
                    label="Reconciliation Summary",
                    interactive=False,
                    wrap=True,
                    value=None
                )
                
            with gr.TabItem("✅ High Confidence Matches"):
                high_confidence_dataframe = gr.Dataframe(
                    label="High Confidence Matches (≥85%)",
                    interactive=False,
                    wrap=True,
                    value=None
                )
                
            with gr.TabItem("⚠️ Medium Confidence Matches"):
                medium_confidence_dataframe = gr.Dataframe(
                    label="Medium Confidence Matches (65-84%)",
                    interactive=False,
                    wrap=True,
                    value=None
                )
                
            with gr.TabItem("🔍 Low Confidence Matches"):
                low_confidence_dataframe = gr.Dataframe(
                    label="Low Confidence Matches (<65%) - Review Required",
                    interactive=False,
                    wrap=True,
                    value=None
                )
                
            with gr.TabItem("❌ Unmatched Transactions"):
                unmatched_dataframe = gr.Dataframe(
                    label="Unmatched Transactions",
                    interactive=False,
                    wrap=True,
                    value=None
                )
        
#         # AI Analysis Section
#         gr.Markdown("## 🤖 AI-Powered Analysis")
        
#         # Questions Section
#         gr.Markdown("### 💬 Ask Questions About Your Data")
#         question_input = gr.Textbox(
#             label="Natural Language Query",
#             placeholder="e.g., What are my highest spending categories? Show me restaurant transactions.",
#             lines=2
#         )
#         query_btn = gr.Button("Get AI Insights", variant="primary", size="lg")
        
#         # AI Results
#         llm_results = gr.Textbox(
#             label="AI Analysis Results",
#             interactive=False,
#             lines=12,
#             value="Upload and analyze a CSV file to enable AI-powered insights"
#         )
        
#         # Quick Test Section
#         gr.Markdown("## 🎯 Quick Test")
#         sample_btn = gr.Button("Load Sample Data", variant="primary", size="lg")
        
#         # System Status Section  
#         gr.Markdown("## 🔗 System Status & Integration")
        
#         # Modal Status
#         if modal_available:
#             gr.Markdown("✅ **Modal AI Status:** Connected and Ready\n🚀 **Features:** Smart transaction analysis, natural language queries\n📡 **Functions:** Session management, keyword search, basic insights")
#         else:
#             gr.Markdown("❌ **Modal AI Status:** Not available\n⚠️ **Mode:** Basic analysis only (no AI features)")
            
#         # MCP Tools Info
#         with gr.Accordion("🛠️ Available MCP Tools", open=False):
#             gr.Markdown("""
# **Core Analysis Tools:**
# - `analyze_ledger_from_csv` - Process and index financial data
# - `get_spending_by_category` - Category-based spending breakdown  
# - `get_vendor_analysis` - Vendor spending patterns
# - `query_financial_data` - Natural language financial queries

# **Integration:** These tools are available for external AI agents via MCP protocol.
#             """)
        
        # Enhanced dual file processing handler
        def process_dual_upload(image_file, csv_file):
            """Process both image and CSV files and reconcile transactions"""
            try:
                result = process_image_and_reconcile(image_file, csv_file)
                
                if result.get("status") == "success":
                    reconciliation = result["reconciliation"]
                    summary = reconciliation["summary"]
                    
                    # Create status message
                    status = f"""✅ Processing Complete!
                    
📊 **Processing Summary:**
• CSV Transactions: {result['csv_transactions']}
• Image Transactions: {result['image_transactions']} 
• Total Matches: {summary['total_matches']} ({summary['match_rate']}% match rate)

🎯 **Confidence Breakdown:**
• High Confidence (≥85%): {summary['high_confidence_matches']} transactions
• Medium Confidence (65-84%): {summary['medium_confidence_matches']} transactions  
• Low Confidence (<65%): {summary['low_confidence_matches']} transactions

💰 **Financial Summary:**
• Total Image Amount: ${summary['total_image_amount']}
• Total Matched Amount: ${summary['total_matched_amount']} ({summary['reconciliation_percentage']}%)
• Unmatched Image Transactions: {summary['unmatched_image_transactions']}

Session ID: {result['session_id'][:20]}..."""

                    # Create summary dataframe
                    summary_data = pd.DataFrame([{
                        "Metric": "CSV Transactions",
                        "Value": summary["total_csv_transactions"]
                    }, {
                        "Metric": "Image Transactions", 
                        "Value": summary["total_image_transactions"]
                    }, {
                        "Metric": "Total Matches",
                        "Value": f"{summary['total_matches']} ({summary['match_rate']}%)"
                    }, {
                        "Metric": "High Confidence Matches",
                        "Value": summary["high_confidence_matches"]
                    }, {
                        "Metric": "Medium Confidence Matches", 
                        "Value": summary["medium_confidence_matches"]
                    }, {
                        "Metric": "Low Confidence Matches",
                        "Value": summary["low_confidence_matches"]
                    }, {
                        "Metric": "Match Rate",
                        "Value": f"{summary['match_rate']}%"
                    }, {
                        "Metric": "Reconciliation %",
                        "Value": f"{summary['reconciliation_percentage']}%"
                    }])
                    
                    # Create match dataframes
                    def format_matches(matches):
                        if not matches:
                            return pd.DataFrame({"Message": ["No matches in this category"]})
                        
                        formatted = []
                        for match in matches:
                            csv_txn = match["csv_transaction"]
                            img_txn = match["image_transaction"]
                            
                            formatted.append({
                                "Confidence": f"{match['confidence_score']*100:.1f}%",
                                "CSV Date": csv_txn.get("date", ""),
                                "CSV Vendor": csv_txn.get("vendor", ""),
                                "CSV Amount": f"${csv_txn.get('amount', 0):.2f}",
                                "Image Date": img_txn.get("date", ""),
                                "Image Vendor": img_txn.get("vendor", ""),
                                "Image Amount": f"${img_txn.get('amount', 0):.2f}",
                                "Match Reasons": ", ".join(match.get("match_reasons", [])),
                                "Discrepancies": ", ".join(match.get("discrepancies", []))
                            })
                        return pd.DataFrame(formatted)
                    
                    high_conf_df = format_matches(reconciliation["matches"]["high_confidence"])
                    med_conf_df = format_matches(reconciliation["matches"]["medium_confidence"])
                    low_conf_df = format_matches(reconciliation["matches"]["low_confidence"])
                    
                    # Create unmatched dataframes
                    unmatched_data = []
                    for txn in reconciliation["unmatched"]["image_transactions"]:
                        unmatched_data.append({
                            "Source": "Image (Unmatched)",
                            "Date": txn.get("date", ""),
                            "Vendor": txn.get("vendor", ""),
                            "Amount": f"${txn.get('amount', 0):.2f}",
                            "Description": txn.get("description", "")
                        })
                    for txn in reconciliation["unmatched"]["csv_transactions"]:
                        unmatched_data.append({
                            "Source": "CSV (Unmatched)",
                            "Date": txn.get("date", ""),
                            "Vendor": txn.get("vendor", ""),
                            "Amount": f"${txn.get('amount', 0):.2f}",
                            "Description": txn.get("description", "")
                        })
                    
                    unmatched_df = pd.DataFrame(unmatched_data) if unmatched_data else pd.DataFrame({"Message": ["No unmatched transactions"]})
                    
                    return status, summary_data, high_conf_df, med_conf_df, low_conf_df, unmatched_df
                    
                else:
                    error_msg = f"❌ Processing Failed: {result.get('error', 'Unknown error')}"
                    empty_df = pd.DataFrame({"Error": [result.get('error', 'Unknown error')]})
                    return error_msg, empty_df, empty_df, empty_df, empty_df, empty_df
                    
            except Exception as e:
                error_msg = f"❌ Error during processing: {str(e)}"
                empty_df = pd.DataFrame({"Error": [str(e)]})
                return error_msg, empty_df, empty_df, empty_df, empty_df, empty_df
        
        # Event handlers
        reconcile_btn.click(
            fn=process_dual_upload,
            inputs=[image_file, csv_file],
            outputs=[reconciliation_status, summary_dataframe, high_confidence_dataframe, 
                    medium_confidence_dataframe, low_confidence_dataframe, unmatched_dataframe]
        )
        
#         # Quick Test Section with sample data
#         gr.Markdown("## 🎯 Quick Test")
#         sample_btn = gr.Button("📄 Load Sample Data (CSV + Mock Image)", variant="secondary", size="lg")
        
#         def load_sample_for_reconciliation():
#             """Load sample data and create a mock reconciliation scenario"""
#             try:
                
#                 # Mock reconciliation result for demonstration
#                 status = """✅ Sample Data Loaded!

# 📊 **Processing Summary:**
# • CSV Transactions: 5
# • Image Transactions: 3 (simulated)
# • Total Matches: 2 (66.7% match rate)

# 🎯 **Confidence Breakdown:**
# • High Confidence (≥85%): 1 transactions
# • Medium Confidence (65-84%): 1 transactions
# • Low Confidence (<65%): 0 transactions

# 💰 **Financial Summary:**
# • Total Image Amount: $68.49
# • Total Matched Amount: $49.50 (72.3%)
# • Unmatched Image Transactions: 1

# 🧪 This is sample data for demonstration purposes."""

#                 # Create sample summary
#                 summary_data = pd.DataFrame([
#                     {"Metric": "CSV Transactions", "Value": 5},
#                     {"Metric": "Image Transactions", "Value": 3},
#                     {"Metric": "Total Matches", "Value": "2 (66.7%)"},
#                     {"Metric": "High Confidence Matches", "Value": 1},
#                     {"Metric": "Medium Confidence Matches", "Value": 1},
#                     {"Metric": "Low Confidence Matches", "Value": 0},
#                     {"Metric": "Match Rate", "Value": "66.7%"},
#                     {"Metric": "Reconciliation %", "Value": "72.3%"}
#                 ])
                
#                 # Sample high confidence match
#                 high_conf = pd.DataFrame([{
#                     "Confidence": "92.5%",
#                     "CSV Date": "2024-01-15",
#                     "CSV Vendor": "Coffee Shop Downtown",
#                     "CSV Amount": "$4.50",
#                     "Image Date": "2024-01-15", 
#                     "Image Vendor": "Coffee Shop",
#                     "Image Amount": "$4.50",
#                     "Match Reasons": "Exact amount match, Exact date match, Partial vendor match",
#                     "Discrepancies": ""
#                 }])
                
#                 # Sample medium confidence match
#                 med_conf = pd.DataFrame([{
#                     "Confidence": "78.0%",
#                     "CSV Date": "2024-01-16",
#                     "CSV Vendor": "Shell Gas Station", 
#                     "CSV Amount": "$45.00",
#                     "Image Date": "2024-01-16",
#                     "Image Vendor": "Shell",
#                     "Image Amount": "$45.00",
#                     "Match Reasons": "Exact amount match, Exact date match, Vendor keyword match",
#                     "Discrepancies": "Vendor difference: SHELL GAS STATION vs SHELL"
#                 }])
                
#                 # Empty low confidence 
#                 low_conf = pd.DataFrame({"Message": ["No matches in this category"]})
                
#                 # Sample unmatched
#                 unmatched = pd.DataFrame([
#                     {"Source": "Image (Unmatched)", "Date": "2024-01-17", "Vendor": "Amazon", "Amount": "$18.99", "Description": "Online purchase"},
#                     {"Source": "CSV (Unmatched)", "Date": "2024-01-17", "Vendor": "Office Depot", "Amount": "$23.99", "Description": "Printer paper"},
#                     {"Source": "CSV (Unmatched)", "Date": "2024-01-18", "Vendor": "Uber Technologies", "Amount": "$18.75", "Description": "Ride to meeting"},
#                     {"Source": "CSV (Unmatched)", "Date": "2024-01-19", "Vendor": "Microsoft Corporation", "Amount": "$99.99", "Description": "Office 365"}
#                 ])
                
#                 return status, summary_data, high_conf, med_conf, low_conf, unmatched
                
#             except Exception as e:
#                 error_msg = f"❌ Error loading sample data: {str(e)}"
#                 empty_df = pd.DataFrame({"Error": [str(e)]})
#                 return error_msg, empty_df, empty_df, empty_df, empty_df, empty_df
        
#         sample_btn.click(
#             fn=load_sample_for_reconciliation,
#             outputs=[reconciliation_status, summary_dataframe, high_confidence_dataframe,
#                     medium_confidence_dataframe, low_confidence_dataframe, unmatched_dataframe]
#         )
        
#         # AI Analysis handler
#         def run_query(question):
#             """Run financial query and return formatted results"""
#             if not question.strip():
#                 return "Please enter a question about your financial data."
            
#             result = query_financial_data(question)
#             if result.get("status") == "success":
#                 return f"💡 QUESTION: {result['question']}\n\n📊 INSIGHTS:\n{result['insights']}"
#             else:
#                 return f"❌ {result.get('error', 'Unknown error')}"
        
#         query_btn.click(
#             fn=run_query,
#             inputs=[question_input],
#             outputs=[llm_results]
#         )
        
#         # Auto-query on Enter
#         question_input.submit(
#             fn=run_query,
#             inputs=[question_input],
#             outputs=[llm_results]
#         )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True
    )