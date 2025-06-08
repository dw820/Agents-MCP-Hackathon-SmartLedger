import gradio as gr
import pandas as pd
from typing import List, Dict
import modal
from dotenv import load_dotenv
import os

load_dotenv()

from llamaindex_core import reconcile_with_llamaindex, query_transactions_mcp

# Internal reconciliation functions (not exposed to MCP)
def internal_reconcile_transactions(documents: List[bytes], ledger_csv: str) -> Dict:
    """Internal reconciliation function for Gradio interface"""
    try:
        # Process documents using Modal
        f = modal.Function.lookup("SmartLedger", "process_documents_modal")
        matched, unmatched, export_data = f.remote(documents, ledger_csv)
        
        return {
            "matched": matched,
            "unmatched": unmatched,
            "export_csv": export_data,
            "status": "success"
        }
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "status": "error"}

def internal_query_transactions(query: str) -> str:
    """Internal query function for Gradio interface"""
    try:
        response = query_transactions_mcp(query)
        return response
    except Exception as e:
        return f"Query failed: {str(e)}"

def process_documents(docs, ledger_file):
    """Main processing function for Gradio interface"""
    if not docs or not ledger_file:
        return None, None, "Please upload both documents and ledger CSV", None
    
    try:
        # Read ledger CSV
        ledger_content = ledger_file.read().decode('utf-8')
        
        # Read document files
        document_files = []
        for doc in docs:
            document_files.append(doc.read())
        
        # Call internal reconciliation function
        result = internal_reconcile_transactions(document_files, ledger_content)
        
        if "error" in result:
            return None, None, result["error"], None
        
        # Convert to DataFrames for display
        matched_df = pd.DataFrame(result["matched"]) if result["matched"] else pd.DataFrame()
        unmatched_df = pd.DataFrame(result["unmatched"]) if result["unmatched"] else pd.DataFrame()
        
        suggestions = f"Processed {len(result['matched'])} matched and {len(result['unmatched'])} unmatched transactions"
        
        # Create export file
        export_file = "reconciliation_results.csv"
        with open(export_file, 'w') as f:
            f.write(result["export_csv"])
        
        return matched_df, unmatched_df, suggestions, export_file
        
    except Exception as e:
        return None, None, f"Error processing documents: {str(e)}", None

def query_handler(query):
    """Handle natural language queries"""
    try:
        response = internal_query_transactions(query)
        return response
    except Exception as e:
        return f"Query failed: {str(e)}"

def create_app():
    with gr.Blocks(title="SmartLedger", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SmartLedger - Smart Business Accounting Reconciler")
        gr.Markdown("Upload your receipts and ledger CSV to automatically reconcile transactions using AI.")

        with gr.Tab("Upload & Process"):
            with gr.Row():
                with gr.Column():
                    docs = gr.File(
                        label="Upload Documents (Receipts, Invoices)", 
                        file_count="multiple",
                        file_types=["image", ".pdf"]
                    )
                    ledger_csv = gr.File(
                        label="Upload Ledger CSV",
                        file_types=[".csv"]
                    )
                    process_btn = gr.Button("Process & Reconcile", variant="primary")

                with gr.Column():
                    gr.Markdown("### Expected CSV Format:")
                    gr.Code("""date,vendor,amount,category,description
2024-01-15,Coffee Shop,4.50,Office Supplies,Morning coffee
2024-01-16,Gas Station,45.00,Vehicle Expenses,Fuel""", language="markdown")

            with gr.Row():
                matched_df = gr.Dataframe(
                    label="✅ Matched Transactions",
                    interactive=False
                )
                unmatched_df = gr.Dataframe(
                    label="❓ Unmatched Transactions", 
                    interactive=False
                )

            suggestions = gr.Textbox(
                label="Processing Summary",
                interactive=False,
                lines=3
            )
            
            export_csv = gr.File(
                label="📥 Download Reconciliation Results",
                interactive=False
            )

        with gr.Tab("Query & Analysis"):
            gr.Markdown("Ask questions about your transactions in natural language")
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="e.g., 'How much did I spend on office supplies this month?'",
                    lines=2
                )
                query_btn = gr.Button("Ask", variant="secondary")
            
            query_output = gr.Textbox(
                label="Analysis Result",
                lines=5,
                interactive=False
            )

        # Event handlers
        process_btn.click(
            fn=process_documents,
            inputs=[docs, ledger_csv],
            outputs=[matched_df, unmatched_df, suggestions, export_csv]
        )
        
        query_btn.click(
            fn=query_handler,
            inputs=[query_input],
            outputs=[query_output]
        )
        
        query_input.submit(
            fn=query_handler,
            inputs=[query_input],
            outputs=[query_output]
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )