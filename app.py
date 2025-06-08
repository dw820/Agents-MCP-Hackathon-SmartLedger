import gradio as gr
import pandas as pd
from utils.ledger_analysis import handle_csv_upload, load_sample_data, analyze_ledger_data
from typing import Dict
import io

def analyze_ledger_from_csv(csv_content: str) -> Dict:
    """
    Analyze a ledger CSV and return comprehensive insights about spending patterns.
    
    This function processes CSV data containing financial transactions and provides
    detailed analysis including spending summaries, top vendors, and category breakdowns.
    
    Args:
        csv_content: CSV content as string with required columns: date,vendor,amount
                    Optional columns: category,description
        
    Returns:
        Dictionary containing analysis results, statistics, and spending insights
    """
    try:
        # Parse CSV content
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Process the data
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'vendor', 'amount'])
        
        if df.empty:
            return {"error": "No valid transactions found"}
        
        # Generate analysis
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
            "analysis_text": analysis_text
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
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Quick Test")
                sample_btn = gr.Button("Load Sample Data", variant="secondary")
                
            with gr.Column():
                gr.Markdown("### 🔗 MCP Server")
                gr.Markdown("**Available MCP Tools:**\n- `analyze_ledger_from_csv`\n- `get_spending_by_category`\n- `get_vendor_analysis`")
        
        # Event handlers
        analyze_btn.click(
            fn=handle_csv_upload,
            inputs=[csv_file],
            outputs=[status_text, ledger_dataframe, analysis_text]
        )
        
        # Auto-analyze when file is uploaded
        csv_file.change(
            fn=handle_csv_upload,
            inputs=[csv_file],
            outputs=[status_text, ledger_dataframe, analysis_text]
        )
        
        # Sample data handler
        sample_btn.click(
            fn=load_sample_data,
            outputs=[status_text, ledger_dataframe, analysis_text]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True
    )