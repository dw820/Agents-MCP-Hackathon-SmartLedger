#!/usr/bin/env python3
"""
SmartLedger MCP Server
A standalone MCP server that exposes SmartLedger functionality to AI agents
"""

from mcp.server.fastmcp import FastMCP
from gradio_client import Client
from typing import List, Dict, Optional
import base64
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("SmartLedger")

# Global client storage
clients = {}

def get_gradio_client(space_url: str = "http://localhost:7860") -> Client:
    """Get or create a Gradio client for SmartLedger app"""
    if space_url not in clients:
        clients[space_url] = Client(space_url)
    return clients[space_url]

@mcp.tool()
async def reconcile_transactions(
    document_urls: List[str], 
    ledger_csv_content: str,
    space_url: str = "http://localhost:7860"
) -> Dict:
    """
    Reconcile uploaded documents against ledger CSV using SmartLedger.
    
    This tool processes receipt/invoice images and matches transactions against 
    an existing accounting ledger, providing smart categorization and confidence scoring.
    
    Args:
        document_urls: List of URLs to document images (receipts, invoices)
        ledger_csv_content: CSV content of existing ledger with columns: date,vendor,amount,category,description
        space_url: URL of the SmartLedger Gradio app (default: localhost:7860)
        
    Returns:
        Dict containing matched transactions, unmatched transactions, and export CSV
    """
    try:
        client = get_gradio_client(space_url)
        
        # Convert URLs to file objects if needed
        document_files = []
        for url in document_urls:
            # For this example, assume URLs are local file paths or base64 encoded
            if url.startswith('data:'):
                # Handle base64 encoded files
                header, data = url.split(',', 1)
                file_data = base64.b64decode(data)
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_data)
                    document_files.append(tmp_file.name)
            else:
                # Assume it's a file path
                document_files.append(url)
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
            csv_file.write(ledger_csv_content)
            csv_file_path = csv_file.name
        
        # Call Gradio interface
        result = client.predict(
            document_files,  # docs parameter
            csv_file_path,   # ledger_csv parameter
            api_name="/process_documents"
        )
        
        # Clean up temporary files
        for file_path in document_files:
            if file_path.startswith('/tmp/'):
                os.unlink(file_path)
        os.unlink(csv_file_path)
        
        # Parse results
        matched_df, unmatched_df, summary, export_file = result
        
        return {
            "matched_transactions": matched_df.to_dict('records') if matched_df is not None else [],
            "unmatched_transactions": unmatched_df.to_dict('records') if unmatched_df is not None else [],
            "summary": summary,
            "export_available": export_file is not None,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Reconciliation failed: {str(e)}",
            "status": "error"
        }

@mcp.tool()
async def query_transactions(
    query: str,
    space_url: str = "http://localhost:7860"
) -> str:
    """
    Query transaction data using natural language.
    
    Ask questions about your transactions in plain English and get AI-powered insights.
    Examples: "How much did I spend on office supplies?", "Show me all travel expenses over $100"
    
    Args:
        query: Natural language question about transactions
        space_url: URL of the SmartLedger Gradio app (default: localhost:7860)
        
    Returns:
        Natural language response with transaction insights
    """
    try:
        client = get_gradio_client(space_url)
        
        result = client.predict(
            query,
            api_name="/query_handler"
        )
        
        return result
        
    except Exception as e:
        return f"Query failed: {str(e)}"

@mcp.tool()
async def get_supported_formats() -> Dict[str, List[str]]:
    """
    Get information about supported file formats and CSV structure.
    
    Returns:
        Dictionary containing supported document formats and required CSV columns
    """
    return {
        "document_formats": [
            "PDF", "JPG", "JPEG", "PNG", "GIF", "WebP"
        ],
        "required_csv_columns": [
            "date", "vendor", "amount", "category", "description"
        ],
        "csv_example": {
            "date": "2024-01-15",
            "vendor": "Coffee Shop Downtown", 
            "amount": "4.50",
            "category": "Meals & Entertainment",
            "description": "Morning coffee"
        },
        "max_file_size_mb": 10,
        "batch_processing": True
    }

@mcp.tool()
async def health_check(space_url: str = "http://localhost:7860") -> Dict[str, str]:
    """
    Check if SmartLedger service is running and accessible.
    
    Args:
        space_url: URL of the SmartLedger Gradio app
        
    Returns:
        Health status information
    """
    try:
        client = get_gradio_client(space_url)
        
        # Try to connect to the Gradio app
        info = client.view_api()
        
        return {
            "status": "healthy",
            "service": "SmartLedger",
            "version": "1.0.0",
            "available_endpoints": len(info.get("named_endpoints", {})),
            "space_url": space_url
        }
        
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "space_url": space_url
        }

# Example usage and testing
@mcp.tool()
async def get_sample_data() -> Dict:
    """
    Get sample data for testing SmartLedger functionality.
    
    Returns:
        Sample ledger CSV content and example queries
    """
    sample_csv = """date,vendor,amount,category,description
2024-01-15,Coffee Shop Downtown,4.50,Meals & Entertainment,Morning coffee
2024-01-16,Shell Gas Station,45.00,Vehicle Expenses,Regular gasoline
2024-01-17,Office Depot,23.99,Office Supplies,Printer paper and pens
2024-01-18,Uber Technologies,18.75,Travel,Ride to client meeting
2024-01-19,Microsoft Corporation,99.99,Software & Technology,Office 365 subscription"""
    
    example_queries = [
        "How much did I spend on office supplies this month?",
        "Show me all travel expenses over $15",
        "What's my total spending on meals and entertainment?",
        "List all transactions from Coffee Shop Downtown"
    ]
    
    return {
        "sample_ledger_csv": sample_csv,
        "example_queries": example_queries,
        "usage_tips": [
            "Upload clear photos of receipts for best results",
            "Ensure CSV has all required columns: date,vendor,amount,category,description", 
            "Use natural language for queries - the AI understands context",
            "Check confidence scores for transaction matches"
        ]
    }

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()