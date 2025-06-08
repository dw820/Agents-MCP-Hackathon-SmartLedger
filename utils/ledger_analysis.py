"""
Ledger analysis utilities for SmartLedger
Contains core functions for CSV processing and analysis
"""

import pandas as pd
from typing import Optional, Tuple
import io

def process_ledger_csv(csv_file) -> Tuple[Optional[pd.DataFrame], str]:
    """Process uploaded CSV ledger file and return DataFrame and status message"""
    if csv_file is None:
        return None, "Please upload a CSV file"
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file.name)
        
        # Validate required columns
        required_columns = ['date', 'vendor', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Basic data validation
        if df.empty:
            return None, "CSV file is empty"
        
        # Convert amount to numeric if it's not already
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        except:
            return None, "Amount column contains invalid values"
        
        # Convert date column
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Check if any dates were successfully converted
            if df['date'].isna().all():
                return None, "No valid dates found in date column"
        except Exception as e:
            return None, f"Date column processing failed: {str(e)}"
        
        # Remove rows with invalid data
        initial_rows = len(df)
        df = df.dropna(subset=['date', 'vendor', 'amount'])
        
        if df.empty:
            return None, "No valid rows found after data cleaning"
        
        rows_removed = initial_rows - len(df)
        status_msg = f"✅ Successfully loaded {len(df)} transactions"
        if rows_removed > 0:
            status_msg += f" ({rows_removed} invalid rows removed)"
        
        return df, status_msg
        
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"

def analyze_ledger_data(df: pd.DataFrame) -> str:
    """Generate basic analysis of the ledger data"""
    if df is None or df.empty:
        return "No data to analyze"
    
    try:
        # Basic statistics
        total_transactions = len(df)
        total_amount = df['amount'].sum()
        avg_amount = df['amount'].mean()
        
        # Date range
        try:
            min_date = df['date'].min()
            max_date = df['date'].max()
            if hasattr(min_date, 'strftime'):
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{min_date} to {max_date}"
        except:
            date_range = "Date range unavailable"
        
        # Top vendors
        top_vendors = df['vendor'].value_counts().head(3)
        
        # Categories if available
        categories_info = ""
        if 'category' in df.columns:
            category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(3)
            categories_info = f"\n\n**Top Categories:**\n"
            for cat, amount in category_totals.items():
                categories_info += f"• {cat}: ${amount:,.2f}\n"
        
        analysis = f"""
**📊 Ledger Analysis:**

**Summary:**
• Total Transactions: {total_transactions:,}
• Total Amount: ${total_amount:,.2f}
• Average Transaction: ${avg_amount:.2f}
• Date Range: {date_range}

**Top Vendors:**
"""
        for vendor, count in top_vendors.items():
            vendor_total = df[df['vendor'] == vendor]['amount'].sum()
            analysis += f"• {vendor}: {count} transactions (${vendor_total:,.2f})\n"
        
        analysis += categories_info
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def load_sample_data() -> Tuple[str, pd.DataFrame, str]:
    """Load sample data for testing"""
    sample_csv = """date,vendor,amount,category,description
2024-01-15,Coffee Shop Downtown,4.50,Meals & Entertainment,Morning coffee
2024-01-16,Shell Gas Station,45.00,Vehicle Expenses,Regular gasoline
2024-01-17,Office Depot,23.99,Office Supplies,Printer paper
2024-01-18,Uber Technologies,18.75,Travel,Ride to meeting
2024-01-19,Microsoft Corporation,99.99,Software,Office 365 subscription"""
    
    # Create a temporary file-like object
    temp_file = io.StringIO(sample_csv)
    df = pd.read_csv(temp_file)
    
    # Process the data like we would for uploaded CSV
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    analysis = analyze_ledger_data(df)
    return "✅ Sample data loaded successfully", df, analysis

def handle_csv_upload(csv_file):
    """Handle CSV file upload and analysis"""
    df, status = process_ledger_csv(csv_file)
    
    if df is not None:
        analysis = analyze_ledger_data(df)
        return status, df, analysis
    else:
        return status, None, ""