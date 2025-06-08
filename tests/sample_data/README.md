# Sample Data for SmartLedger Testing

This directory contains sample data files for testing and demonstrating SmartLedger functionality.

## Files Included

### sample_ledger.csv
A sample business ledger with 30 realistic transactions covering common business expense categories:
- Meals & Entertainment
- Vehicle Expenses  
- Office Supplies
- Travel
- Software & Technology
- Professional Services
- Utilities

The data spans January-February 2024 and includes vendors like:
- Coffee shops (Starbucks, Coffee Shop Downtown)
- Gas stations (Shell, Chevron, Tesla Supercharger)
- Restaurants (McDonald's, Chipotle, Olive Garden)
- Tech companies (Microsoft, Adobe, GitHub)
- Travel services (Uber, Delta, Marriott)
- Office suppliers (Office Depot, Amazon, Best Buy)

### Usage in Testing

```python
import pandas as pd

# Load sample ledger
df = pd.read_csv('tests/sample_data/sample_ledger.csv')

# Use with reconciler
from utils.reconciler import TransactionReconciler
reconciler = TransactionReconciler()
reconciler.load_ledger(df)
```

## Demo Transactions

You can create test transactions that should match entries in the sample ledger:

```python
test_transactions = [
    {
        'date': '2024-01-15',
        'vendor': 'Coffee Shop Downtown',
        'amount': 4.50,
        'description': 'Morning coffee'
    },
    {
        'date': '2024-01-17', 
        'vendor': 'Office Depot Store',  # Similar but not exact
        'amount': 23.99,
        'description': 'Office supplies'
    }
]
```

## Sample Receipt Images

For testing document processing, you would typically include sample receipt images in formats like:
- `receipt_coffee_shop.jpg`
- `receipt_gas_station.png` 
- `invoice_software_subscription.pdf`

Note: Actual image files are not included in this text-based setup, but should be added for full testing.

## Categories Used

The sample data uses these business expense categories:
- **Meals & Entertainment**: Restaurant meals, coffee, client entertainment
- **Vehicle Expenses**: Gas, charging, car rentals
- **Office Supplies**: Equipment, furniture, supplies
- **Travel**: Hotels, flights, ride-sharing
- **Software & Technology**: Subscriptions, cloud services
- **Professional Services**: Consulting, printing, memberships
- **Utilities**: Phone, internet service

This categorization follows common business accounting practices and helps test the reconciliation engine's category suggestion capabilities.