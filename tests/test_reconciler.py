"""
Test cases for SmartLedger reconciliation functionality
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta
from utils.reconciler import TransactionReconciler, MatchCriteria
from utils.data_models import Transaction, LedgerEntry

class TestTransactionReconciler:
    
    def setup_method(self):
        """Setup test data for each test"""
        self.reconciler = TransactionReconciler()
        
        # Sample ledger data
        self.sample_ledger_data = [
            {
                'date': '2024-01-15',
                'vendor': 'Coffee Shop Downtown',
                'amount': 4.50,
                'category': 'Meals & Entertainment',
                'description': 'Morning coffee'
            },
            {
                'date': '2024-01-16', 
                'vendor': 'Shell Gas Station',
                'amount': 45.00,
                'category': 'Vehicle Expenses',
                'description': 'Fuel'
            },
            {
                'date': '2024-01-17',
                'vendor': 'Office Depot',
                'amount': 23.99,
                'category': 'Office Supplies',
                'description': 'Printer paper'
            }
        ]
        
        # Create DataFrame and load into reconciler
        self.ledger_df = pd.DataFrame(self.sample_ledger_data)
        self.reconciler.load_ledger(self.ledger_df)
    
    def test_exact_match_found(self):
        """Test that exact matches are found correctly"""
        # Transaction that should exactly match the coffee shop entry
        transactions = [{
            'date': '2024-01-15',
            'vendor': 'Coffee Shop Downtown',
            'amount': 4.50,
            'description': 'Morning coffee',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(matched) == 1
        assert len(unmatched) == 0
        assert matched[0]['confidence'] == 1.0
        assert matched[0]['vendor'] == 'Coffee Shop Downtown'
    
    def test_amount_tolerance_match(self):
        """Test matching with amount tolerance"""
        # Transaction with slightly different amount (within tolerance)
        transactions = [{
            'date': '2024-01-15',
            'vendor': 'Coffee Shop Downtown', 
            'amount': 4.75,  # $0.25 difference, within $2 tolerance
            'description': 'Coffee',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(matched) == 1
        assert len(unmatched) == 0
        assert matched[0]['confidence'] == 1.0
    
    def test_date_tolerance_match(self):
        """Test matching with date tolerance"""
        # Transaction with date 3 days later (within 5 day tolerance)
        transactions = [{
            'date': '2024-01-18',  # 3 days after ledger entry
            'vendor': 'Coffee Shop Downtown',
            'amount': 4.50,
            'description': 'Coffee',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(matched) == 1
        assert len(unmatched) == 0
    
    def test_vendor_similarity_matching(self):
        """Test fuzzy matching based on vendor similarity"""
        # Similar vendor name but not exact
        transactions = [{
            'date': '2024-01-17',
            'vendor': 'Office Depot Store',  # Similar to "Office Depot"
            'amount': 23.99,
            'description': 'Office supplies',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        # Should find a fuzzy match
        assert len(matched) == 1
        assert matched[0]['confidence'] < 1.0  # Fuzzy match
        assert matched[0]['suggested_category'] == 'Office Supplies'
    
    def test_no_match_found(self):
        """Test transaction that should not match anything"""
        transactions = [{
            'date': '2024-02-01',
            'vendor': 'Unknown Restaurant',
            'amount': 99.99,
            'description': 'Expensive dinner',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(matched) == 0
        assert len(unmatched) == 1
        assert 'suggested_categories' in unmatched[0]
    
    def test_category_suggestions(self):
        """Test that category suggestions are generated for unmatched transactions"""
        transactions = [{
            'date': '2024-01-20',
            'vendor': 'McDonald\'s Restaurant',
            'amount': 12.50,
            'description': 'Fast food',
            'category': None
        }]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(unmatched) == 1
        suggested_cats = unmatched[0].get('suggested_categories', [])
        assert 'Meals & Entertainment' in suggested_cats
    
    def test_error_handling(self):
        """Test handling of malformed transaction data"""
        transactions = [
            {'vendor': 'Test', 'amount': 'invalid'},  # Invalid amount
            {'amount': 10.0},  # Missing vendor
            {'error': 'Processing failed'}  # Error transaction
        ]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        
        assert len(matched) == 0
        assert len(unmatched) == 3  # All should be unmatched
        
        # Check that error transactions are preserved
        error_trans = [t for t in unmatched if 'error' in t]
        assert len(error_trans) >= 1
    
    def test_vendor_name_normalization(self):
        """Test vendor name normalization"""
        # Test various vendor name formats
        test_cases = [
            ('Coffee Shop LLC', 'coffee shop'),
            ('Office Depot, Inc.', 'office depot'),
            ('Shell Gas & Food', 'shell gas'),
            ('Starbucks Coffee Company', 'starbucks coffee')
        ]
        
        for original, expected in test_cases:
            normalized = self.reconciler._normalize_vendor_name(original)
            assert expected in normalized.lower()
    
    def test_match_confidence_calculation(self):
        """Test confidence score calculation"""
        # Create a transaction similar to existing ledger entry
        transaction = Transaction(
            date=date(2024, 1, 15),
            vendor='Coffee Shop',  # Similar but not exact
            amount=4.50,
            description='Coffee purchase'
        )
        
        ledger_entry = LedgerEntry(
            date=date(2024, 1, 15),
            vendor='Coffee Shop Downtown',
            amount=4.50,
            category='Meals & Entertainment'
        )
        
        confidence = self.reconciler._calculate_match_confidence(transaction, ledger_entry)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_reconciliation_stats(self):
        """Test reconciliation statistics generation"""
        transactions = [
            {
                'date': '2024-01-15',
                'vendor': 'Coffee Shop Downtown',
                'amount': 4.50,
                'description': 'Coffee'
            },
            {
                'date': '2024-02-01', 
                'vendor': 'Unknown Vendor',
                'amount': 50.00,
                'description': 'Unknown expense'
            }
        ]
        
        matched, unmatched = self.reconciler.reconcile_transactions(transactions)
        stats = self.reconciler.get_reconciliation_stats(matched, unmatched)
        
        assert stats['total_transactions'] == 2
        assert stats['matched_count'] == 1
        assert stats['unmatched_count'] == 1
        assert stats['match_rate'] == 50.0
        assert stats['total_amount'] == 54.50
        assert stats['matched_amount'] == 4.50

class TestMatchCriteria:
    
    def test_custom_match_criteria(self):
        """Test reconciler with custom matching criteria"""
        strict_criteria = MatchCriteria(
            amount_tolerance=0.50,  # Stricter amount tolerance
            date_tolerance_days=2,   # Stricter date tolerance
            vendor_similarity_threshold=0.9  # Higher similarity requirement
        )
        
        reconciler = TransactionReconciler(strict_criteria)
        
        # Test that strict criteria result in fewer matches
        ledger_data = [{
            'date': '2024-01-15',
            'vendor': 'Coffee Shop',
            'amount': 4.50,
            'category': 'Meals'
        }]
        
        ledger_df = pd.DataFrame(ledger_data)
        reconciler.load_ledger(ledger_df)
        
        # Transaction that would match with default criteria but not strict
        transactions = [{
            'date': '2024-01-18',  # 3 days difference (over strict limit)
            'vendor': 'Coffee Shop',
            'amount': 4.50
        }]
        
        matched, unmatched = reconciler.reconcile_transactions(transactions)
        
        # With strict date tolerance, this should not match
        assert len(unmatched) >= 1

def test_sample_data_loading():
    """Test loading sample data files"""
    # This test ensures sample CSV files can be loaded
    sample_ledger_path = "tests/sample_data/sample_ledger.csv"
    
    try:
        df = pd.read_csv(sample_ledger_path)
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'vendor' in df.columns
        assert 'amount' in df.columns
    except FileNotFoundError:
        pytest.skip("Sample data file not found - run with sample data generation")

if __name__ == "__main__":
    pytest.main([__file__])