"""
Core reconciliation logic for SmartLedger
Handles matching transactions against ledger entries
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from utils.data_models import Transaction, LedgerEntry, MatchedTransaction, UnmatchedTransaction

@dataclass
class MatchCriteria:
    """Configuration for transaction matching"""
    amount_tolerance: float = 2.0  # Dollar tolerance for amount matching
    date_tolerance_days: int = 5   # Days tolerance for date matching
    vendor_similarity_threshold: float = 0.7  # Similarity threshold for vendor matching
    category_confidence_threshold: float = 0.6  # Minimum confidence for category suggestions

class TransactionReconciler:
    """Core reconciliation engine for matching transactions"""
    
    def __init__(self, match_criteria: MatchCriteria = None):
        self.criteria = match_criteria or MatchCriteria()
        self.ledger_entries = []
        
    def load_ledger(self, ledger_df: pd.DataFrame):
        """Load ledger entries for reconciliation"""
        self.ledger_entries = []
        
        for _, row in ledger_df.iterrows():
            try:
                entry = LedgerEntry(
                    date=row['date'],
                    vendor=row['vendor'],
                    amount=float(row['amount']),
                    category=row.get('category', 'Unknown'),
                    description=row.get('description', ''),
                    account=row.get('account', '')
                )
                self.ledger_entries.append(entry)
            except Exception as e:
                print(f"Error loading ledger entry: {e}")
                continue
    
    def reconcile_transactions(self, transactions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Main reconciliation method"""
        matched = []
        unmatched = []
        
        for trans_data in transactions:
            # Skip error transactions
            if 'error' in trans_data:
                unmatched.append(trans_data)
                continue
                
            try:
                # Convert to Transaction model
                transaction = Transaction(**trans_data)
                
                # Attempt matching
                match_result = self._find_best_match(transaction)
                
                if match_result:
                    matched_trans = self._create_matched_transaction(transaction, match_result)
                    matched.append(matched_trans.dict())
                else:
                    unmatched_trans = self._create_unmatched_transaction(transaction)
                    unmatched.append(unmatched_trans.dict())
                    
            except Exception as e:
                # Add to unmatched with error info
                error_trans = trans_data.copy()
                error_trans['error'] = f"Processing error: {str(e)}"
                unmatched.append(error_trans)
        
        return matched, unmatched
    
    def _find_best_match(self, transaction: Transaction) -> Optional[Dict]:
        """Find the best matching ledger entry for a transaction"""
        # First try exact matching
        exact_match = self._find_exact_match(transaction)
        if exact_match:
            return {
                'type': 'exact',
                'entry': exact_match,
                'confidence': 1.0
            }
        
        # Then try fuzzy matching
        fuzzy_match = self._find_fuzzy_match(transaction)
        if fuzzy_match:
            return fuzzy_match
        
        return None
    
    def _find_exact_match(self, transaction: Transaction) -> Optional[LedgerEntry]:
        """Find exact match based on amount and date"""
        if not transaction.date:
            return None
            
        for entry in self.ledger_entries:
            # Check amount tolerance
            amount_diff = abs(entry.amount - transaction.amount)
            if amount_diff > self.criteria.amount_tolerance:
                continue
            
            # Check date tolerance
            date_diff = abs((entry.date - transaction.date).days)
            if date_diff > self.criteria.date_tolerance_days:
                continue
            
            # Check vendor similarity
            vendor_similarity = self._calculate_vendor_similarity(
                transaction.vendor, entry.vendor
            )
            
            if vendor_similarity > 0.8:  # High similarity threshold for exact match
                return entry
        
        return None
    
    def _find_fuzzy_match(self, transaction: Transaction) -> Optional[Dict]:
        """Find fuzzy match based on vendor similarity and category patterns"""
        best_match = None
        best_confidence = 0.0
        
        for entry in self.ledger_entries:
            confidence = self._calculate_match_confidence(transaction, entry)
            
            if confidence > self.criteria.category_confidence_threshold and confidence > best_confidence:
                best_match = entry
                best_confidence = confidence
        
        if best_match:
            return {
                'type': 'fuzzy',
                'entry': best_match,
                'confidence': best_confidence
            }
        
        return None
    
    def _calculate_match_confidence(self, transaction: Transaction, entry: LedgerEntry) -> float:
        """Calculate overall match confidence between transaction and ledger entry"""
        confidence_factors = []
        
        # Vendor similarity (40% weight)
        vendor_sim = self._calculate_vendor_similarity(transaction.vendor, entry.vendor)
        confidence_factors.append(vendor_sim * 0.4)
        
        # Amount proximity (30% weight)
        if transaction.amount > 0:
            amount_diff = abs(entry.amount - transaction.amount)
            max_diff = max(entry.amount, transaction.amount) * 0.2  # 20% tolerance
            amount_sim = max(0, 1 - (amount_diff / max_diff)) if max_diff > 0 else 0
            confidence_factors.append(amount_sim * 0.3)
        
        # Date proximity (20% weight) 
        if transaction.date:
            date_diff = abs((entry.date - transaction.date).days)
            date_sim = max(0, 1 - (date_diff / 30))  # 30 day scale
            confidence_factors.append(date_sim * 0.2)
        
        # Category/description similarity (10% weight)
        desc_sim = self._calculate_description_similarity(
            transaction.description or "", 
            entry.description or ""
        )
        confidence_factors.append(desc_sim * 0.1)
        
        return sum(confidence_factors)
    
    def _calculate_vendor_similarity(self, vendor1: str, vendor2: str) -> float:
        """Calculate similarity between vendor names"""
        if not vendor1 or not vendor2:
            return 0.0
        
        # Normalize vendor names
        v1 = self._normalize_vendor_name(vendor1)
        v2 = self._normalize_vendor_name(vendor2)
        
        # Exact match
        if v1 == v2:
            return 1.0
        
        # Contains check
        if v1 in v2 or v2 in v1:
            return 0.8
        
        # Jaccard similarity on words
        words1 = set(v1.split())
        words2 = set(v2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_vendor_name(self, vendor: str) -> str:
        """Normalize vendor name for comparison"""
        import re
        
        # Convert to lowercase
        vendor = vendor.lower().strip()
        
        # Remove common business suffixes
        suffixes = ['inc', 'llc', 'corp', 'ltd', 'co', 'company', '&', 'and']
        for suffix in suffixes:
            vendor = re.sub(rf'\b{suffix}\b', '', vendor)
        
        # Remove special characters
        vendor = re.sub(r'[^\w\s]', ' ', vendor)
        
        # Collapse whitespace
        vendor = ' '.join(vendor.split())
        
        return vendor
    
    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between descriptions"""
        if not desc1 or not desc2:
            return 0.0
        
        # Simple word overlap calculation
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_matched_transaction(self, transaction: Transaction, match_result: Dict) -> MatchedTransaction:
        """Create a matched transaction object"""
        entry = match_result['entry']
        confidence = match_result['confidence']
        match_type = match_result['type']
        
        return MatchedTransaction(
            date=transaction.date,
            vendor=transaction.vendor,
            amount=transaction.amount,
            description=transaction.description,
            category=transaction.category or entry.category,
            confidence=confidence,
            ledger_match=entry,
            suggested_category=entry.category if match_type == 'fuzzy' else None
        )
    
    def _create_unmatched_transaction(self, transaction: Transaction) -> UnmatchedTransaction:
        """Create an unmatched transaction with category suggestions"""
        # Get category suggestions based on vendor patterns
        suggested_categories = self._get_category_suggestions(transaction.vendor)
        
        return UnmatchedTransaction(
            date=transaction.date,
            vendor=transaction.vendor,
            amount=transaction.amount,
            description=transaction.description,
            category=transaction.category,
            suggested_categories=suggested_categories
        )
    
    def _get_category_suggestions(self, vendor: str) -> List[str]:
        """Get category suggestions based on vendor name patterns"""
        if not vendor:
            return []
        
        vendor_lower = vendor.lower()
        suggestions = []
        
        # Common vendor patterns
        category_patterns = {
            'Office Supplies': ['office', 'depot', 'staples', 'supplies', 'paper'],
            'Meals & Entertainment': ['restaurant', 'cafe', 'coffee', 'food', 'dining', 'pizza', 'burger'],
            'Travel': ['hotel', 'airline', 'airport', 'uber', 'lyft', 'rental', 'gas', 'fuel'],
            'Software & Technology': ['software', 'tech', 'microsoft', 'adobe', 'google', 'aws', 'cloud'],
            'Professional Services': ['consulting', 'legal', 'accounting', 'services', 'professional'],
            'Utilities': ['electric', 'power', 'water', 'internet', 'phone', 'utility'],
            'Vehicle Expenses': ['gas', 'fuel', 'auto', 'car', 'mechanic', 'repair', 'maintenance']
        }
        
        for category, keywords in category_patterns.items():
            if any(keyword in vendor_lower for keyword in keywords):
                suggestions.append(category)
        
        # Find similar vendors in ledger
        for entry in self.ledger_entries:
            vendor_sim = self._calculate_vendor_similarity(vendor, entry.vendor)
            if vendor_sim > 0.6 and entry.category not in suggestions:
                suggestions.append(entry.category)
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_reconciliation_stats(self, matched: List[Dict], unmatched: List[Dict]) -> Dict:
        """Generate reconciliation statistics"""
        total_transactions = len(matched) + len(unmatched)
        
        if total_transactions == 0:
            return {
                'total_transactions': 0,
                'matched_count': 0,
                'unmatched_count': 0,
                'match_rate': 0.0,
                'total_amount': 0.0,
                'matched_amount': 0.0
            }
        
        total_amount = sum(t.get('amount', 0) for t in matched + unmatched if 'error' not in t)
        matched_amount = sum(t.get('amount', 0) for t in matched if 'error' not in t)
        
        return {
            'total_transactions': total_transactions,
            'matched_count': len(matched),
            'unmatched_count': len(unmatched),
            'match_rate': (len(matched) / total_transactions) * 100,
            'total_amount': total_amount,
            'matched_amount': matched_amount,
            'average_confidence': sum(t.get('confidence', 0) for t in matched) / len(matched) if matched else 0
        }

# Utility functions

def create_reconciliation_summary(matched: List[Dict], unmatched: List[Dict]) -> str:
    """Create a human-readable reconciliation summary"""
    reconciler = TransactionReconciler()
    stats = reconciler.get_reconciliation_stats(matched, unmatched)
    
    summary = f"""
Reconciliation Summary:
- Total Transactions: {stats['total_transactions']}
- Matched: {stats['matched_count']} ({stats['match_rate']:.1f}%)
- Unmatched: {stats['unmatched_count']}
- Total Amount: ${stats['total_amount']:.2f}
- Matched Amount: ${stats['matched_amount']:.2f}
"""
    
    if stats['matched_count'] > 0:
        summary += f"- Average Match Confidence: {stats['average_confidence']:.2f}"
    
    return summary.strip()

def export_reconciliation_csv(matched: List[Dict], unmatched: List[Dict]) -> str:
    """Export reconciliation results as CSV string"""
    rows = []
    
    # Header
    rows.append("date,vendor,amount,category,status,confidence,suggested_action,notes")
    
    # Matched transactions
    for trans in matched:
        if 'error' in trans:
            continue
        
        date = trans.get('date', '')
        vendor = trans.get('vendor', '')
        amount = trans.get('amount', 0)
        category = trans.get('category', '')
        confidence = trans.get('confidence', 0)
        
        rows.append(f"{date},{vendor},{amount},{category},matched,{confidence:.2f},none,")
    
    # Unmatched transactions
    for trans in unmatched:
        if 'error' in trans:
            error_msg = trans.get('error', 'Unknown error')
            rows.append(f",,0,,error,0.0,review,{error_msg}")
            continue
        
        date = trans.get('date', '')
        vendor = trans.get('vendor', '')
        amount = trans.get('amount', 0)
        category = trans.get('category', 'Unknown')
        suggested_cats = trans.get('suggested_categories', [])
        
        suggested_action = f"create_entry"
        notes = f"Suggested categories: {', '.join(suggested_cats)}" if suggested_cats else ""
        
        rows.append(f"{date},{vendor},{amount},{category},unmatched,0.0,{suggested_action},{notes}")
    
    return "\n".join(rows)