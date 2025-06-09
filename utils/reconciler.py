"""
Transaction reconciliation engine for matching image-extracted transactions with CSV ledger entries
Uses multiple criteria and confidence scoring to identify potential matches
"""

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import difflib
import re
from dataclasses import dataclass

@dataclass
class TransactionMatch:
    """Represents a potential match between two transactions"""
    csv_transaction: Dict[str, Any]
    image_transaction: Dict[str, Any]
    confidence_score: float
    match_reasons: List[str]
    discrepancies: List[str]
    match_type: str  # "exact", "high", "medium", "low", "no_match"

class TransactionReconciler:
    """
    Reconciles transactions from image extraction with CSV ledger entries
    """
    
    def __init__(self, 
                 amount_tolerance: float = 0.01,
                 date_window_days: int = 3,
                 vendor_similarity_threshold: float = 0.6,
                 high_confidence_threshold: float = 0.85,
                 medium_confidence_threshold: float = 0.65):
        """
        Initialize the reconciler with matching parameters
        
        Args:
            amount_tolerance: Acceptable difference in amounts for matching
            date_window_days: Days before/after to consider for date matching
            vendor_similarity_threshold: Minimum similarity for vendor matching
            high_confidence_threshold: Minimum score for high confidence matches
            medium_confidence_threshold: Minimum score for medium confidence matches
        """
        self.amount_tolerance = amount_tolerance
        self.date_window_days = date_window_days
        self.vendor_similarity_threshold = vendor_similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
    
    def reconcile_transactions(self, 
                             csv_transactions: List[Dict[str, Any]], 
                             image_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main reconciliation function that matches image transactions with CSV entries
        
        Args:
            csv_transactions: List of transactions from CSV ledger
            image_transactions: List of transactions extracted from image
            
        Returns:
            Dictionary containing matches, unmatched transactions, and summary statistics
        """
        try:
            print(f"🔄 Reconciling {len(image_transactions)} image transactions with {len(csv_transactions)} CSV transactions")
            
            # Prepare transactions for matching
            csv_prepared = [self._prepare_transaction(txn, "csv") for txn in csv_transactions]
            image_prepared = [self._prepare_transaction(txn, "image") for txn in image_transactions]
            
            # Find all potential matches
            all_matches = []
            matched_csv_indices = set()
            matched_image_indices = set()
            
            for i, image_txn in enumerate(image_prepared):
                best_matches = []
                
                for j, csv_txn in enumerate(csv_prepared):
                    if j in matched_csv_indices:
                        continue
                    
                    match = self._evaluate_match(csv_txn, image_txn)
                    if match.confidence_score > 0:
                        best_matches.append((j, match))
                
                # Sort by confidence and take the best match if above threshold
                best_matches.sort(key=lambda x: x[1].confidence_score, reverse=True)
                
                if best_matches and best_matches[0][1].confidence_score >= 0.3:  # Minimum threshold
                    csv_idx, best_match = best_matches[0]
                    all_matches.append(best_match)
                    matched_csv_indices.add(csv_idx)
                    matched_image_indices.add(i)
            
            # Categorize matches by confidence
            high_confidence_matches = [m for m in all_matches if m.confidence_score >= self.high_confidence_threshold]
            medium_confidence_matches = [m for m in all_matches if self.medium_confidence_threshold <= m.confidence_score < self.high_confidence_threshold]
            low_confidence_matches = [m for m in all_matches if m.confidence_score < self.medium_confidence_threshold]
            
            # Find unmatched transactions
            unmatched_csv = [csv_prepared[i] for i in range(len(csv_prepared)) if i not in matched_csv_indices]
            unmatched_image = [image_prepared[i] for i in range(len(image_prepared)) if i not in matched_image_indices]
            
            # Calculate summary statistics
            total_image_amount = sum(float(txn.get('amount', 0)) for txn in image_prepared)
            total_matched_amount = sum(float(match.image_transaction.get('amount', 0)) for match in all_matches)
            match_rate = len(all_matches) / len(image_prepared) if image_prepared else 0
            
            result = {
                "status": "success",
                "summary": {
                    "total_image_transactions": len(image_transactions),
                    "total_csv_transactions": len(csv_transactions),
                    "total_matches": len(all_matches),
                    "match_rate": round(match_rate * 100, 1),
                    "high_confidence_matches": len(high_confidence_matches),
                    "medium_confidence_matches": len(medium_confidence_matches),
                    "low_confidence_matches": len(low_confidence_matches),
                    "unmatched_image_transactions": len(unmatched_image),
                    "unmatched_csv_transactions": len(unmatched_csv),
                    "total_image_amount": round(total_image_amount, 2),
                    "total_matched_amount": round(total_matched_amount, 2),
                    "reconciliation_percentage": round((total_matched_amount / total_image_amount * 100) if total_image_amount else 0, 1)
                },
                "matches": {
                    "high_confidence": [self._serialize_match(m) for m in high_confidence_matches],
                    "medium_confidence": [self._serialize_match(m) for m in medium_confidence_matches],
                    "low_confidence": [self._serialize_match(m) for m in low_confidence_matches]
                },
                "unmatched": {
                    "image_transactions": unmatched_image,
                    "csv_transactions": unmatched_csv
                },
                "reconciled_at": datetime.now().isoformat()
            }
            
            print(f"✅ Reconciliation complete: {len(all_matches)} matches found ({match_rate*100:.1f}% match rate)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during reconciliation: {e}")
            return {
                "status": "error",
                "error": str(e),
                "summary": {},
                "matches": {"high_confidence": [], "medium_confidence": [], "low_confidence": []},
                "unmatched": {"image_transactions": [], "csv_transactions": []},
                "reconciled_at": datetime.now().isoformat()
            }
    
    def _prepare_transaction(self, txn: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Prepare transaction data for matching"""
        prepared = txn.copy()
        prepared["source"] = source
        
        # Standardize date format
        if "date" in prepared:
            try:
                if isinstance(prepared["date"], str):
                    prepared["date"] = datetime.fromisoformat(prepared["date"].replace("Z", "+00:00"))
                elif not isinstance(prepared["date"], datetime):
                    prepared["date"] = datetime.strptime(str(prepared["date"]), "%Y-%m-%d")
            except:
                prepared["date"] = None
        
        # Standardize amount
        if "amount" in prepared:
            try:
                prepared["amount"] = float(prepared["amount"])
            except:
                prepared["amount"] = 0.0
        
        # Clean and standardize vendor/description
        for field in ["vendor", "description"]:
            if field in prepared and prepared[field]:
                prepared[field] = self._clean_text(str(prepared[field]))
        
        return prepared
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better matching"""
        # Remove extra whitespace, standardize case
        text = re.sub(r'\s+', ' ', text.strip().upper())
        
        # Remove common business suffixes
        suffixes = ["INC", "LLC", "LTD", "CORP", "CO", "&", "AND", "THE"]
        words = text.split()
        cleaned_words = [w for w in words if w not in suffixes]
        
        return " ".join(cleaned_words)
    
    def _evaluate_match(self, csv_txn: Dict[str, Any], image_txn: Dict[str, Any]) -> TransactionMatch:
        """Evaluate the match between two transactions"""
        confidence_score = 0.0
        match_reasons = []
        discrepancies = []
        
        # Amount matching (40% weight)
        amount_score = self._compare_amounts(csv_txn.get("amount", 0), image_txn.get("amount", 0))
        confidence_score += amount_score * 0.4
        if amount_score > 0.8:
            match_reasons.append(f"Amount match: ${csv_txn.get('amount', 0)} ≈ ${image_txn.get('amount', 0)}")
        elif amount_score < 0.5:
            discrepancies.append(f"Amount difference: ${csv_txn.get('amount', 0)} vs ${image_txn.get('amount', 0)}")
        
        # Date matching (25% weight)
        date_score = self._compare_dates(csv_txn.get("date"), image_txn.get("date"))
        confidence_score += date_score * 0.25
        if date_score > 0.8:
            match_reasons.append(f"Date match: {csv_txn.get('date')} ≈ {image_txn.get('date')}")
        elif date_score < 0.5:
            discrepancies.append(f"Date difference: {csv_txn.get('date')} vs {image_txn.get('date')}")
        
        # Vendor matching (25% weight)
        vendor_score = self._compare_vendors(
            csv_txn.get("vendor", ""), 
            image_txn.get("vendor", "")
        )
        confidence_score += vendor_score * 0.25
        if vendor_score > 0.7:
            match_reasons.append(f"Vendor match: '{csv_txn.get('vendor', '')}' ≈ '{image_txn.get('vendor', '')}'")
        elif vendor_score < 0.3:
            discrepancies.append(f"Vendor difference: '{csv_txn.get('vendor', '')}' vs '{image_txn.get('vendor', '')}'")
        
        # Description matching (10% weight)
        desc_score = self._compare_descriptions(
            csv_txn.get("description", ""), 
            image_txn.get("description", "")
        )
        confidence_score += desc_score * 0.1
        if desc_score > 0.7:
            match_reasons.append(f"Description similarity: '{csv_txn.get('description', '')}' ≈ '{image_txn.get('description', '')}'")
        
        # Determine match type
        if confidence_score >= self.high_confidence_threshold:
            match_type = "high"
        elif confidence_score >= self.medium_confidence_threshold:
            match_type = "medium"
        elif confidence_score >= 0.3:
            match_type = "low"
        else:
            match_type = "no_match"
        
        return TransactionMatch(
            csv_transaction=csv_txn,
            image_transaction=image_txn,
            confidence_score=round(confidence_score, 3),
            match_reasons=match_reasons,
            discrepancies=discrepancies,
            match_type=match_type
        )
    
    def _compare_amounts(self, amount1: float, amount2: float) -> float:
        """Compare transaction amounts with tolerance"""
        try:
            amount1, amount2 = float(amount1), float(amount2)
            
            # Exact match
            if abs(amount1 - amount2) <= self.amount_tolerance:
                return 1.0
            
            # Close match with scaling tolerance
            diff = abs(amount1 - amount2)
            avg_amount = (abs(amount1) + abs(amount2)) / 2
            
            if avg_amount == 0:
                return 0.0
            
            # Scale tolerance based on amount size
            scaled_tolerance = max(self.amount_tolerance, avg_amount * 0.02)  # 2% tolerance
            
            if diff <= scaled_tolerance:
                return 0.9
            elif diff <= scaled_tolerance * 3:
                return 0.7
            elif diff <= scaled_tolerance * 5:
                return 0.5
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _compare_dates(self, date1: datetime, date2: datetime) -> float:
        """Compare transaction dates with window tolerance"""
        try:
            if not date1 or not date2:
                return 0.0
            
            # Ensure both are datetime objects
            if isinstance(date1, str):
                date1 = datetime.fromisoformat(date1.replace("Z", "+00:00"))
            if isinstance(date2, str):
                date2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))
            
            diff_days = abs((date1 - date2).days)
            
            if diff_days == 0:
                return 1.0
            elif diff_days <= self.date_window_days:
                return 1.0 - (diff_days / self.date_window_days) * 0.3  # Linear decay
            elif diff_days <= self.date_window_days * 2:
                return 0.5
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _compare_vendors(self, vendor1: str, vendor2: str) -> float:
        """Compare vendor names using fuzzy matching"""
        try:
            if not vendor1 or not vendor2:
                return 0.0
            
            vendor1 = self._clean_text(vendor1)
            vendor2 = self._clean_text(vendor2)
            
            # Exact match
            if vendor1 == vendor2:
                return 1.0
            
            # Check if one contains the other
            if vendor1 in vendor2 or vendor2 in vendor1:
                return 0.9
            
            # Use sequence matching
            similarity = difflib.SequenceMatcher(None, vendor1, vendor2).ratio()
            
            # Check for partial word matches
            words1 = set(vendor1.split())
            words2 = set(vendor2.split())
            
            if words1 and words2:
                word_intersection = len(words1.intersection(words2))
                word_union = len(words1.union(words2))
                word_similarity = word_intersection / word_union if word_union > 0 else 0
                
                # Take the maximum of sequence similarity and word similarity
                similarity = max(similarity, word_similarity)
            
            return similarity
            
        except:
            return 0.0
    
    def _compare_descriptions(self, desc1: str, desc2: str) -> float:
        """Compare transaction descriptions"""
        try:
            if not desc1 or not desc2:
                return 0.0
            
            desc1 = self._clean_text(desc1)
            desc2 = self._clean_text(desc2)
            
            if desc1 == desc2:
                return 1.0
            
            # Use sequence matching
            similarity = difflib.SequenceMatcher(None, desc1, desc2).ratio()
            
            # Check for keyword matches
            words1 = set(desc1.split())
            words2 = set(desc2.split())
            
            if words1 and words2:
                word_intersection = len(words1.intersection(words2))
                if word_intersection > 0:
                    similarity = max(similarity, word_intersection / max(len(words1), len(words2)))
            
            return similarity
            
        except:
            return 0.0
    
    def _serialize_match(self, match: TransactionMatch) -> Dict[str, Any]:
        """Convert TransactionMatch to serializable dictionary"""
        return {
            "csv_transaction": match.csv_transaction,
            "image_transaction": match.image_transaction,
            "confidence_score": match.confidence_score,
            "match_reasons": match.match_reasons,
            "discrepancies": match.discrepancies,
            "match_type": match.match_type,
            "recommendation": self._get_recommendation(match)
        }
    
    def _get_recommendation(self, match: TransactionMatch) -> str:
        """Get recommendation for user action based on match confidence"""
        if match.confidence_score >= self.high_confidence_threshold:
            return "Auto-approve: High confidence match"
        elif match.confidence_score >= self.medium_confidence_threshold:
            return "Review recommended: Medium confidence match"
        else:
            return "Manual review required: Low confidence match"


def test_reconciler():
    """Test function for the reconciler"""
    # Sample test data
    csv_transactions = [
        {"date": "2024-01-15", "amount": 4.50, "vendor": "Coffee Shop Downtown", "description": "Morning coffee"},
        {"date": "2024-01-16", "amount": 45.00, "vendor": "Shell Gas Station", "description": "Fuel"},
        {"date": "2024-01-17", "amount": 23.99, "vendor": "Office Depot", "description": "Supplies"}
    ]
    
    image_transactions = [
        {"date": "2024-01-15", "amount": 4.50, "vendor": "Coffee Shop", "description": "Coffee"},
        {"date": "2024-01-16", "amount": 45.00, "vendor": "Shell", "description": "Gas"},
        {"date": "2024-01-18", "amount": 12.99, "vendor": "Amazon", "description": "Online purchase"}
    ]
    
    reconciler = TransactionReconciler()
    result = reconciler.reconcile_transactions(csv_transactions, image_transactions)
    
    print("✅ Reconciler test completed")
    print(f"Matches found: {result['summary']['total_matches']}")
    print(f"Match rate: {result['summary']['match_rate']}%")


if __name__ == "__main__":
    test_reconciler()