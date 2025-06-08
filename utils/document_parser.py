"""
Document parsing utilities for SmartLedger
Handles vision model integration and document processing
"""

import base64
import io
from typing import List, Dict, Optional, Tuple
from PIL import Image
import json
import re
from datetime import datetime

class DocumentParser:
    """Handles parsing of receipts and invoices using vision models"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf']
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)
    
    def detect_file_type(self, file_bytes: bytes) -> str:
        """Detect file type from magic bytes"""
        if len(file_bytes) < 8:
            return "unknown"
        
        # Check magic bytes for common formats
        if file_bytes.startswith(b'\xff\xd8\xff'):
            return "jpeg"
        elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            return "png"
        elif file_bytes.startswith(b'GIF8'):
            return "gif"
        elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[:12]:
            return "webp"
        elif file_bytes.startswith(b'%PDF'):
            return "pdf"
        else:
            return "unknown"
    
    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """Preprocess image for better OCR results"""
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 2048px on longest side)
            max_size = 2048
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save processed image
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95)
            return output.getvalue()
            
        except Exception as e:
            print(f"Image preprocessing failed: {e}")
            return image_bytes
    
    def encode_image_base64(self, image_bytes: bytes) -> str:
        """Encode image as base64 for API calls"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def create_vision_prompt(self, extraction_type: str = "receipt") -> str:
        """Create optimized prompt for vision model"""
        if extraction_type == "receipt":
            return """
            Analyze this receipt/invoice image and extract transaction details.
            
            Return a JSON object with this exact structure:
            {
                "transactions": [
                    {
                        "date": "YYYY-MM-DD or null if unclear",
                        "vendor": "business name",
                        "amount": numeric_value,
                        "description": "items purchased or service description",
                        "category": "best guess category (Office Supplies, Meals, Travel, Utilities, etc.)"
                    }
                ]
            }
            
            Important guidelines:
            - If multiple line items exist, create separate transaction entries
            - Extract the TOTAL amount, not individual line items unless specified
            - Use null for date if it's unclear or unreadable
            - Provide your best guess for vendor name even if partially obscured
            - Categories should be business expense categories
            - Amount should be numeric (no currency symbols)
            - If you cannot read the image clearly, return an error in the description field
            """
        
        elif extraction_type == "invoice":
            return """
            Analyze this invoice image and extract transaction details.
            
            Return a JSON object with this exact structure:
            {
                "transactions": [
                    {
                        "date": "YYYY-MM-DD (invoice date)",
                        "vendor": "company/service provider name", 
                        "amount": total_amount_numeric,
                        "description": "services or products description",
                        "category": "business category (Professional Services, Software, Equipment, etc.)"
                    }
                ]
            }
            
            Focus on:
            - Invoice date (not due date)
            - Total amount due
            - Service provider name
            - Description of services/products
            """
        
        else:
            return """
            Extract transaction information from this document image.
            Return JSON with transaction details including date, vendor, amount, description, and category.
            """
    
    def parse_vision_response(self, response_text: str) -> List[Dict]:
        """Parse and validate vision model response"""
        try:
            # Try to parse JSON
            response_data = json.loads(response_text)
            
            # Extract transactions
            transactions = response_data.get("transactions", [])
            
            # Validate and clean transactions
            cleaned_transactions = []
            for trans in transactions:
                cleaned_trans = self._clean_transaction(trans)
                if cleaned_trans:
                    cleaned_transactions.append(cleaned_trans)
            
            return cleaned_transactions
            
        except json.JSONDecodeError:
            # Try to extract JSON from response if it's embedded in text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    response_data = json.loads(json_match.group())
                    transactions = response_data.get("transactions", [])
                    return [self._clean_transaction(t) for t in transactions if self._clean_transaction(t)]
                except:
                    pass
            
            # Return error transaction if parsing fails
            return [{"error": f"Failed to parse vision response: {response_text[:200]}..."}]
    
    def _clean_transaction(self, transaction: Dict) -> Optional[Dict]:
        """Clean and validate individual transaction data"""
        try:
            # Required fields
            vendor = transaction.get("vendor", "").strip()
            amount = transaction.get("amount")
            
            if not vendor or amount is None:
                return None
            
            # Clean amount
            if isinstance(amount, str):
                # Remove currency symbols and parse
                amount_str = re.sub(r'[^\d.-]', '', amount)
                try:
                    amount = float(amount_str)
                except ValueError:
                    return None
            
            if amount <= 0:
                return None
            
            # Clean date
            date_str = transaction.get("date")
            if date_str and date_str != "null":
                date_str = self._parse_date_string(date_str)
            else:
                date_str = None
            
            # Clean other fields
            description = transaction.get("description", "").strip()
            category = transaction.get("category", "Unknown").strip()
            
            return {
                "date": date_str,
                "vendor": vendor,
                "amount": round(amount, 2),
                "description": description if description else None,
                "category": category if category else "Unknown"
            }
            
        except Exception as e:
            print(f"Error cleaning transaction: {e}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[str]:
        """Parse various date formats to YYYY-MM-DD"""
        if not date_str or date_str.lower() == "null":
            return None
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y', 
            '%d/%m/%Y',
            '%m/%d/%y',
            '%Y/%m/%d',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def batch_process_documents(self, document_files: List[bytes]) -> List[Dict]:
        """Process multiple documents and return combined results"""
        all_transactions = []
        
        for i, doc_bytes in enumerate(document_files):
            try:
                file_type = self.detect_file_type(doc_bytes)
                
                if file_type == "pdf":
                    # PDF processing would need additional libraries
                    all_transactions.append({
                        "error": f"PDF processing not yet implemented for document {i+1}"
                    })
                elif file_type in ["jpeg", "png", "gif", "webp"]:
                    # Process image - this would call the vision model
                    # For now, return placeholder
                    all_transactions.append({
                        "error": f"Vision model processing placeholder for document {i+1}"
                    })
                else:
                    all_transactions.append({
                        "error": f"Unsupported file type '{file_type}' for document {i+1}"
                    })
                    
            except Exception as e:
                all_transactions.append({
                    "error": f"Failed to process document {i+1}: {str(e)}"
                })
        
        return all_transactions
    
    def extract_vendor_patterns(self, text: str) -> List[str]:
        """Extract potential vendor names from text using patterns"""
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # "Company Name"
            r'([A-Z]+\s[A-Z]+)',           # "COMPANY NAME"
            r'([A-Z][a-z]+[A-Z][a-z]+)',   # "CompanyName"
        ]
        
        vendors = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            vendors.extend(matches)
        
        return list(set(vendors))
    
    def extract_amounts(self, text: str) -> List[float]:
        """Extract potential amounts from text"""
        # Pattern for currency amounts
        amount_pattern = r'\$?(\d+\.?\d{0,2})'
        matches = re.findall(amount_pattern, text)
        
        amounts = []
        for match in matches:
            try:
                amount = float(match)
                if amount > 0:
                    amounts.append(amount)
            except ValueError:
                continue
        
        return amounts

# Utility functions for document processing

def validate_document_size(file_bytes: bytes, max_size_mb: int = 10) -> bool:
    """Validate document file size"""
    size_mb = len(file_bytes) / (1024 * 1024)
    return size_mb <= max_size_mb

def get_file_info(file_bytes: bytes) -> Dict[str, any]:
    """Get information about uploaded file"""
    parser = DocumentParser()
    
    return {
        "size_bytes": len(file_bytes),
        "size_mb": len(file_bytes) / (1024 * 1024),
        "file_type": parser.detect_file_type(file_bytes),
        "is_supported": parser.detect_file_type(file_bytes) != "unknown"
    }