"""
Document parser for extracting transaction data from images using Llama 3.2 Vision Instruct
Interfaces with Modal's vision model deployment for OCR and structured data extraction
"""

import base64
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

class VisionTransactionParser:
    """
    Parser that uses Llama 3.2 Vision Instruct via Modal to extract transaction data from images
    """
    
    def __init__(self, modal_app_name: str = "llama-3.2-11B-Vision-Instruct", api_key: str = "super-secret-key"):
        """
        Initialize the vision parser
        
        Args:
            modal_app_name: Name of the Modal app hosting the vision model
            api_key: API key for Modal vision model access
        """
        self.modal_app_name = modal_app_name
        self.api_key = api_key
        self.base_url = None
        self._setup_modal_connection()
    
    def _setup_modal_connection(self):
        """Setup connection to Modal vision model"""
        try:
            import modal
            # Get the Modal app URL
            workspace = modal.config._profile
            environment = modal.config.config.get("environment", "")
            prefix = workspace + (f"-{environment}" if environment else "")
            
            self.base_url = f"https://{prefix}--{self.modal_app_name}-serve.modal.run/v1"
            print(f"🔗 Connected to Modal vision model at: {self.base_url}")
            
        except Exception as e:
            print(f"⚠️ Could not setup Modal connection: {e}")
            self.base_url = None
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API transmission"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
    
    def _create_vision_prompt(self) -> str:
        """Create optimized prompt for transaction extraction"""
        return """You are a financial document analyzer. Extract transaction data from this bank statement, receipt, or financial document.

For each transaction you find, return ONLY a valid JSON array with this exact format:

[
  {
    "date": "YYYY-MM-DD",
    "amount": 123.45,
    "vendor": "Vendor Name",
    "description": "Transaction description",
    "type": "debit" or "credit"
  }
]

Rules:
- Extract ALL transactions visible in the image
- Use negative amounts for debits/expenses, positive for credits/income
- Parse dates to YYYY-MM-DD format
- Clean vendor names (remove extra spaces, standardize)
- Include meaningful descriptions
- Return ONLY the JSON array, no other text
- If no transactions found, return: []

Analyze the image and extract all transaction data:"""

    def _call_vision_model(self, image_base64: str, prompt: str) -> str:
        """Call the Modal vision model API"""
        if not self.base_url:
            raise ConnectionError("Modal vision model not available")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1  # Low temperature for consistent extraction
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Vision model call failed: {e}")
    
    def _parse_vision_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the vision model response into structured transaction data"""
        try:
            # Clean the response - sometimes models add extra text
            response_text = response_text.strip()
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print(f"⚠️ No JSON array found in response: {response_text[:200]}...")
                return []
            
            json_str = response_text[start_idx:end_idx]
            transactions = json.loads(json_str)
            
            # Validate and clean transaction data
            cleaned_transactions = []
            for txn in transactions:
                if self._validate_transaction(txn):
                    cleaned_transactions.append(self._clean_transaction(txn))
            
            return cleaned_transactions
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            print(f"❌ Error parsing vision response: {e}")
            return []
    
    def _validate_transaction(self, txn: Dict[str, Any]) -> bool:
        """Validate that transaction has required fields"""
        required_fields = ["date", "amount", "vendor", "description"]
        
        for field in required_fields:
            if field not in txn or txn[field] is None:
                print(f"⚠️ Transaction missing required field '{field}': {txn}")
                return False
        
        # Validate amount is numeric
        try:
            float(txn["amount"])
        except (ValueError, TypeError):
            print(f"⚠️ Invalid amount in transaction: {txn}")
            return False
        
        return True
    
    def _clean_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize transaction data"""
        cleaned = {
            "date": str(txn["date"]).strip(),
            "amount": float(txn["amount"]),
            "vendor": str(txn["vendor"]).strip(),
            "description": str(txn["description"]).strip(),
            "type": txn.get("type", "debit").strip().lower(),
            "source": "image_extraction",
            "extracted_at": datetime.now().isoformat()
        }
        
        # Standardize vendor names
        cleaned["vendor"] = self._standardize_vendor_name(cleaned["vendor"])
        
        return cleaned
    
    def _standardize_vendor_name(self, vendor: str) -> str:
        """Standardize vendor names for better matching"""
        # Remove common suffixes/prefixes
        vendor = vendor.upper()
        
        # Remove common business suffixes
        suffixes = ["INC", "LLC", "LTD", "CORP", "CO", "&", "AND"]
        words = vendor.split()
        cleaned_words = [w for w in words if w not in suffixes]
        
        return " ".join(cleaned_words).title()
    
    def extract_transactions_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extract transaction data from an image file
        
        Args:
            image_path: Path to the image file (bank statement, receipt, etc.)
            
        Returns:
            Dictionary containing extracted transactions and metadata
        """
        try:
            print(f"📷 Processing image: {image_path}")
            
            # Encode image
            image_base64 = self._encode_image(image_path)
            
            # Create prompt
            prompt = self._create_vision_prompt()
            
            # Call vision model
            print("🧠 Calling Llama 3.2 Vision model...")
            response = self._call_vision_model(image_base64, prompt)
            
            # Parse response
            transactions = self._parse_vision_response(response)
            
            print(f"✅ Extracted {len(transactions)} transactions from image")
            
            return {
                "status": "success",
                "transactions": transactions,
                "total_transactions": len(transactions),
                "image_path": image_path,
                "extracted_at": datetime.now().isoformat(),
                "raw_response": response[:500] if len(response) > 500 else response  # Truncated for debugging
            }
            
        except Exception as e:
            print(f"❌ Error extracting transactions: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transactions": [],
                "total_transactions": 0,
                "image_path": image_path,
                "extracted_at": datetime.now().isoformat()
            }
    
    def extract_transactions_from_bytes(self, image_bytes: bytes, filename: str = "uploaded_image") -> Dict[str, Any]:
        """
        Extract transaction data from image bytes (for Gradio file upload)
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename for reference
            
        Returns:
            Dictionary containing extracted transactions and metadata
        """
        try:
            print(f"📷 Processing uploaded image: {filename}")
            
            # Encode bytes to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create prompt
            prompt = self._create_vision_prompt()
            
            # Call vision model
            print("🧠 Calling Llama 3.2 Vision model...")
            response = self._call_vision_model(image_base64, prompt)
            
            # Parse response
            transactions = self._parse_vision_response(response)
            
            print(f"✅ Extracted {len(transactions)} transactions from uploaded image")
            
            return {
                "status": "success",
                "transactions": transactions,
                "total_transactions": len(transactions),
                "filename": filename,
                "extracted_at": datetime.now().isoformat(),
                "raw_response": response[:500] if len(response) > 500 else response
            }
            
        except Exception as e:
            print(f"❌ Error extracting transactions from bytes: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transactions": [],
                "total_transactions": 0,
                "filename": filename,
                "extracted_at": datetime.now().isoformat()
            }


def test_vision_parser():
    """Test function for the vision parser"""
    parser = VisionTransactionParser()
    
    # Test with a sample image (you would need to provide an actual image)
    # result = parser.extract_transactions_from_image("/path/to/test/image.jpg")
    # print(json.dumps(result, indent=2))
    
    print("✅ Vision parser initialized successfully")
    print(f"Modal base URL: {parser.base_url}")


if __name__ == "__main__":
    test_vision_parser()