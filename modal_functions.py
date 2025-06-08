import modal
from modal import App, Image
import json
import base64
import io
from typing import List, Dict
import pandas as pd

app = App("SmartLedger")

image = (
    Image.debian_slim()
    .pip_install([
        "openai>=1.0.0",
        "pillow>=10.0.0", 
        "pandas>=2.0.0",
        "llama-index>=0.9.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0"
    ])
    .env({"OPENAI_API_KEY": modal.Secret.from_name("openai-secret").get("OPENAI_API_KEY")})
)

@app.cls(image=image, gpu="T4", timeout=300)
class DocumentProcessor:
    def __init__(self):
        import openai
        import os
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @modal.method
    async def process_documents_batch(self, document_files: List[bytes]) -> List[Dict]:
        """Process multiple documents using vision models"""
        results = []

        for doc_file in document_files:
            try:
                if self._is_image_file(doc_file):
                    parsed_data = await self._parse_with_vision(doc_file)
                else:
                    parsed_data = {"error": "Unsupported file type. Please upload images (JPG, PNG) or PDFs."}
                
                if isinstance(parsed_data, list):
                    results.extend(parsed_data)
                else:
                    results.append(parsed_data)
                    
            except Exception as e:
                results.append({"error": f"Failed to process document: {str(e)}"})

        return results

    def _is_image_file(self, file_bytes: bytes) -> bool:
        """Check if file is an image based on magic bytes"""
        if len(file_bytes) < 8:
            return False
        
        # Check common image file signatures
        if file_bytes.startswith(b'\xff\xd8\xff'):  # JPEG
            return True
        elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
            return True
        elif file_bytes.startswith(b'GIF8'):  # GIF
            return True
        elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[:12]:  # WebP
            return True
        
        return False

    async def _parse_with_vision(self, image_bytes: bytes) -> Dict:
        """Extract transaction data using OpenAI Vision API"""
        try:
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode()

            response = await self.client.chat.completions.acreate(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract transaction details from this receipt/invoice image.
                            
                            Return a JSON object with the following structure:
                            {
                                "transactions": [
                                    {
                                        "date": "YYYY-MM-DD",
                                        "vendor": "vendor name",
                                        "amount": 0.00,
                                        "description": "item description",
                                        "category": "best guess category (e.g., Office Supplies, Meals, Travel, etc.)"
                                    }
                                ]
                            }
                            
                            If multiple line items exist, create separate transaction entries.
                            If you cannot clearly read the amount or date, mark those fields as null.
                            Always provide your best guess for vendor name and category.
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }],
                response_format={"type": "json_object"},
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("transactions", [])

        except Exception as e:
            return {"error": f"Vision parsing failed: {str(e)}"}

@app.function(image=image, timeout=600)
def process_documents_modal(document_files: List[bytes], ledger_csv_content: str) -> tuple:
    """Main processing function called from Gradio"""
    try:
        # Process documents with vision models
        processor = DocumentProcessor()
        transactions = processor.process_documents_batch.remote(document_files)

        # Load ledger CSV
        ledger_df = pd.read_csv(io.StringIO(ledger_csv_content))
        
        # Import reconciliation logic
        from llamaindex_core import reconcile_with_llamaindex
        
        # Reconcile using LlamaIndex
        matched, unmatched = reconcile_with_llamaindex(transactions, ledger_df)

        # Generate export CSV
        export_data = _create_export_csv(matched, unmatched)

        return matched, unmatched, export_data

    except Exception as e:
        error_msg = f"Modal processing failed: {str(e)}"
        return [], [{"error": error_msg}], f"date,vendor,amount,status,error\n,,,failed,{error_msg}"

def _create_export_csv(matched: List[Dict], unmatched: List[Dict]) -> str:
    """Create CSV export data for reconciliation results"""
    export_rows = []
    
    # Add header
    export_rows.append("date,vendor,amount,category,status,confidence,suggested_action")
    
    # Add matched transactions
    for transaction in matched:
        date = transaction.get('date', '')
        vendor = transaction.get('vendor', '')
        amount = transaction.get('amount', 0)
        category = transaction.get('category', 'Unknown')
        confidence = transaction.get('confidence', 0.0)
        
        export_rows.append(f"{date},{vendor},{amount},{category},matched,{confidence},none")
    
    # Add unmatched transactions
    for transaction in unmatched:
        date = transaction.get('date', '')
        vendor = transaction.get('vendor', '')
        amount = transaction.get('amount', 0)
        category = transaction.get('category', 'Unknown')
        
        export_rows.append(f"{date},{vendor},{amount},{category},unmatched,0.0,create_entry")
    
    return "\n".join(export_rows)

# Alternative processor for open-source models (optional)
@app.cls(image=image, gpu="A10G", timeout=300)
class OpenSourceProcessor:
    """Alternative processor using open-source vision models"""
    
    def __init__(self):
        # This could use Hugging Face transformers for open-source models
        pass

    @modal.method
    def extract_text_from_image(self, image_bytes: bytes) -> Dict:
        """Extract text using open-source vision model"""
        # Placeholder for open-source implementation
        # Could use models like BLIP, LayoutLM, etc.
        return {"error": "Open-source processor not implemented yet"}

if __name__ == "__main__":
    # Test the app locally
    with app.run():
        print("SmartLedger Modal app is ready!")