from pydantic import BaseModel, Field, validator
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class TransactionStatus(str, Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    PENDING = "pending"
    ERROR = "error"

class ConfidenceLevel(str, Enum):
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.79
    LOW = "low"        # 0.0-0.49

class Transaction(BaseModel):
    """Represents a parsed transaction from a document"""
    date: Optional[date] = None
    vendor: str = Field(..., min_length=1, description="Business or vendor name")
    amount: float = Field(..., gt=0, description="Transaction amount")
    description: Optional[str] = Field(None, description="Transaction description or items")
    category: Optional[str] = Field(None, description="Expense category")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Matching confidence score")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return round(v, 2)
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                try:
                    return datetime.strptime(v, '%m/%d/%Y').date()
                except ValueError:
                    return None
        return v
    
    def get_confidence_level(self) -> ConfidenceLevel:
        if self.confidence is None:
            return ConfidenceLevel.LOW
        elif self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

class LedgerEntry(BaseModel):
    """Represents an entry from the uploaded ledger CSV"""
    date: date
    vendor: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    category: str = Field(..., min_length=1)
    description: Optional[str] = None
    account: Optional[str] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        return round(v, 2)
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            try:
                return datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                try:
                    return datetime.strptime(v, '%m/%d/%Y').date()
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}")
        return v

class MatchedTransaction(Transaction):
    """Transaction that has been matched with a ledger entry"""
    ledger_match: Optional[LedgerEntry] = None
    suggested_category: Optional[str] = None
    similar_transaction: Optional[Dict[str, Any]] = None
    status: TransactionStatus = TransactionStatus.MATCHED
    
    class Config:
        use_enum_values = True

class UnmatchedTransaction(Transaction):
    """Transaction that could not be matched"""
    status: TransactionStatus = TransactionStatus.UNMATCHED
    suggested_categories: Optional[List[str]] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True

class ReconciliationResult(BaseModel):
    """Result of the reconciliation process"""
    matched: List[MatchedTransaction] = []
    unmatched: List[UnmatchedTransaction] = []
    total_processed: int = 0
    total_matched: int = 0
    total_unmatched: int = 0
    processing_errors: List[str] = []
    export_csv: Optional[str] = None
    
    @validator('total_processed', always=True)
    def calculate_total_processed(cls, v, values):
        matched = values.get('matched', [])
        unmatched = values.get('unmatched', [])
        return len(matched) + len(unmatched)
    
    @validator('total_matched', always=True)
    def calculate_total_matched(cls, v, values):
        return len(values.get('matched', []))
    
    @validator('total_unmatched', always=True)
    def calculate_total_unmatched(cls, v, values):
        return len(values.get('unmatched', []))
    
    def get_match_rate(self) -> float:
        """Calculate the percentage of successfully matched transactions"""
        if self.total_processed == 0:
            return 0.0
        return (self.total_matched / self.total_processed) * 100

class DocumentProcessingError(BaseModel):
    """Error information for failed document processing"""
    file_name: Optional[str] = None
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ProcessingStats(BaseModel):
    """Statistics about the processing session"""
    documents_uploaded: int = 0
    documents_processed: int = 0
    transactions_extracted: int = 0
    transactions_matched: int = 0
    processing_time_seconds: Optional[float] = None
    errors: List[DocumentProcessingError] = []
    
    def add_error(self, error_type: str, message: str, file_name: str = None):
        """Add an error to the processing stats"""
        error = DocumentProcessingError(
            file_name=file_name,
            error_type=error_type,
            error_message=message
        )
        self.errors.append(error)

class ExportFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"

class ExportData(BaseModel):
    """Data structure for exporting reconciliation results"""
    format: ExportFormat
    content: str
    filename: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True

# Utility functions for model validation and conversion

def validate_csv_headers(headers: List[str]) -> bool:
    """Validate that CSV has required headers for ledger import"""
    required_headers = {'date', 'vendor', 'amount'}
    header_set = {h.lower().strip() for h in headers}
    return required_headers.issubset(header_set)

def convert_transaction_to_dict(transaction: Transaction) -> Dict[str, Any]:
    """Convert transaction model to dictionary for display"""
    return {
        "date": transaction.date.isoformat() if transaction.date else None,
        "vendor": transaction.vendor,
        "amount": transaction.amount,
        "description": transaction.description,
        "category": transaction.category,
        "confidence": transaction.confidence
    }

def create_export_filename(format: ExportFormat, prefix: str = "smartledger") -> str:
    """Create standardized export filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_reconciliation_{timestamp}.{format.value}"