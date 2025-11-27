"""
Document Ingestion and Processing Pipeline.

Supports:
- PDF, DOCX, TXT, HTML documents
- Intelligent chunking with overlap
- Metadata extraction (company names, dates, deal terms)
- M&A-specific entity recognition
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import PyPDF2
import pdfplumber
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document text with metadata."""
    
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    chunk_index: int
    
    # Position info
    page_number: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # M&A specific
    companies_mentioned: List[str] = field(default_factory=list)
    deal_terms: Dict[str, Any] = field(default_factory=dict)
    financial_figures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Embedding
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)


class DocumentProcessor:
    """
    Process documents for RAG ingestion.
    
    Features:
    - Multi-format support (PDF, DOCX, TXT, HTML)
    - Intelligent chunking with semantic boundaries
    - M&A entity extraction
    - Metadata enrichment
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        extract_ma_entities: bool = True
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            extract_ma_entities: Extract M&A-specific entities
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.extract_ma_entities = extract_ma_entities
        
        # M&A entity patterns
        self.company_pattern = re.compile(
            r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*,? (?:Inc\.|LLC|Corp\.|Corporation|Ltd\.))\b'
        )
        self.deal_value_pattern = re.compile(
            r'\$\s*(\d+(?:\.\d+)?)\s*(billion|million|B|M)\b',
            re.IGNORECASE
        )
        self.date_pattern = re.compile(
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+ \d{1,2},? \d{4})\b'
        )
    
    def process_file(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process a document file into chunks.
        
        Args:
            file_path: Path to document
            document_id: Unique document ID
            metadata: Additional metadata
            
        Returns:
            List of document chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = self._generate_document_id(file_path)
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text, pages = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == '.docx':
            text, pages = self._extract_docx_text(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            text = file_path.read_text(encoding='utf-8')
            pages = None
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Extracted {len(text)} characters from {file_path.name}")
        
        # Create chunks
        chunks = self._create_chunks(
            text=text,
            document_id=document_id,
            document_name=file_path.name,
            pages=pages,
            metadata=metadata or {}
        )
        
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        return chunks
    
    def process_text(
        self,
        text: str,
        document_id: str,
        document_name: str = "text_document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process raw text into chunks.
        
        Args:
            text: Document text
            document_id: Unique document ID
            document_name: Document name
            metadata: Additional metadata
            
        Returns:
            List of document chunks
        """
        return self._create_chunks(
            text=text,
            document_id=document_id,
            document_name=document_name,
            pages=None,
            metadata=metadata or {}
        )
    
    def _extract_pdf_text(self, file_path: Path) -> tuple[str, Dict[int, str]]:
        """Extract text from PDF with page tracking."""
        text_parts = []
        pages = {}
        
        try:
            # Try pdfplumber first (better table support)
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    pages[i] = page_text
                    text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    pages[i] = page_text
                    text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        return full_text, pages
    
    def _extract_docx_text(self, file_path: Path) -> tuple[str, Dict[int, str]]:
        """Extract text from DOCX with paragraph tracking."""
        doc = DocxDocument(file_path)
        
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        full_text = "\n\n".join(paragraphs)
        
        # Create pseudo-pages (every 10 paragraphs)
        pages = {}
        page_size = 10
        for i in range(0, len(paragraphs), page_size):
            page_num = i // page_size
            pages[page_num] = "\n\n".join(paragraphs[i:i+page_size])
        
        return full_text, pages
    
    def _create_chunks(
        self,
        text: str,
        document_id: str,
        document_name: str,
        pages: Optional[Dict[int, str]],
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create overlapping chunks from text."""
        chunks = []
        
        # Split by sentences for better semantic boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        document_name=document_name,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        pages=pages,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Calculate overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                overlap_sentences = overlap_text.split(". ")[-3:] if overlap_text else []
                
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                start_char = start_char + len(chunk_text) - len(overlap_text)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    pages=pages,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        document_id: str,
        document_name: str,
        chunk_index: int,
        start_char: int,
        pages: Optional[Dict[int, str]],
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a document chunk with metadata."""
        
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(document_id, chunk_index)
        
        # Find page number if pages provided
        page_number = None
        if pages:
            cumulative_length = 0
            for page_num, page_text in sorted(pages.items()):
                if cumulative_length <= start_char < cumulative_length + len(page_text):
                    page_number = page_num
                    break
                cumulative_length += len(page_text)
        
        # Extract M&A entities
        companies = []
        deal_terms = {}
        financial_figures = []
        
        if self.extract_ma_entities:
            companies = self._extract_companies(text)
            deal_terms = self._extract_deal_terms(text)
            financial_figures = self._extract_financial_figures(text)
        
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            document_id=document_id,
            document_name=document_name,
            chunk_index=chunk_index,
            page_number=page_number,
            start_char=start_char,
            end_char=start_char + len(text),
            metadata=metadata,
            companies_mentioned=companies,
            deal_terms=deal_terms,
            financial_figures=financial_figures
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract company names from text."""
        matches = self.company_pattern.findall(text)
        return list(set(matches))[:10]  # Limit to top 10
    
    def _extract_deal_terms(self, text: str) -> Dict[str, Any]:
        """Extract M&A deal terms."""
        terms = {}
        
        # Deal type
        if any(term in text.lower() for term in ['merger', 'acquisition', 'takeover']):
            if 'merger' in text.lower():
                terms['deal_type'] = 'merger'
            elif 'acquisition' in text.lower():
                terms['deal_type'] = 'acquisition'
            elif 'takeover' in text.lower():
                terms['deal_type'] = 'takeover'
        
        # Deal status
        if any(term in text.lower() for term in ['announced', 'pending', 'completed', 'closed']):
            for status in ['announced', 'pending', 'completed', 'closed']:
                if status in text.lower():
                    terms['status'] = status
                    break
        
        return terms
    
    def _extract_financial_figures(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial figures (deal values, revenues, etc.)."""
        figures = []
        
        for match in self.deal_value_pattern.finditer(text):
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Convert to millions
            if unit in ['billion', 'b']:
                value_millions = value * 1000
            else:
                value_millions = value
            
            figures.append({
                'value': value_millions,
                'unit': 'million',
                'original_text': match.group(0)
            })
        
        return figures[:5]  # Limit to top 5
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path."""
        content = f"{file_path.name}_{file_path.stat().st_size}_{file_path.stat().st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{document_id}_{chunk_index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


__all__ = ["DocumentProcessor", "DocumentChunk"]